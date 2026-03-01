#!/usr/bin/env python3
"""
Cluster benchmark: tests viable model×type×strategy×accum configs on 2-node cluster.

Logic:
  - DDP preferred over FSDP (less overhead)
  - Only use FSDP when DDP can't fit the model
  - Test each viable config with accum=1, then accum=4
  - Launches via torchrun + SSH (same as the launcher script)

Usage:
  python3 benchmark_configs.py                    # estimate only (safe, no training)
  python3 benchmark_configs.py --run              # run each config on the cluster
  python3 benchmark_configs.py --world-size 1     # single node estimates only
"""
import argparse
import json
import os
import subprocess
import sys
import time

# ── Memory Estimation ──────────────────────────────────────────────────
MODEL_PARAMS = {
    "google/gemma-3-1b-it": 1.0,
    "google/gemma-3-4b-it": 4.0,
    "google/gemma-3-12b-it": 12.0,
    "google/gemma-3-27b-it": 27.0,
}

# Calibrated from measure_unsloth_memory.py on Strix Halo (Gemma 1B + 4B).
# Discount = unsloth_peak / hf_peak. Values <1 = Unsloth uses less memory.
UNSLOTH_DISCOUNT = {
    "full": 0.29,        # Unsloth uses ~29% of HF memory
    "lora": 0.39,        # Unsloth uses ~39% of HF memory
    "8bit-lora": 0.39,   # conservative: same as lora (not yet profiled)
    "qlora": 1.30,       # Unsloth actually uses ~30% MORE for qlora
}

def estimate_memory_gb(model_name, train_type, strategy, batch_size, max_length, world_size, use_unsloth=False):
    """Estimate peak GPU memory in GB per node."""
    params_b = MODEL_PARAMS.get(model_name, 1.0)
    params = params_b * 1e9
    shard = world_size if strategy == "fsdp" else 1

    if train_type == "full":
        weights = params * 2
        grads = params * 2
        optim = params * 4 * 2
    elif train_type == "lora":
        weights = params * 2
        t = params * 0.02
        grads, optim = t * 2, t * 4 * 2
    elif train_type == "8bit-lora":
        weights = params * 1
        t = params * 0.02
        grads, optim = t * 2, t * 4 * 2
    elif train_type == "qlora":
        weights = params * 0.5
        t = params * 0.02
        grads, optim = t * 2, t * 4 * 2
    else:
        return 999

    state = (weights + grads + optim) / shard / 1e9
    if strategy == "fsdp":
        state += (weights * 0.1) / 1e9

    hidden = {1.0: 1024, 4.0: 2048, 12.0: 3840, 27.0: 5184}.get(params_b, 2048)
    layers = {1.0: 26, 4.0: 34, 12.0: 48, 27.0: 62}.get(params_b, 32)
    act_per_layer = (batch_size * max_length * hidden * 4) / 1e9
    use_ckpt = strategy == "fsdp" or train_type in ("8bit-lora", "qlora")
    activations = act_per_layer * (layers ** 0.5 if use_ckpt else layers)

    base = state + activations + 2.0
    if use_unsloth:
        discount = UNSLOTH_DISCOUNT.get(train_type, 1.0)
        return base * discount
    return base


def pick_strategy(model, train_type, batch, max_length, world_size, margin, use_unsloth=False):
    """Pick best strategy: DDP if it fits, else FSDP if it fits, else None."""
    available = 120
    safe = available * margin

    ddp_est = estimate_memory_gb(model, train_type, "ddp", batch, max_length, world_size, use_unsloth)
    if ddp_est <= safe:
        return "ddp", ddp_est

    if world_size > 1 and not use_unsloth:  # Unsloth doesn't support FSDP
        fsdp_est = estimate_memory_gb(model, train_type, "fsdp", batch, max_length, world_size)
        if fsdp_est <= safe:
            return "fsdp", fsdp_est

    return None, ddp_est


# ── Networking helpers (same as start-finetuning-cluster.py) ───────────
def get_subnet_from_ip(ip):
    parts = ip.split(".")
    return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"

def get_dynamic_iface(subnet):
    try:
        cmd = f"ip -o addr show to {subnet} | awk '{{print $2}}' | head -n1"
        iface = subprocess.check_output(cmd, shell=True, text=True).strip()
        if iface:
            return iface
    except Exception:
        pass
    return "eth0"


def run_multinode(train_script, train_args, head_ip, worker_ip, worker_script, timeout=600):
    """Launch a multi-node training run via torchrun + SSH. Returns (returncode, elapsed, stdout, stderr)."""
    subnet = get_subnet_from_ip(head_ip)
    iface = get_dynamic_iface(subnet)

    # Worker SSH command
    worker_env = f"""
    export RDMA_IFACE={iface}
    export NCCL_SOCKET_IFNAME={iface}
    export GLOO_SOCKET_IFNAME={iface}
    export NCCL_IB_TIMEOUT=23
    export NCCL_IB_RETRY_CNT=7
    export NCCL_IB_DISABLE=0
    """
    worker_dir = os.path.dirname(worker_script)
    worker_cmd = f"""
    {worker_env}
    cd {worker_dir}
    torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr={head_ip} --master_port=12345 {worker_script} {train_args}
    """

    hf_token_opt = ""
    if "HF_TOKEN" in os.environ:
        hf_token_opt = f"export HF_TOKEN={os.environ['HF_TOKEN']}; "

    ssh_cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no", worker_ip,
        f"toolbox run -c strix-halo-llm-finetuning -- bash -c '{hf_token_opt}{worker_cmd}'"
    ]

    # Launch worker in background
    worker_proc = subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(3)

    # Launch head
    head_env = os.environ.copy()
    head_env["RDMA_IFACE"] = iface
    head_env["NCCL_SOCKET_IFNAME"] = iface
    head_env["GLOO_SOCKET_IFNAME"] = iface
    head_env["NCCL_IB_TIMEOUT"] = "23"
    head_env["NCCL_IB_RETRY_CNT"] = "7"
    head_env["NCCL_IB_DISABLE"] = "0"

    head_cmd = [
        "torchrun", "--nproc_per_node=1", "--nnodes=2", "--node_rank=0",
        f"--master_addr={head_ip}", "--master_port=12345", train_script
    ] + train_args.split()

    start = time.time()
    try:
        head_result = subprocess.run(
            head_cmd, env=head_env, capture_output=True, text=True, timeout=timeout
        )
        elapsed = time.time() - start
        worker_proc.terminate()
        worker_proc.wait(timeout=10)
        return head_result.returncode, elapsed, head_result.stdout, head_result.stderr
    except subprocess.TimeoutExpired:
        worker_proc.terminate()
        worker_proc.wait(timeout=10)
        return -1, timeout, "", "TIMEOUT"


def run_singlenode(train_script, train_args, timeout=600):
    """Launch a single-node training run. Returns (returncode, elapsed, stdout, stderr)."""
    cmd = [sys.executable, train_script] + train_args.split()
    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        elapsed = time.time() - start
        return result.returncode, elapsed, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, timeout, "", "TIMEOUT"


def main():
    parser = argparse.ArgumentParser(description="Benchmark viable training configurations")
    parser.add_argument("--models", nargs="*",
                        default=["google/gemma-3-1b-it", "google/gemma-3-4b-it",
                                 "google/gemma-3-12b-it", "google/gemma-3-27b-it"])
    parser.add_argument("--types", nargs="*", default=["full", "lora", "qlora"])
    parser.add_argument("--batches", nargs="*", type=int, default=[1, 4])
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--world-size", type=int, default=None,
                        help="Override world size (default: env WORLD_SIZE or 2)")
    parser.add_argument("--head-ip", type=str, default=None,
                        help="Head node IP (default: env VLLM_HEAD_IP or 192.168.100.1)")
    parser.add_argument("--worker-ip", type=str, default=None,
                        help="Worker node IP (default: env VLLM_WORKER_IP or 192.168.100.2)")
    parser.add_argument("--margin", type=float, default=0.85)
    parser.add_argument("--run", action="store_true",
                        help="Actually launch each config on the cluster for 1 epoch")
    parser.add_argument("--rerun", action="store_true",
                        help="Re-run all configs, ignoring previous results")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Timeout per run in seconds (default: 600)")
    parser.add_argument("--results-file", type=str, default="benchmark_results.json",
                        help="Results file (default: benchmark_results.json)")
    parser.add_argument("--unsloth", action="store_true",
                        help="Benchmark with Unsloth optimizations enabled")
    args = parser.parse_args()

    world_size = args.world_size or int(os.environ.get("WORLD_SIZE", "2"))
    head_ip = args.head_ip or os.environ.get("VLLM_HEAD_IP", "192.168.100.1")
    worker_ip = args.worker_ip or os.environ.get("VLLM_WORKER_IP", "192.168.100.2")
    accums = [1, 4]
    use_unsloth = args.unsloth

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, "train.py")
    worker_train_script = "/opt/workspace/train.py"

    # ── Phase 1: Build test plan ───────────────────────────────────
    plan = []
    skipped = []

    for model in args.models:
        for train_type in args.types:
            for batch in args.batches:
                strategy, est = pick_strategy(
                    model, train_type, batch, args.max_length, world_size, args.margin, use_unsloth
                )
                if strategy is None:
                    skipped.append((model, train_type, batch, est))
                    continue
                if strategy == "fsdp" and train_type in ("8bit-lora", "qlora"):
                    skipped.append((model, train_type, batch, est))
                    continue
                # Unsloth does not support FSDP
                if use_unsloth and strategy == "fsdp":
                    skipped.append((model, train_type, batch, est))
                    continue
                for accum in accums:
                    plan.append((model, train_type, strategy, batch, accum, est))

    # ── Phase 2: Print plan ────────────────────────────────────────
    mode_str = f"Multi-Node ({head_ip} ↔ {worker_ip})" if world_size > 1 else "Single Node"
    unsloth_str = " [Unsloth]" if use_unsloth else ""
    print("=" * 95)
    print(f"BENCHMARK PLAN — {mode_str}{unsloth_str}")
    print(f"world_size={world_size}, max_length={args.max_length}, accum={accums}")
    print(f"Rule: DDP preferred → FSDP only when DDP can't fit → skip if neither fits")
    print("=" * 95)
    print(f"{'#':<4} {'Model':<16} {'Type':<10} {'Strategy':<8} {'Batch':<6} {'Accum':<6} {'Est GB':<8}")
    print("-" * 95)

    for i, (model, tt, strat, batch, accum, est) in enumerate(plan, 1):
        name = model.split("/")[-1]
        print(f"{i:<4} {name:<16} {tt:<10} {strat:<8} {batch:<6} {accum:<6} {est:<7.0f}G")

    if skipped:
        print(f"\n⚠️  Skipped {len(skipped)} configs (would OOM):")
        for model, tt, batch, est in skipped:
            name = model.split("/")[-1]
            print(f"   {name} {tt} batch={batch} → needs ~{est:.0f}GB")

    print(f"\nTotal configs to benchmark: {len(plan)}")

    if not args.run:
        print("\nAdd --run to actually launch training on the cluster.")
        return

    # ── Phase 3: Load previous results & skip completed ────────────
    prev_results = []
    completed_keys = set()
    if not args.rerun and os.path.exists(args.results_file):
        try:
            with open(args.results_file) as f:
                prev_results = json.load(f)
            for r in prev_results:
                unsloth_key = "unsloth" if r.get("unsloth", False) else "hf"
                key = f"{r['model']}|{r['type']}|{r['strategy']}|{r['batch']}|{r['accum']}|{unsloth_key}"
                completed_keys.add(key)
            print(f"\n📂 Loaded {len(prev_results)} previous results from {args.results_file}")
        except (json.JSONDecodeError, KeyError):
            print(f"\n⚠️  Could not parse {args.results_file}, starting fresh")
            prev_results = []

    # Filter plan to only new configs
    remaining = []
    skipped_done = 0
    for entry in plan:
        model, tt, strat, batch, accum, est = entry
        name = model.split("/")[-1]
        unsloth_key = "unsloth" if use_unsloth else "hf"
        key = f"{name}|{tt}|{strat}|{batch}|{accum}|{unsloth_key}"
        if key in completed_keys:
            skipped_done += 1
        else:
            remaining.append(entry)

    if skipped_done > 0:
        print(f"⏭️  Skipping {skipped_done} already-benchmarked configs")
    if not remaining:
        print("\n✅ All configs already benchmarked! Use --rerun to re-run.")
        _print_summary(prev_results)
        return

    print(f"🔄 {len(remaining)} configs to run")

    # ── Phase 4: Run benchmarks ────────────────────────────────────
    all_results = list(prev_results)

    print(f"\n{'='*95}")
    print(f"RUNNING BENCHMARKS — {mode_str}")
    print(f"Each config runs 1 epoch. Timeout: {args.timeout}s")
    print(f"{'='*95}")

    for i, (model, tt, strat, batch, accum, est) in enumerate(remaining, 1):
        name = model.split("/")[-1]
        label = f"[{i}/{len(remaining)}] {name} {tt} {strat} batch={batch} accum={accum}"
        if use_unsloth:
            label += " [unsloth]"
        print(f"\n{'─'*80}")
        print(f"▶ {label} (est: {est:.0f}GB)")

        train_args = (f"--model {model} --type {tt} --strategy {strat} "
                      f"--batch-size {batch} --gradient-accumulation {accum} "
                      f"--epochs 1 --max-length {args.max_length}")
        if use_unsloth:
            train_args += " --unsloth"

        if world_size > 1:
            rc, elapsed, stdout, stderr = run_multinode(
                train_script, train_args, head_ip, worker_ip,
                worker_train_script, timeout=args.timeout
            )
        else:
            rc, elapsed, stdout, stderr = run_singlenode(
                train_script, train_args, timeout=args.timeout
            )

        result = {"model": name, "type": tt, "strategy": strat,
                  "batch": batch, "accum": accum, "est_gb": round(est, 1),
                  "unsloth": use_unsloth}

        if rc == 0:
            time_lines = [l for l in stdout.split("\n") if "Training time:" in l]
            peak_lines = [l for l in stdout.split("\n") if "Peak GPU" in l]
            loss_lines = [l for l in stdout.split("\n") if "'loss'" in l]

            print(f"  ✅ SUCCESS ({elapsed:.1f}s)")
            for l in time_lines[-1:]:
                print(f"  {l.strip()}")
            for l in peak_lines[-1:]:
                print(f"  {l.strip()}")
            for l in loss_lines[-1:]:
                print(f"  {l.strip()}")

            result.update({"status": "OK", "time_s": round(elapsed, 1)})
        elif rc == -1:
            print(f"  ⏰ TIMEOUT (>{args.timeout}s)")
            result.update({"status": "TIMEOUT", "time_s": args.timeout})
        else:
            is_oom = "OutOfMemoryError" in stderr or "out of memory" in stderr.lower()
            status = "OOM" if is_oom else "FAIL"
            print(f"  ❌ {status} ({elapsed:.1f}s)")
            print(f"  {stderr[-200:]}")
            result.update({"status": status, "time_s": round(elapsed, 1)})

        # Save after each run
        all_results.append(result)
        with open(args.results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  💾 Saved ({len(all_results)} total results in {args.results_file})")

    _print_summary(all_results)


def _print_summary(results):
    """Print final summary table."""
    print(f"\n{'='*105}")
    print("BENCHMARK RESULTS")
    print(f"{'='*105}")
    print(f"{'Model':<16} {'Type':<10} {'Strategy':<8} {'Batch':<6} {'Accum':<6} {'Est GB':<8} {'Time':<8} {'Status':<8} {'Engine':<8}")
    print("-" * 105)
    for r in results:
        engine = "unsloth" if r.get("unsloth", False) else "hf"
        print(f"{r['model']:<16} {r['type']:<10} {r['strategy']:<8} {r['batch']:<6} "
              f"{r['accum']:<6} {r['est_gb']:<7.0f}G {r['time_s']:<7.1f}s {r['status']:<8} {engine}")

    ok = [r for r in results if r["status"] == "OK"]
    fail = [r for r in results if r["status"] != "OK"]
    print(f"\n✅ {len(ok)} passed | ❌ {len(fail)} failed")


if __name__ == "__main__":
    main()
