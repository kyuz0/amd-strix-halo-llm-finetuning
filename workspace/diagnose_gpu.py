#!/usr/bin/env python3
"""
Diagnostic script — run on EACH node to check where compute actually happens.

Usage:
  Single node:  python3 diagnose_gpu.py
  Multi-node:   torchrun --nnodes=2 --nproc_per_node=1 ... diagnose_gpu.py

This script checks:
  1. Is a GPU actually visible to PyTorch?
  2. Does a tensor on "cuda:0" actually compute on the GPU?
  3. Does model.to("cuda:0") actually use GPU compute units?
  4. Where do the Trainer's model parameters end up?
  5. Timing breakdown: forward vs backward vs idle
"""
import torch
import os
import time

def main():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    print(f"\n{'='*60}")
    print(f"RANK {rank} / WORLD_SIZE {world_size} / LOCAL_RANK {local_rank}")
    print(f"{'='*60}")

    # ── 1. GPU Visibility ──────────────────────────────────────────
    print(f"\n[1] GPU VISIBILITY")
    print(f"  torch.cuda.is_available()     = {torch.cuda.is_available()}")
    print(f"  torch.cuda.device_count()     = {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"  torch.cuda.current_device()   = {torch.cuda.current_device()}")
        print(f"  torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"  total_memory                  = {props.total_memory / 1e9:.2f} GB")
        print(f"  gcnArchName                   = {props.gcnArchName}")
    else:
        print("  ERROR: No CUDA/ROCm GPU found!")
        return

    # ── 2. Raw GPU Compute Test ────────────────────────────────────
    print(f"\n[2] RAW GPU COMPUTE TEST")
    torch.cuda.set_device(local_rank)

    # Create tensors on GPU and do matmul
    a = torch.randn(2048, 2048, device=f"cuda:{local_rank}", dtype=torch.float32)
    b = torch.randn(2048, 2048, device=f"cuda:{local_rank}", dtype=torch.float32)

    # Warmup
    for _ in range(3):
        _ = torch.mm(a, b)
    torch.cuda.synchronize()

    # Timed GPU matmul
    start = time.perf_counter()
    for _ in range(20):
        c = torch.mm(a, b)
    torch.cuda.synchronize()
    gpu_time = time.perf_counter() - start
    print(f"  GPU matmul 2048x2048 x20:     {gpu_time*1000:.1f} ms")
    print(f"  Result device:                {c.device}")
    print(f"  Result sum (sanity):          {c.sum().item():.2f}")

    # Timed CPU matmul for comparison
    a_cpu = a.cpu()
    b_cpu = b.cpu()
    start = time.perf_counter()
    for _ in range(20):
        c_cpu = torch.mm(a_cpu, b_cpu)
    cpu_time = time.perf_counter() - start
    print(f"  CPU matmul 2048x2048 x20:     {cpu_time*1000:.1f} ms")
    print(f"  GPU/CPU speedup:              {cpu_time/gpu_time:.1f}x")

    if gpu_time >= cpu_time * 0.8:
        print(f"  ⚠️  WARNING: GPU is NOT faster than CPU! Compute may be falling back.")
    else:
        print(f"  ✅ GPU compute is working and faster than CPU.")

    # ── 3. Model Loading Test ──────────────────────────────────────
    print(f"\n[3] MODEL LOADING TEST (small model)")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load model like the training script does — no device_map
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-1b-it", torch_dtype=torch.bfloat16, attn_implementation="eager"
    )

    print(f"  After from_pretrained():")
    first_param = next(model.parameters())
    print(f"    First param device: {first_param.device}")
    print(f"    First param dtype:  {first_param.dtype}")

    # Now check what happens with trainer's .to() call
    model = model.to(f"cuda:{local_rank}")
    first_param = next(model.parameters())
    print(f"  After model.to('cuda:{local_rank}'):")
    print(f"    First param device: {first_param.device}")

    # Count params by device
    device_counts = {}
    for name, p in model.named_parameters():
        d = str(p.device)
        device_counts[d] = device_counts.get(d, 0) + 1
    print(f"  Parameter devices: {device_counts}")

    # ── 4. Forward Pass Test ───────────────────────────────────────
    print(f"\n[4] FORWARD PASS TIMING")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    inputs = tokenizer("Hello world, this is a test", return_tensors="pt")
    inputs = {k: v.to(f"cuda:{local_rank}") for k, v in inputs.items()}
    print(f"  Input device: {inputs['input_ids'].device}")

    # Warmup
    with torch.no_grad():
        _ = model(**inputs)
    torch.cuda.synchronize()

    # Timed forward
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        out = model(**inputs)
    torch.cuda.synchronize()
    fwd_time = time.perf_counter() - start
    print(f"  Forward pass time:            {fwd_time*1000:.1f} ms")
    print(f"  Output device:                {out.logits.device}")

    # Timed forward+backward
    model.train()
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = model(**inputs, labels=inputs["input_ids"])
    loss = out.loss
    loss.backward()
    torch.cuda.synchronize()
    fwdbwd_time = time.perf_counter() - start
    print(f"  Forward+Backward time:        {fwdbwd_time*1000:.1f} ms")
    print(f"  Loss device:                  {loss.device}")
    print(f"  Loss value:                   {loss.item():.4f}")

    # Check gradient devices
    grad_devices = set()
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_devices.add(str(p.grad.device))
    print(f"  Gradient devices:             {grad_devices}")

    # ── 5. GPU Memory Usage ────────────────────────────────────────
    print(f"\n[5] GPU MEMORY")
    print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"  Reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB")
    print(f"  Max alloc: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # ── 6. NCCL Distributed Test (if multi-node) ──────────────────
    if world_size > 1:
        print(f"\n[6] DISTRIBUTED ALL-REDUCE TEST")
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")

        tensor = torch.ones(1024, 1024, device=f"cuda:{local_rank}") * (rank + 1)
        print(f"  Before all-reduce: tensor[0,0] = {tensor[0,0].item()}")

        torch.cuda.synchronize()
        start = time.perf_counter()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        ar_time = time.perf_counter() - start

        print(f"  After all-reduce:  tensor[0,0] = {tensor[0,0].item()} (expect {world_size * (world_size + 1) / 2})")
        print(f"  All-reduce time (4MB):        {ar_time*1000:.1f} ms")
        print(f"  Result device:                {tensor.device}")

        # Test larger all-reduce (simulating gradient sync)
        big = torch.ones(64 * 1024 * 1024, device=f"cuda:{local_rank}")
        torch.cuda.synchronize()
        start = time.perf_counter()
        dist.all_reduce(big, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        big_time = time.perf_counter() - start
        print(f"  All-reduce time (256MB):      {big_time*1000:.1f} ms")
        effective_bw = (256 * 2) / big_time / 1000  # 2x for ring all-reduce
        print(f"  Effective bandwidth:          {effective_bw:.1f} GB/s")

        dist.destroy_process_group()
    else:
        print(f"\n[6] DISTRIBUTED TEST: skipped (single node)")

    print(f"\n{'='*60}")
    print(f"DIAGNOSTICS COMPLETE for rank {rank}")
    print(f"{'='*60}\n")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
