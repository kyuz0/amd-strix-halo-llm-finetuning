#!/usr/bin/env python3
import sys
import os
import time
import json
import argparse
import subprocess
import re
from pathlib import Path

MODELS = [
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it"
]

TRAIN_TYPES = ["full", "lora", "qlora-8bit", "qlora-4bit"]

# Default to ~/finetuning-workspace/benchmarks.json
RESULTS_FILE = Path.home() / "finetuning-workspace" / "benchmarks.json"

def load_results():
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r") as f:
            try:
                return json.load(f)
            except:
                return {}
    return {}

def save_result(model, train_type, time_sec, peak_mem_gb=None, status="success"):
    results = load_results()
    if model not in results:
        results[model] = {}
    results[model][train_type] = {
        "time_sec": time_sec,
        "peak_mem_gb": peak_mem_gb,
        "status": status
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)

def format_time(seconds):
    if seconds is None:
        return "N/A"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m}m"
    return f"{m}m{s}s"

def print_table():
    results = load_results()
    print("\n" + "="*80)
    print(f"{'Model':<25} | {'Full FT':<15} | {'LoRA':<15} | {'8-bit + LoRA':<15} | {'QLoRA':<15}")
    print("-" * 80)
    for model in MODELS:
        row = [f"{model.split('/')[-1]:<25}"]
        for t in TRAIN_TYPES:
            res = results.get(model, {}).get(t, {})
            if res.get("status") == "success":
                mem = f"{res.get('peak_mem_gb')}GB / " if res.get('peak_mem_gb') else ""
                time_str = format_time(res.get('time_sec'))
                cell = f"{mem}{time_str}"
            elif res.get("status") in ["OOM", "failed"]:
                cell = res.get("status")
            else:
                cell = "-"
            row.append(f"{cell:<15}")
        print(" | ".join(row))
    print("="*80 + "\n")

    
def launch_training(head_ip, worker_ip, rdma_iface, model, train_type, epochs, batch_size):
    workspace_dir = Path("/opt/workspace")
    output_dir = Path.home() / "finetuning-workspace" / "benchmark_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    accel_base = [
        "accelerate", "launch",
        "--config_file", f"{workspace_dir}/accelerate_fsdp.yaml",
        "--num_machines", "2",
        "--main_process_ip", head_ip,
        "--main_process_port", "29500"
    ]
    
    train_script = [
        f"{workspace_dir}/distributed_train.py", 
        "--model", model, 
        "--type", train_type,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--output_dir", str(output_dir)
    ]
    
    worker_cmd = accel_base + ["--machine_rank", "1"] + train_script
    worker_cmd_str = " ".join(worker_cmd)
    
    ssh_cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no", worker_ip,
        f"toolbox run -c strix-halo-llm-finetuning -- bash -c 'cd {workspace_dir} && NCCL_SOCKET_IFNAME={rdma_iface} GLOO_SOCKET_IFNAME={rdma_iface} {worker_cmd_str}'"
    ]
    
    print(f"[{model}] Launching {train_type} Worker Process on {worker_ip}...")
    worker_process = subprocess.Popen(ssh_cmd)
    time.sleep(3)
    
    head_cmd = accel_base + ["--machine_rank", "0"] + train_script
    env = os.environ.copy()
    env["NCCL_SOCKET_IFNAME"] = rdma_iface
    env["GLOO_SOCKET_IFNAME"] = rdma_iface
    
    print(f"[{model}] Launching {train_type} Head Process on {head_ip}...")
    start_time = time.time()
    peak_mem = None
    status = "success"
    
    try:
        # Run head process, capture stdout and stderr to get memory footprint and errors
        head_process = subprocess.Popen(
            head_cmd, env=env, cwd=workspace_dir, 
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        
        for line in head_process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            
            # e.g., Peak training memory qlora-4bit-fsdp: 12.34 GB
            mem_match = re.search(r"Peak training memory.*?: ([\d\.]+) GB", line)
            if mem_match:
                peak_mem = float(mem_match.group(1))
            
            if "OutOfMemoryError" in line or "CUDA out of memory" in line:
                status = "OOM"

        head_process.wait()
        
        if head_process.returncode != 0 and status != "OOM":
            status = "failed"
            
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
        status = "interrupted"
    except Exception as e:
        print(f"\nError running training: {e}")
        status = "failed"
    finally:
        print(f"[{model}] Cleaning up processes for {train_type}...")
        worker_process.terminate()
        try:
            worker_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            worker_process.kill()
        subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", worker_ip, "pkill -f accelerate"], stderr=subprocess.DEVNULL)
        
    duration = time.time() - start_time
    return status, duration, peak_mem

def main():
    parser = argparse.ArgumentParser(description="Automated Gemma Model Benchmark for Strix Halo Cluster")
    parser.add_argument("--head_ip", default="192.168.100.1")
    parser.add_argument("--worker_ip", default="192.168.100.2")
    parser.add_argument("--rdma_iface", default="auto", help="Use 'auto' to auto-detect based on head_ip")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    
    # Auto-detect Network Interface
    if args.rdma_iface == "auto":
        parts = args.head_ip.split('.')
        subnet = f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"
        try:
            iface_cmd = f"ip -o addr show to {subnet} | awk '{{print $2}}' | head -n1"
            rdma_iface = subprocess.check_output(iface_cmd, shell=True, text=True).strip()
            print(f"Auto-detected RDMA Interface: {rdma_iface}")
        except:
            rdma_iface = "eth0"
            print("Failed to auto-detect interface. Defaulting to eth0")
    else:
        rdma_iface = args.rdma_iface

    # Ensure result dir exists
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("\n--- Current Benchmark State ---")
    print_table()
    
    for model in MODELS:
        for t_type in TRAIN_TYPES:
            results = load_results()
            res = results.get(model, {}).get(t_type, {})
            
            # Skip if already successfully completed or definitively failed (like OOM)
            if res.get("status") in ["success", "OOM"]:
                print(f"Skipping {model} / {t_type} - already recorded with status: {res.get('status')}")
                continue
                
            print(f"\n[{model}] -> Starting benchmark for type: {t_type}")
            status, duration, peak_mem = launch_training(
                args.head_ip, args.worker_ip, rdma_iface, model, t_type, args.epochs, args.batch_size
            )
            
            if status == "interrupted":
                print("Exiting benchmark suite.")
                sys.exit(1)
                
            save_result(model, t_type, duration, peak_mem, status)
            
            # Print updated table after each run
            print_table()

if __name__ == "__main__":
    main()
