#!/usr/bin/env python3
import sys
import os
import shutil
import tempfile
import subprocess
import time
from pathlib import Path

def check_dependencies():
    missing = []
    if not shutil.which("dialog"):
        missing.append("dialog")
    if not shutil.which("ssh"):
        missing.append("ssh")
        
    if missing:
        print(f"Error: Missing dependencies: {', '.join(missing)}.")
        print("Please install them (e.g., sudo dnf install dialog openssh-clients).")
        sys.exit(1)

def run_dialog(args):
    """Runs dialog and returns stderr (selection)."""
    with tempfile.NamedTemporaryFile(mode="w+") as tf:
        cmd = ["dialog"] + args
        try:
            subprocess.run(cmd, stderr=tf, check=True)
            tf.seek(0)
            return tf.read().strip()
        except subprocess.CalledProcessError:
            return None # User cancelled

def get_subnet_from_ip(ip):
    parts = ip.split('.')
    return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"

def setup_ips_dialog(current_head, current_worker):
    form_args = [
        "--title", "Cluster Configuration",
        "--form", "Enter IP addresses for Head and Worker nodes:",
        "10", "60", "2",
        "Head Node IP:", "1", "1",  current_head, "1", "20", "20", "0",
        "Worker Node IP:", "2", "1", current_worker, "2", "20", "20", "0"
    ]
    
    result = run_dialog(form_args)
    if not result:
        return None
        
    lines = result.splitlines()
    if len(lines) >= 2:
        return lines[0].strip(), lines[1].strip()
    return None

def setup_training_params(current_epochs, current_bs):
    form_args = [
        "--title", "Training Parameters",
        "--form", "Configure hyperparameters:",
        "10", "60", "2",
        "Epochs:", "1", "1",  str(current_epochs), "1", "20", "20", "0",
        "Batch Size (Per Node):", "2", "1", str(current_bs), "2", "20", "20", "0"
    ]
    
    result = run_dialog(form_args)
    if not result:
        return None
        
    lines = result.splitlines()
    if len(lines) >= 2:
        return lines[0].strip(), lines[1].strip()
    return None

def launch_training(head_ip, worker_ip, model, train_type, epochs, batch_size, force_ethernet=False, enable_nccl_debug=False):
    workspace_dir = Path("/opt/workspace")
        
    if not workspace_dir.exists():
        print(f"Error: Could not locate workspace directory inside container at {workspace_dir}")
        sys.exit(1)
        
    subnet = get_subnet_from_ip(head_ip)
    try:
        iface_cmd = f"ip -o addr show to {subnet} | awk '{{print $2}}' | head -n1"
        rdma_iface = subprocess.check_output(iface_cmd, shell=True, text=True).strip()
    except Exception as e:
        rdma_iface = "eth0"
        print(f"Warning: Could not detect RDMA IFACE ({e}), defaulting to eth0")

    print("\n" + "="*60)
    print(f" Launching Distributed FSDP Training Cluster")
    print(f" Head Node:   {head_ip}")
    print(f" Worker Node: {worker_ip}")
    print(f" Model:       {model}")
    print(f" Type:        {train_type}")
    print(f" Epochs:      {epochs}")
    print(f" Batch Size:  {batch_size}")
    print(f" Network:     {rdma_iface} (Force ETH: {force_ethernet}, Debug: {enable_nccl_debug})")
    print("="*60 + "\n")
    
    # Construct base accelerate command
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
        "--batch_size", str(batch_size)
    ]
    
    # 1. Launch Worker Process via SSH
    print(f"-> Starting Worker process on {worker_ip} (Background)...")
    worker_cmd = accel_base + ["--machine_rank", "1"] + train_script
    worker_cmd_str = " ".join(worker_cmd)
    
    nccl_ib_disable = "1" if force_ethernet else "0"
    worker_env = f"NCCL_SOCKET_IFNAME={rdma_iface} GLOO_SOCKET_IFNAME={rdma_iface} NCCL_IB_DISABLE={nccl_ib_disable}"
    if enable_nccl_debug:
        worker_env += " NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET"
    
    # We use toolbox run to execute the cd and the command INSIDE the container on the worker
    ssh_cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no", worker_ip,
        f"toolbox run -c strix-halo-llm-finetuning -- bash -c 'cd {workspace_dir} && {worker_env} {worker_cmd_str}'"
    ]
    
    # Run worker in a Popen so it streams to our console without blocking the python script from continuing
    worker_process = subprocess.Popen(ssh_cmd)
    
    # Give the worker a few seconds to initialize
    time.sleep(3)
    
    # 2. Launch Local Head Process
    print(f"-> Starting Head process on {head_ip} (Foreground)...")
    head_cmd = accel_base + ["--machine_rank", "0"] + train_script
    
    env = os.environ.copy()
    env["NCCL_SOCKET_IFNAME"] = rdma_iface
    env["GLOO_SOCKET_IFNAME"] = rdma_iface
    env["NCCL_IB_DISABLE"] = "1" if force_ethernet else "0"
    if enable_nccl_debug:
        env["NCCL_DEBUG"] = "INFO"
        env["NCCL_DEBUG_SUBSYS"] = "INIT,NET"
    
    try:
        # Run head process in foreground, wait for it to finish
        subprocess.run(head_cmd, env=env, cwd=workspace_dir, check=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except subprocess.CalledProcessError:
        print("\nTraining failed on head node.")
    finally:
        print("\nCleaning up worker process...")
        worker_process.terminate()
        try:
            worker_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            worker_process.kill()
        
        # Ensure remote accelerate is dead
        print("Sending kill signal to remote accelerate processes...")
        subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", worker_ip, "pkill -f accelerate"], stderr=subprocess.DEVNULL)
        
        print("\nDone.")

def main():
    check_dependencies()
    
    # Defaults
    head_ip = "192.168.100.1"
    worker_ip = "192.168.100.2"
    epochs = "2"
    batch_size = "4"
    force_ethernet = False
    enable_nccl_debug = False
    
    MODELS = [
        "google/gemma-3-1b-it",
        "google/gemma-3-4b-it",
        "google/gemma-3-12b-it",
        "google/gemma-3-27b-it"
    ]
    
    TRAIN_TYPES = [
        "qlora-4bit",
        "qlora-8bit",
        "lora",
        "full"
    ]
    
    while True:
        eth_status = "YES" if force_ethernet else "NO"
        debug_status = "YES" if enable_nccl_debug else "NO"
        
        choice = run_dialog([
            "--clear", "--backtitle", "AMD Strix Halo FSDP Training Launcher",
            "--title", "Main Menu",
            "--menu", "Select Action:", "17", "75", "6",
            "1", f"Configure IPs (Head: {head_ip}, Worker: {worker_ip})",
            "2", f"Configure Training (Epochs: {epochs}, BS: {batch_size})",
            "3", f"Network Settings (Force ETH: {eth_status}, Debug: {debug_status})",
            "4", "Start Distributed Training",
            "5", "Exit"
        ])
        
        if not choice or choice == "5":
            subprocess.run(["clear"])
            sys.exit(0)
            
        if choice == "1":
            res = setup_ips_dialog(head_ip, worker_ip)
            if res:
                head_ip, worker_ip = res
                
        elif choice == "2":
            res = setup_training_params(epochs, batch_size)
            if res:
                epochs, batch_size = res
                
        elif choice == "3":
            while True:
                e_stat = "YES" if force_ethernet else "NO"
                d_stat = "YES" if enable_nccl_debug else "NO"
                c_choice = run_dialog([
                    "--clear", "--title", "Cluster Network Configuration",
                    "--menu", "Set Network Parameters:", "15", "65", "3",
                    "1", f"Toggle Force Ethernet (Disable RDMA/RoCE): {e_stat}",
                    "2", f"Toggle Enable NCCL Debug Logging:          {d_stat}",
                    "3", "Return to Main Menu"
                ])
                if not c_choice or c_choice == "3": break
                
                if c_choice == "1":
                    force_ethernet = not force_ethernet
                elif c_choice == "2":
                    enable_nccl_debug = not enable_nccl_debug
                    
        elif choice == "4":
            # Select Model
            m_items = []
            for i, m in enumerate(MODELS):
                m_items.extend([str(i), m])
                
            m_choice = run_dialog([
                "--title", "Target Model",
                "--menu", "Select model to fine-tune:", "15", "60", "5"
            ] + m_items)
            
            if not m_choice: continue
            selected_model = MODELS[int(m_choice)]
            
            # Select Type
            t_items = []
            for i, t in enumerate(TRAIN_TYPES):
                t_items.extend([str(i), t])
                
            t_choice = run_dialog([
                "--title", "Training Type",
                "--menu", "Select quantization and tuning strategy:", "15", "60", "5"
            ] + t_items)
            
            if not t_choice: continue
            selected_type = TRAIN_TYPES[int(t_choice)]
            
            # Clear and run
            subprocess.run(["clear"])
            launch_training(head_ip, worker_ip, selected_model, selected_type, epochs, batch_size, force_ethernet, enable_nccl_debug)
            input("\nPress Enter to return to menu...")

if __name__ == "__main__":
    main()
