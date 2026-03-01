#!/usr/bin/env python3
import sys
import os
import tempfile
import subprocess
import time

def check_dependencies():
    missing = []
    if not subprocess.run(["which", "dialog"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        missing.append("dialog")
    if not subprocess.run(["which", "ssh"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
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

def get_subnet_from_ip(ip):
    """Accurately gets the /24 subnet string for the given IP."""
    parts = ip.split('.')
    return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"

def get_dynamic_iface(subnet):
    try:
        iface_cmd = f"ip -o addr show to {subnet} | awk '{{print $2}}' | head -n1"
        rdma_iface = subprocess.check_output(iface_cmd, shell=True, text=True).strip()
        if rdma_iface:
            return rdma_iface
    except:
        pass
    return "eth0" # Fallback

def get_rdma_env_script(head_ip, ip, iface, force_ethernet, enable_nccl_debug):
    """Generates the environment variables script for ROCm/RDMA injection."""
    nccl_disable_val = "1" if force_ethernet else "0"
    
    env_script = f"""
    export RDMA_IFACE={iface}
    export NCCL_SOCKET_IFNAME={iface}
    export GLOO_SOCKET_IFNAME={iface}
    export NCCL_IB_TIMEOUT=23
    export NCCL_IB_RETRY_CNT=7
    export NCCL_IB_DISABLE={nccl_disable_val}
    """
    if enable_nccl_debug:
        env_script += """
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=INIT,NET
        """
    return env_script

# Approximate parameter counts for memory estimation
_MODEL_PARAMS_B = {
    "google/gemma-3-1b-it": 1.0,
    "google/gemma-3-4b-it": 4.0,
    "google/gemma-3-12b-it": 12.0,
    "google/gemma-3-27b-it": 27.0,
}

def _estimate_memory(model_id, train_type, strategy, batch_size, max_length, world_size):
    """Estimate peak GPU memory in GB per node. Conservative estimate."""
    params_b = _MODEL_PARAMS_B.get(model_id, 1.0)
    params = params_b * 1e9
    shard = world_size if strategy == "fsdp" else 1

    if train_type == "full":
        weights = params * 2  # bf16
        grads = params * 2
        optim = params * 4 * 2  # Adam fp32, 2 states
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
        state += (weights * 0.1) / 1e9  # peak during gather

    # Activations estimate
    hidden = {1.0: 1024, 4.0: 2048, 12.0: 3840, 27.0: 5184}.get(params_b, 2048)
    layers = {1.0: 26, 4.0: 34, 12.0: 48, 27.0: 62}.get(params_b, 32)
    act_per_layer = (batch_size * max_length * hidden * 4) / 1e9
    use_ckpt = strategy == "fsdp" or train_type in ("8bit-lora", "qlora")
    activations = act_per_layer * (layers ** 0.5 if use_ckpt else layers)

    return state + activations + 2.0  # +2GB overhead

def launch_training(mode, head_ip, worker_ip, force_ethernet, enable_nccl_debug):
    models = ["google/gemma-3-1b-it", "google/gemma-3-4b-it", "google/gemma-3-12b-it", "google/gemma-3-27b-it"]
    types = ["lora", "full", "8bit-lora", "qlora"]
    
    strategies = ["ddp", "fsdp"]
    # Defaults based on notebook
    current_model_idx = 0
    current_type_idx = 0
    current_strategy_idx = 0
    current_batch = 4
    current_epochs = 2
    current_lr = "5e-5"
    current_ctx = 512
    current_grad_accum = 1 if mode == "Multi-Node" else 1
    
    while True:
        model_name = models[current_model_idx].split("/")[-1]
        type_name = types[current_type_idx]
        strategy_name = strategies[current_strategy_idx].upper()
        
        menu_items = [
            "1", f"Model:             {model_name}",
            "2", f"Finetune Type:     {type_name}",
            "3", f"Batch Size:        {current_batch}",
            "4", f"Epochs:            {current_epochs}",
            "5", f"Learning Rate:     {current_lr}",
            "6", f"Max Context Len:   {current_ctx}",
            "7", f"Grad Accumulation: {current_grad_accum}",
        ]
        if mode == "Multi-Node":
            menu_items.extend(["8", f"Strategy:          {strategy_name}"])
            launch_idx = "9"
            cancel_idx = "10"
            item_count = "11"
        else:
            launch_idx = "8"
            cancel_idx = "9"
            item_count = "10"
        menu_items.extend([launch_idx, "LAUNCH TRAINING", cancel_idx, "CANCEL"])
        
        menu_args = [
            "--clear", "--backtitle", f"AMD Finetuning Launcher (Mode: {mode})",
            "--title", f"Configuration",
            "--menu", "Customize Launch Parameters:", "26", "65", item_count,
        ] + menu_items
        
        choice = run_dialog(menu_args)
        if not choice or choice == cancel_idx:
            return
            
        if choice == "1":
            menu_items = []
            for i, m in enumerate(models):
                menu_items.extend([str(i), m])
            m_choice = run_dialog(["--title", "Select Model", "--menu", "Choose a model:", "15", "60", "6"] + menu_items)
            if m_choice: current_model_idx = int(m_choice)
        elif choice == "2":
            menu_items = []
            for i, t in enumerate(types):
                menu_items.extend([str(i), t])
            t_choice = run_dialog(["--title", "Select Type", "--menu", "Choose a finetune type:", "15", "60", "4"] + menu_items)
            if t_choice: current_type_idx = int(t_choice)
        elif choice == "3":
            new_val = run_dialog(["--title", "Batch Size", "--inputbox", "Enter Batch Size (per device):", "10", "40", str(current_batch)])
            if new_val: current_batch = int(new_val)
        elif choice == "4":
            new_val = run_dialog(["--title", "Epochs", "--inputbox", "Enter Epochs:", "10", "40", str(current_epochs)])
            if new_val: current_epochs = int(new_val)
        elif choice == "5":
            new_val = run_dialog(["--title", "Learning Rate", "--inputbox", "Enter Learning Rate:", "10", "40", str(current_lr)])
            if new_val: current_lr = new_val
        elif choice == "6":
            new_val = run_dialog(["--title", "Max Context Len", "--inputbox", "Enter Max Context Length:", "10", "40", str(current_ctx)])
            if new_val: current_ctx = int(new_val)
        elif choice == "7":
            new_val = run_dialog(["--title", "Gradient Accumulation", "--inputbox", "Steps to accumulate before gradient sync\n(higher = less network overhead, larger effective batch):", "12", "55", str(current_grad_accum)])
            if new_val: current_grad_accum = int(new_val)
        elif choice == "8" and mode == "Multi-Node":
            s_choice = run_dialog(["--title", "Strategy", "--menu", "DDP: replicate model (fast, needs model to fit per node)\nFSDP: shard model (for large models like 27B)", "14", "65", "2",
                                   "0", "DDP  - Data Parallel (default)",
                                   "1", "FSDP - Fully Sharded (large models)"])
            if s_choice is not None: current_strategy_idx = int(s_choice)
        elif choice == launch_idx:
            break

    subprocess.run(["clear"])
    model_id = models[current_model_idx]
    train_type = types[current_type_idx]

    # ── Memory estimation check ────────────────────────────────────
    world_size = 2 if mode == "Multi-Node" else 1
    strategy = strategies[current_strategy_idx]
    est_gb = _estimate_memory(model_id, train_type, strategy, current_batch, current_ctx, world_size)
    available_gb = 120  # 128GB - ~8GB OS overhead

    if est_gb > available_gb:
        msg = (f"WARNING: This configuration is estimated to need ~{est_gb:.0f} GB per node.\n"
               f"Available: ~{available_gb} GB (128 GB minus OS overhead).\n\n"
               f"Model: {model_id} ({train_type})\n"
               f"Strategy: {strategy.upper()}, Batch: {current_batch}\n\n"
               f"This WILL cause a system OOM that requires REBOOT!\n\n"
               f"Suggestions:\n"
               f"- Use LoRA or QLoRA instead of full fine-tuning\n"
               f"- Reduce batch size\n"
               f"- Use FSDP to shard the model across nodes\n\n"
               f"Proceed anyway?")
        warn_result = run_dialog(["--title", "⚠️ MEMORY WARNING", "--yesno", msg, "22", "65"])
        if warn_result is None:
            return
    elif est_gb > available_gb * 0.85:
        msg = (f"This config is estimated to use ~{est_gb:.0f} GB of {available_gb} GB available.\n"
               f"It may be tight. Consider reducing batch size if it OOMs.\n\n"
               f"Proceed?")
        warn_result = run_dialog(["--title", "⚠️ Memory Tight", "--yesno", msg, "12", "65"])
        if warn_result is None:
            return
    
    # Common arguments
    strategy = strategies[current_strategy_idx]
    train_args = f"--model {model_id} --type {train_type} --strategy {strategy} --batch-size {current_batch} --epochs {current_epochs} --learning-rate {current_lr} --max-length {current_ctx} --gradient-accumulation {current_grad_accum}"
    
    # Head node script path
    script_dir = os.path.abspath(os.path.dirname(__file__))
    train_script = os.path.join(script_dir, "train.py")
    
    # Worker node script path (fixed at /opt/workspace as per toolbox configuration)
    worker_script_dir = "/opt/workspace"
    worker_train_script = f"{worker_script_dir}/train.py"
    
    if mode == "Single Node":
        print(f"Launching Single Node Training: {model_id} ({train_type})")
        print("Command:", f"{train_script} {train_args}")
        try:
            subprocess.run(f"python3 {train_script} {train_args}", shell=True, check=True)
        except subprocess.CalledProcessError:
            print("Training failed.")
        input("Press Enter to continue...")
        
    elif mode == "Multi-Node":
        print(f"Launching Multi-Node DDP Training: {model_id} ({train_type})")
        print(f"Head: {head_ip}, Worker: {worker_ip}")
        
        subnet_head = get_subnet_from_ip(head_ip)
        iface_head = get_dynamic_iface(subnet_head)
        
        # Prepare Worker Command over SSH
        print("Initializing worker node...")
        worker_env = get_rdma_env_script(head_ip, worker_ip, iface_head, force_ethernet, enable_nccl_debug)
        worker_cmd = f"""
        {worker_env}
        cd {worker_script_dir}
        torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr={head_ip} --master_port=12345 {worker_train_script} {train_args}
        """
        
        hf_token_opt = ""
        if "HF_TOKEN" in os.environ:
            hf_token_opt = f"export HF_TOKEN={os.environ['HF_TOKEN']}; "
            
        ssh_cmd = [
            "ssh", "-o", "StrictHostKeyChecking=no", worker_ip, 
            f"toolbox run -c strix-halo-llm-finetuning -- bash -c '{hf_token_opt}{worker_cmd}'"
        ]
        
        # Launch worker in background
        print("Starting worker process over SSH...")
        worker_proc = subprocess.Popen(ssh_cmd, stdout=sys.stdout, stderr=sys.stderr)
        
        # Give worker a moment to start
        time.sleep(3)
        
        print("Starting head process...")
        head_env = os.environ.copy()
        head_env["RDMA_IFACE"] = iface_head
        head_env["NCCL_SOCKET_IFNAME"] = iface_head
        head_env["GLOO_SOCKET_IFNAME"] = iface_head
        head_env["NCCL_IB_TIMEOUT"] = "23"
        head_env["NCCL_IB_RETRY_CNT"] = "7"
        head_env["NCCL_IB_DISABLE"] = "1" if force_ethernet else "0"
        if enable_nccl_debug:
            head_env["NCCL_DEBUG"] = "INFO"
            head_env["NCCL_DEBUG_SUBSYS"] = "INIT,NET"
            
        head_cmd = [
            "torchrun", "--nproc_per_node=1", "--nnodes=2", "--node_rank=0",
            f"--master_addr={head_ip}", "--master_port=12345", train_script
        ] + train_args.split()
        
        try:
            subprocess.run(head_cmd, env=head_env, check=True)
        except subprocess.CalledProcessError:
            print("Head node training failed.")
            worker_proc.terminate()
            
        print("Waiting for worker to complete...")
        worker_proc.wait()
        
        print("Multi-Node Training completed.")
        input("Press Enter to continue...")

def main():
    check_dependencies()
    
    head_ip = os.getenv("VLLM_HEAD_IP", "192.168.100.1")
    worker_ip = os.getenv("VLLM_WORKER_IP", "192.168.100.2")
    
    force_ethernet = False
    enable_nccl_debug = False
    
    while True:
        eth_status = "YES" if force_ethernet else "NO"
        debug_status = "YES" if enable_nccl_debug else "NO"
        
        choice = run_dialog([
            "--clear", "--backtitle", "AMD Finetuning Cluster Manager",
            "--title", "Main Menu",
            "--menu", "Select Action:", "16", "65", "6",
            "1", f"Configure IPs (Head: {head_ip}, Worker: {worker_ip})",
            "2", "Cluster Network Configuration",
            "3", "Run Single Node Benchmark",
            "4", "Run Multi-Node (2 nodes) Cluster",
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
            while True:
                eth_st = "YES" if force_ethernet else "NO"
                dbg_st = "YES" if enable_nccl_debug else "NO"
                
                c_choice = run_dialog([
                    "--clear", "--backtitle", "AMD Finetuning Cluster Manager",
                    "--title", "Cluster Network Configuration",
                    "--menu", "Set Network Parameters:", "15", "65", "3",
                    "1", f"Force Ethernet (Disable RDMA/RoCE):  {eth_st}",
                    "2", f"Enable NCCL Debug Logging:           {dbg_st}",
                    "3", "BACK TO MAIN MENU"
                ])
                if not c_choice or c_choice == "3": break
                
                if c_choice == "1": force_ethernet = not force_ethernet
                elif c_choice == "2": enable_nccl_debug = not enable_nccl_debug
                
        elif choice == "3":
            launch_training("Single Node", head_ip, worker_ip, force_ethernet, enable_nccl_debug)
            
        elif choice == "4":
            launch_training("Multi-Node", head_ip, worker_ip, force_ethernet, enable_nccl_debug)

if __name__ == "__main__":
    main()
