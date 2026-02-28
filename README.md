
```

███████╗████████╗██████╗ ██╗██╗  ██╗      ██╗  ██╗ █████╗ ██╗      ██████╗
██╔════╝╚══██╔══╝██╔══██╗██║╚██╗██╔╝      ██║  ██║██╔══██╗██║     ██╔═══██╗
███████╗   ██║   ██████╔╝██║ ╚███╔╝       ███████║███████║██║     ██║   ██║
╚════██║   ██║   ██╔══██╗██║ ██╔██╗       ██╔══██║██╔══██║██║     ██║   ██║
███████║   ██║   ██║  ██║██║██╔╝ ██╗      ██║  ██║██║  ██║███████╗╚██████╔╝
╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝      ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝ ╚═════╝

                     L L M   F I N E - T U N I N G                        
```

# STRIX HALO - LLM Finetuning Toolbox (gfx1151)

Fedora toolbox with ROCm 7 nightly (TheRock) for fine-tuning Gemma-3, Qwen-3, and GPT-OSS-20B on AMD Strix Halo.

- Image: `docker.io/kyuz0/amd-strix-halo-llm-finetuning:latest`  
- Repo: `https://github.com/kyuz0/amd-strix-halo-llm-finetuning`

---

### 📦 Project Context

This repository is part of the **[Strix Halo AI Toolboxes](https://strix-halo-toolboxes.com)** project. Check out the website for an overview of all toolboxes, tutorials, and host configuration guides.

### ❤️ Support

This is a hobby project maintained in my spare time. If you find these toolboxes and tutorials useful, you can **[buy me a coffee](https://buymeacoffee.com/dcapitella)** to support the work! ☕

---

## Updates (2026-02-03)
- **ROCm Nightly**: Toolbox updated to include the latest version of ROCm 7 nightly builds from "TheRock".
- **RCCL Patch**: Added a custom patch for RCCL to properly support `gfx1151` (Strix Halo).

---


## Watch the YouTube Video

[![Watch the YouTube Video](https://img.youtube.com/vi/nxugSRDg_jg/maxresdefault.jpg)](https://youtu.be/nxugSRDg_jg)  


## Performance on Strix Halo

| Model | Full FT | LoRA | 8-bit + LoRA | QLoRA |
|:------|:------:|:----:|:-------------:|:-----:|
| Gemma-3 1B-IT | 19 GB / 2m52s | 15 GB / 2m | 13 GB / 8m | 13 GB / 9m |
| Gemma-3 4B-IT | 46 GB / 9m | 30 GB / 5m | 21 GB / 41m | 13 GB / 9m |
| Gemma-3 12B-IT | 115 GB / 25m | 67 GB / 13m | 43 GB / 2h38m | 26 GB / 23m |
| Gemma-3 27B-IT | OOM | OOM | 32 GB unstable | 19 GB runs |
| GPT-OSS-20B (MXFP4) | - | 32-38 GB / ~1h | - | - |

Notes: Gemma results are 2 epochs at max_length 512. GPT-OSS uses LoRA with MXFP4 dequantized to bf16.

---

## 1) Create the toolbox

```bash
toolbox create strix-halo-llm-finetuning \
  --image docker.io/kyuz0/amd-strix-halo-llm-finetuning:latest \
  -- --device /dev/dri --device /dev/kfd \
  --group-add video --group-add render --security-opt seccomp=unconfined
```

What the flags do:

* `--device /dev/dri` expose GPU display path
* `--device /dev/kfd` enable ROCm compute
* `--group-add video --group-add render` allow user access
* `--security-opt seccomp=unconfined` avoid GPU syscall blocks

Ubuntu fix if devices are not visible inside the toolbox:

```bash
sudo tee /etc/udev/rules.d/99-amd-kfd.rules >/dev/null <<'EOF'
SUBSYSTEM=="kfd", GROUP="render", MODE="0666", OPTIONS+="last_rule"
SUBSYSTEM=="drm", KERNEL=="card[0-9]*", GROUP="render", MODE="0666", OPTIONS+="last_rule"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger
```

---

## 2) Enter the toolbox

```bash
toolbox enter strix-halo-llm-finetuning
```

You will see the banner with ROCm version, GPU info, and the Jupyter command hint.

---

## 3) Start Jupyter Lab

On your first run, you want to create a worksapce folder for Jupyter and copy the default notebooks like this:

```bash
mkdir -p ~/finetuning-workspace/
cp -r /opt/workspace ~/finetuning-workspace/
```

Run:

```bash
jupyter lab --notebook-dir ~/finetuning-workspace/
```

This starts Jupyter Lab in `~/finetuning-workspace/`, where you copiled all notebooks.
Jupyter will print a full URL containing a token — just copy that entire link and open it in your browser.

If you’re connected over SSH, forward port 8888 first so you can open it locally:

```bash
ssh -L 8888:localhost:8888 user@your-strix-halo-host
```

Then open the same full URL (with the token) in your browser at [http://localhost:8888](http://localhost:8888).

---

## 4) Notebooks

| Notebook                   | Purpose                                                                         |
| -------------------------- | ------------------------------------------------------------------------------- |
| `gemma-finetuning.ipynb`   | Fine-tuning Gemma-3 and Qwen-3 with Full, LoRA, 8-bit LoRA, and QLoRA.          |
| `gpt-oss-finetuning.ipynb` | LoRA fine-tuning for GPT-OSS-20B (MXFP4), reasoning and non-reasoning adapters. |


---

## 5) Practical notes

* Training for Gemma and Qwen: use `attn_implementation="eager"`.
* FlashAttention 2: inference only.
* GPT-OSS on ROCm: `Mxfp4Config(dequantize=True)` to use bf16 in memory.
* Check memory: `torch.cuda.max_memory_allocated()`.

---

## 6) Persistence

| Item                  | Path on host            |
| --------------------- | ----------------------- |
| Notebooks and outputs | `~/finetuning-workspace/`          |
| Hugging Face cache    | `~/.cache/huggingface/` |
| Hugging Face token    | `~/.huggingface/token`  |

Everything under your home persists across toolbox rebuilds.


## 7) Kernel Parameters (tested on Fedora 42)

Add these these boot parameters to enable unified memory and optimal performance:

```
amd_iommu=off amdgpu.gttsize=131072 ttm.pages_limit=33554432

```
| Parameter                   | Purpose                                                                                  |
| --------------------------- | ----------------------------------------------------------------------------------------- |
| `amd_iommu=off`             | Disables IOMMU for lower latency                                                         |
| `amdgpu.gttsize=131072`     | Enables unified GPU/system memory (up to 128 GiB); 131072 MiB ÷ 1024 = 128 GiB           |
| `ttm.pages_limit=33554432`  | Allows large pinned memory allocations; 33554432 × 4 KiB = 134217728 KiB ÷ 1024² = 128 GiB |

Source: https://www.reddit.com/r/LocalLLaMA/comments/1m9wcdc/comment/n5gf53d/?context=3&utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button.

**Apply the changes:**

```
# Edit /etc/default/grub to add parameters to GRUB_CMDLINE_LINUX
sudo grub2-mkconfig -o /boot/grub2/grub.cfg
sudo reboot
```

---

## 8) Multi-Node Distributed Training (FSDP)

You can train much larger models (e.g. Gemma 3 27B) by pooling the VRAM of multiple Strix Halo APUs across different physical machines using the experimental PyTorch FSDP wrapper over RCCL.

**Prerequisites:**
1. You need two physical machines on the same high-speed network (e.g., 100GbE RoCE v2 or Thunderbolt direct connection).
2. The `strix-halo-llm-finetuning` toolbox container must be installed and created on **both** machines.
3. SSH access must be enabled and accessible without a password between the host machines.
4. You must copy the `workspace` folder to `~/finetuning-workspace` on **both** machines.

**Launch the Cluster (Tested on 192.168.100.1 / 192.168.100.2):**
To start the distributed training, run the TUI script from the Head node:

```bash
cd ~/finetuning-workspace
./start_training_cluster.py
```

The terminal interface will guide you through:
1. **Configuring IPs**: Enter the IP addresses of the Head and Worker node (defaults to `192.168.100.1` and `192.168.100.2`).
2. **Configuring Training**: Set Hyperparameters like Epochs and Batch Size per node.
3. **Network Settings**: Troubleshoot connectivity by forcing standard Ethernet over RDMA if needed, or activating NCCL debug logs.
4. **Starting Training**: Select your target model and fine-tuning quantization strategy.

The script automatically connects securely to the remote machine via SSH, locates the worker container, and simultaneously spins up the Accelerate PyTorch distributed processes.

**Outputs**:
All final model weight checkpoints and adapter configurations are automatically saved into your configured "Output Dir" (defaults to `~/finetuning-workspace/output-{model_name}-{quant_type}-fsdp`).