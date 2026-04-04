
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

## Building llama.cpp Forks with ROCm (e.g. PrismML Bonsai)

When building llama.cpp forks (like [PrismML's Bonsai](https://github.com/PrismML-Eng/llama.cpp)) outside the toolbox against system ROCm on Strix Halo, cmake may fail to find HIP headers and libraries:

```
fatal error: 'hip/hip_fp16.h' file not found
ld.lld: error: unable to find library -lamdhip64
```

**Fix:** Pass explicit ROCm include and library paths to cmake:

```bash
CMAKE_PREFIX_PATH=/opt/rocm HIP_PATH=/opt/rocm cmake -B build \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1151 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_FLAGS="-I/opt/rocm/include" \
  -DCMAKE_EXE_LINKER_FLAGS="-L/opt/rocm/lib" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L/opt/rocm/lib"

cmake --build build -j$(nproc)
```

**Benchmark (PrismML Bonsai 8B, 1-bit, 1.15 GB):**

| Metric | CPU-only | ROCm/HIP (GPU) |
|--------|----------|-----------------|
| Prompt | 3.3 t/s | 81.4 t/s |
| Generation | 2.3 t/s | 76.7 t/s |

---

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

## Multi-Node Training (2× Strix Halo)

Two strategies for distributing training across nodes:

| | DDP | FSDP |
|---|---|---|
| **What** | Replicate full model on each node, sync gradients | Shard model across nodes |
| **When** | Model fits in one node's memory | Model too large for one node |
| **Speed** | Faster (less communication) | Slower (more communication) |
| **Use for** | 1B, 4B, 12B (LoRA/QLoRA), 12B (full w/ FSDP) | 12B full, 27B LoRA |
| **Docs** | [PyTorch DDP](https://pytorch.org/docs/stable/notes/ddp.html) | [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html) |

**TLDR**: Use DDP unless you get OOM, then switch to FSDP. The launcher picks the right one.

### Memory budget (128 GB per node, ~120 GB usable)

Full fine-tuning needs **weights + gradients + optimizer (2× Adam states)** ≈ `params × 16 bytes`. For 27B that's **324 GB** — impossible even with FSDP on 2 nodes. Use **LoRA** or **QLoRA** for 27B.

### Launcher (`start-finetuning-cluster.py`)

Interactive TUI to configure and launch training. Includes:
- Model, type, batch size, learning rate, context length
- **Strategy** (DDP/FSDP) — only shown in multi-node mode
- **Gradient accumulation** — amortizes network sync cost (default: 1)
- **Memory estimation** — warns before launching configs that would OOM

```bash
toolbox enter strix-halo-llm-finetuning
cd ~/finetuning-workspace/workspace
python3 start-finetuning-cluster.py
```

### Benchmark (`benchmark_configs.py`)

Estimates memory for all model×type×batch combos and optionally runs each on the cluster:

```bash
python3 benchmark_configs.py          # preview plan (safe)
python3 benchmark_configs.py --run    # run each config for 1 epoch
```

### RDMA / InfiniBand

For multi-node, the toolbox must map `/dev/infiniband` for RDMA. Use the provided `refresh-toolbox.sh` which auto-detects InfiniBand devices:

```bash
./refresh-toolbox.sh    # recreates toolbox with RDMA support
```

### Custom Datasets

You can use your own datasets for fine-tuning via the interactive UI (Launcher) or by passing the `--dataset` argument to `train.py`. The dataset can be either a Hugging Face repository ID (e.g., `mlabonne/guanaco-llama2-1k`, `philschmid/dolly-15k-oai-style`) or a local path to a JSON/JSONL file.

**Format Requirements**:
The custom dataset must include a `"messages"` column matching the standard Hugging Face Chat Template format. For example, a local `.jsonl` file should look like this:
```json
{"messages": [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "The capital of France is Paris."}]}
{"messages": [{"role": "user", "content": "Write a python loop."}, {"role": "assistant", "content": "for i in range(10):\n    print(i)"}]}
```

If the `Dataset` argument is left as default, the script falls back to `Abirate/english_quotes` and automatically formats it to the required structure for demonstration.

