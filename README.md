
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