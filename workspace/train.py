#!/usr/bin/env python3
"""
Multi-node fine-tuning script for AMD Strix Halo cluster.

Based on the official TRL sft.py example:
  https://github.com/huggingface/trl/blob/main/trl/scripts/sft.py

Launch methods:
  Single:      python3 train.py --model google/gemma-3-1b-it --type lora
  Multi-node:  accelerate launch --config_file config.yaml train.py --model ... --type ...
  or:          torchrun --nnodes=2 --nproc_per_node=1 ... train.py --model ... --type ...

Key principles (from official HuggingFace TRL source):
  - Do NOT set device_map for non-quantized models; let the Trainer handle device placement
  - For quantized models (8bit/4bit), use get_kbit_device_map() from trl
  - The Trainer wraps with DDP automatically when launched in a distributed environment
"""
import time
import argparse
import sys
import gc
import os

# Unsloth MUST be imported before transformers/peft/trl to apply patches.
# We check sys.argv early to do this before any other ML imports.
if "--unsloth" in sys.argv:
    os.environ["UNSLOTH_SKIP_TORCHVISION_CHECK"] = "1"
    import unsloth  # noqa: F401 — triggers patch hooks

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel
from trl import SFTConfig, SFTTrainer
from trl.trainer.utils import get_kbit_device_map
from datasets import load_dataset


class DebugCallback(TrainerCallback):
    """Prints device/timing info from INSIDE the training loop on every rank."""

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        if step > 3 and step % 50 != 0:
            return
        rank = int(os.environ.get("RANK", "0"))
        # Check where model params actually are
        devices = set()
        for p in model.parameters():
            devices.add(str(p.device))
            break  # just check first param
        mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        print(f"[RANK {rank}] step={step} param_device={devices} gpu_mem={mem:.2f}GB", flush=True)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        if step > 3 and step % 50 != 0:
            return
        rank = int(os.environ.get("RANK", "0"))
        mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        print(f"[RANK {rank}] step={step} DONE gpu_mem={mem:.2f}GB peak={peak:.2f}GB", flush=True)


def reset_peak_mem():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_peak_mem():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def full_cleanup(model=None, trainer=None):
    try:
        acc = getattr(trainer, "accelerator", None)
        if acc and hasattr(acc, "free_memory"):
            acc.free_memory()
    except Exception:
        pass
    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Multi-node Finetuning for AMD Strix Halo")
    parser.add_argument("--model", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--type", type=str, choices=["full", "lora", "8bit-lora", "qlora"], default="lora")
    parser.add_argument("--strategy", type=str, choices=["ddp", "fsdp"], default="ddp",
                        help="Distributed strategy: ddp (replicate model) or fsdp (shard model for large models)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=0,
                        help="Gradient accumulation steps. 0=auto (8 for multi-node, 1 for single)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--unsloth", action="store_true",
                        help="Use Unsloth FastLanguageModel for optimized loading and training")
    parser.add_argument("--debug", action="store_true", help="Enable per-step device/memory debug output")
    args = parser.parse_args()

    if args.unsloth and args.strategy == "fsdp":
        parser.error("Unsloth does not support FSDP. Use --strategy ddp instead.")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1
    is_main = rank == 0

    # Auto-select gradient accumulation: 8 for multi-node to amortize
    # the ~2-3s gradient sync cost across multiple compute steps.
    # Without this, each ~1s compute step pays ~2-3s network overhead.
    # With accum=8: 8 local steps (~8s) + 1 sync (~2s) = 10s vs 8×3s=24s.
    if args.gradient_accumulation == 0:
        grad_accum = 8 if is_distributed else 1
    else:
        grad_accum = args.gradient_accumulation

    if is_main:
        strategy_label = args.strategy.upper() if is_distributed else "Single"
        unsloth_label = " [Unsloth]" if args.unsloth else ""
        print(f"Training: {args.model} | type={args.type} | world_size={world_size} | strategy={strategy_label}{unsloth_label}")
        print(f"Params:   lr={args.learning_rate} epochs={args.epochs} batch={args.batch_size} maxlen={args.max_length}")
        if is_distributed:
            print(f"Dist:     gradient_accumulation_steps={grad_accum} (amortize sync cost)")

    # ── Dataset ────────────────────────────────────────────────────────
    ds = load_dataset("Abirate/english_quotes", split="train").shuffle(seed=42).select(range(1000))

    def format_chat(ex):
        return {
            "messages": [
                {"role": "user", "content": f"Give me a quote about: {ex['tags']}"},
                {"role": "assistant", "content": f"{ex['quote']} - {ex['author']}"},
            ]
        }

    ds = ds.map(format_chat, remove_columns=ds.column_names)
    ds = ds.train_test_split(test_size=0.2)

    # ── Model + Tokenizer ──────────────────────────────────────────────
    if args.unsloth:
        # ── Unsloth path ──────────────────────────────────────────────
        # Uses FastLanguageModel which handles device placement, attention
        # implementation, and quantization internally. Matches the tested
        # configs from gemma-finetuning-unsloth.ipynb exactly.
        os.environ["UNSLOTH_SKIP_TORCHVISION_CHECK"] = "1"
        from unsloth import FastLanguageModel  # noqa: already imported early for patches

        if args.type == "full":
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.model, max_seq_length=args.max_length,
                dtype=None, load_in_4bit=False, full_finetuning=True,
            )
            bf16, fp16 = True, False
            optim = "adamw_torch_fused"

        elif args.type == "lora":
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.model, max_seq_length=args.max_length,
                dtype=None, load_in_4bit=False,
            )
            model = FastLanguageModel.get_peft_model(
                model, r=16, lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            )
            bf16, fp16 = True, False
            optim = "adamw_torch_fused"

        elif args.type == "8bit-lora":
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.model, max_seq_length=args.max_length,
                dtype=None, load_in_8bit=True, load_in_4bit=False,
            )
            model = FastLanguageModel.get_peft_model(
                model, r=16, lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            )
            bf16, fp16 = True, False
            optim = "adamw_8bit"

        elif args.type == "qlora":
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.model, max_seq_length=args.max_length,
                dtype=None, load_in_4bit=True,
            )
            model.config.use_cache = False
            model = FastLanguageModel.get_peft_model(
                model, r=16, lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            )
            bf16, fp16 = True, False
            optim = "paged_adamw_8bit"

        # Pre-format dataset with chat template for Unsloth
        # (Unsloth SFTTrainer expects a pre-formatted "text" column)
        from unsloth.chat_templates import get_chat_template
        tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

        def apply_template(examples):
            texts = [tokenizer.apply_chat_template(
                m, tokenize=False, add_generation_prompt=False
            ).removeprefix('<bos>') for m in examples["messages"]]
            return {"text": texts}

        ds["train"] = ds["train"].map(apply_template, batched=True)
        ds["test"] = ds["test"].map(apply_template, batched=True)

    else:
        # ── Standard HF path ──────────────────────────────────────────
        # Following the official TRL sft.py pattern:
        #   - Non-quantized: NO device_map, NO .to() — Trainer handles placement
        #   - Quantized: device_map = get_kbit_device_map() from trl
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        if args.type == "full":
            model = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.bfloat16, attn_implementation="eager"
            )
            bf16, fp16 = True, False
            optim = "adamw_torch_fused"

        elif args.type == "lora":
            model = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.bfloat16, attn_implementation="eager"
            )
            lora_config = LoraConfig(
                r=16, lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            bf16, fp16 = True, False
            optim = "adamw_torch_fused"

        elif args.type == "8bit-lora":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                args.model, quantization_config=bnb_config,
                device_map=get_kbit_device_map(), attn_implementation="eager"
            )
            model = prepare_model_for_kbit_training(model)
            lora_config = LoraConfig(
                r=16, lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            bf16, fp16 = True, False
            optim = "adamw_8bit"

        elif args.type == "qlora":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.model, quantization_config=bnb_config,
                device_map=get_kbit_device_map(), attn_implementation="eager"
            )
            model = prepare_model_for_kbit_training(model)
            model.config.use_cache = False
            lora_config = LoraConfig(
                r=16, lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            bf16, fp16 = True, False
            optim = "paged_adamw_8bit"

    if is_main and args.type != "full":
        model.print_trainable_parameters()
        print(f"Footprint: {model.get_memory_footprint() / 1e9:.2f} GB")

    if args.debug:
        first_p = next(model.parameters())
        print(f"[RANK {rank}] PRE-TRAINER model device={first_p.device} dtype={first_p.dtype}", flush=True)

    # ── SFTConfig / Trainer ────────────────────────────────────────────
    unsloth_tag = "unsloth-" if args.unsloth else ""
    output_dir = f"output-{unsloth_tag}{args.model.split('/')[-1]}-{args.type}"

    # FSDP needs gradient checkpointing to fit large models
    use_grad_ckpt = True if args.strategy == "fsdp" or args.type in ("8bit-lora", "qlora") else False
    grad_ckpt_kwargs = {"use_reentrant": False} if use_grad_ckpt else None

    # Build FSDP config if needed
    fsdp_args = {}
    if args.strategy == "fsdp" and is_distributed:
        # Detect the transformer layer class for auto_wrap
        layer_cls = None
        for name, module in model.named_modules():
            cls_name = type(module).__name__
            if "DecoderLayer" in cls_name:
                layer_cls = cls_name
                break
        if layer_cls and is_main:
            print(f"FSDP:     auto-wrap layer: {layer_cls}")

        fsdp_args["fsdp"] = "full_shard auto_wrap"
        fsdp_config = {
            "backward_prefetch": "backward_pre",
            "forward_prefetch": True,
            "use_orig_params": True,
        }
        if layer_cls:
            fsdp_config["transformer_layer_cls_to_wrap"] = [layer_cls]
        fsdp_args["fsdp_config"] = fsdp_config

    # Unsloth uses pre-formatted "text" column; standard HF uses "messages"
    sft_dataset_kwargs = {"add_special_tokens": False, "append_concat_token": True}
    sft_extra = {}
    if args.unsloth:
        sft_extra["dataset_text_field"] = "text"
        # Unsloth patches the tokenizer with closures that can't be pickled,
        # so we must disable multiprocess tokenization inside SFTTrainer.
        sft_extra["dataset_num_proc"] = 1

    training_args = SFTConfig(
        output_dir=output_dir,
        max_length=args.max_length,
        packing=False,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=use_grad_ckpt,
        gradient_checkpointing_kwargs=grad_ckpt_kwargs,
        optim=optim,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        learning_rate=args.learning_rate,
        fp16=fp16,
        bf16=bf16,
        lr_scheduler_type="constant",
        report_to="none",
        dataset_kwargs=sft_dataset_kwargs,
        save_total_limit=1,
        # DDP optimizations: find_unused_parameters=True is needed because
        # Gemma 3 12B+ are multimodal (vision params unused in text-only training)
        ddp_find_unused_parameters=True,
        ddp_bucket_cap_mb=256,
        **sft_extra,
        **fsdp_args,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        processing_class=tokenizer,
        callbacks=[DebugCallback()] if args.debug else [],
    )

    if args.debug:
        wrapped = trainer.model
        first_p = next(wrapped.parameters())
        print(f"[RANK {rank}] POST-TRAINER model device={first_p.device} type={type(wrapped).__name__}", flush=True)
        print(f"[RANK {rank}] trainer.args.device={training_args.device}", flush=True)
        print(f"[RANK {rank}] is_model_parallel={getattr(trainer, 'is_model_parallel', 'N/A')}", flush=True)

    reset_peak_mem()
    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time
    peak = get_peak_mem()

    if is_main:
        print("-" * 50)
        print("TRAINING COMPLETED")
        print(f"Time:   {elapsed:.2f}s")
        print(f"Peak:   {peak:.2f} GB")
        print("-" * 50)
        trainer.save_model()

    full_cleanup(model, trainer)


if __name__ == "__main__":
    main()
