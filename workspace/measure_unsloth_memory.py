#!/usr/bin/env python3
"""
Memory profiling script for Unsloth vs standard HF training.

Measures peak GPU memory for each (model, type, batch_size) combination using
both standard HF and Unsloth loading. Runs 3 training steps per config to
capture the training peak (not just model loading).

Usage:
  python3 measure_unsloth_memory.py
  python3 measure_unsloth_memory.py --models google/gemma-3-1b-it google/gemma-3-4b-it
  python3 measure_unsloth_memory.py --types lora qlora --batch-sizes 1 4

Output:
  Prints a comparison table and saves results to unsloth_memory_profile.json.
  The discount factors can be used to update the memory estimators in
  benchmark_configs.py and start-finetuning-cluster.py.
"""
import argparse
import gc
import json
import os
import time
import torch
from datasets import load_dataset


def cleanup():
    """Aggressive cleanup between runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()


def get_peak_gb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def prepare_dataset():
    """Load and format the standard benchmark dataset."""
    ds = load_dataset("Abirate/english_quotes", split="train").shuffle(seed=42).select(range(200))

    def format_chat(ex):
        return {
            "messages": [
                {"role": "user", "content": f"Give me a quote about: {ex['tags']}"},
                {"role": "assistant", "content": f"{ex['quote']} - {ex['author']}"},
            ]
        }

    ds = ds.map(format_chat, remove_columns=ds.column_names)
    return ds.train_test_split(test_size=0.2)


def measure_hf(model_id, train_type, batch_size, max_length, ds):
    """Measure peak memory using standard HF loading."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if train_type == "full":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, attn_implementation="eager"
        )
        optim = "adamw_torch_fused"
    elif train_type == "lora":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, attn_implementation="eager"
        )
        lora_config = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        optim = "adamw_torch_fused"
    elif train_type == "8bit-lora":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config,
            device_map="auto", attn_implementation="eager"
        )
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        optim = "adamw_8bit"
    elif train_type == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config,
            device_map="auto", attn_implementation="eager"
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
        optim = "paged_adamw_8bit"

    use_grad_ckpt = train_type in ("8bit-lora", "qlora")
    training_args = SFTConfig(
        output_dir="/tmp/measure-hf",
        max_length=max_length,
        packing=False,
        num_train_epochs=1,
        max_steps=3,
        per_device_train_batch_size=batch_size,
        gradient_checkpointing=use_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_grad_ckpt else None,
        optim=optim,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        bf16=True, fp16=False,
        lr_scheduler_type="constant",
        report_to="none",
        dataset_kwargs={"add_special_tokens": False, "append_concat_token": True},
    )

    trainer = SFTTrainer(
        model=model, args=training_args,
        train_dataset=ds["train"], processing_class=tokenizer,
    )

    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    peak = get_peak_gb()

    del trainer, model
    cleanup()
    return peak


def measure_unsloth(model_id, train_type, batch_size, max_length, ds):
    """Measure peak memory using Unsloth loading."""
    os.environ["UNSLOTH_SKIP_TORCHVISION_CHECK"] = "1"
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from trl import SFTConfig, SFTTrainer

    if train_type == "full":
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id, max_seq_length=max_length,
            dtype=None, load_in_4bit=False,
        )
        optim = "adamw_torch_fused"
    elif train_type == "lora":
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id, max_seq_length=max_length,
            dtype=None, load_in_4bit=False,
        )
        model = FastLanguageModel.get_peft_model(
            model, r=16, lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        optim = "adamw_torch_fused"
    elif train_type == "8bit-lora":
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id, max_seq_length=max_length,
            dtype=None, load_in_8bit=True, load_in_4bit=False,
        )
        model = FastLanguageModel.get_peft_model(
            model, r=16, lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        optim = "adamw_8bit"
    elif train_type == "qlora":
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id, max_seq_length=max_length,
            dtype=None, load_in_4bit=True,
        )
        model.config.use_cache = False
        model = FastLanguageModel.get_peft_model(
            model, r=16, lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        optim = "paged_adamw_8bit"

    # Pre-format dataset with chat template
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

    def apply_template(examples):
        texts = [tokenizer.apply_chat_template(
            m, tokenize=False, add_generation_prompt=False
        ).removeprefix('<bos>') for m in examples["messages"]]
        return {"text": texts}

    ds_formatted = {
        "train": ds["train"].map(apply_template, batched=True),
    }

    use_grad_ckpt = train_type in ("8bit-lora", "qlora")
    training_args = SFTConfig(
        output_dir="/tmp/measure-unsloth",
        dataset_text_field="text",
        max_length=max_length,
        packing=False,
        num_train_epochs=1,
        max_steps=3,
        per_device_train_batch_size=batch_size,
        gradient_checkpointing=use_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_grad_ckpt else None,
        optim=optim,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        bf16=True, fp16=False,
        lr_scheduler_type="constant",
        report_to="none",
        dataset_kwargs={"add_special_tokens": False, "append_concat_token": True},
    )

    trainer = SFTTrainer(
        model=model, args=training_args,
        train_dataset=ds_formatted["train"], processing_class=tokenizer,
    )

    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    peak = get_peak_gb()

    del trainer, model
    cleanup()
    return peak


def main():
    parser = argparse.ArgumentParser(description="Profile memory: HF vs Unsloth")
    parser.add_argument("--models", nargs="*",
                        default=["google/gemma-3-1b-it", "google/gemma-3-4b-it"])
    parser.add_argument("--types", nargs="*", default=["full", "lora", "qlora"])
    parser.add_argument("--batch-sizes", nargs="*", type=int, default=[4])
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output", type=str, default="unsloth_memory_profile.json")
    parser.add_argument("--unsloth-only", action="store_true",
                        help="Only measure Unsloth (skip HF baseline)")
    parser.add_argument("--hf-only", action="store_true",
                        help="Only measure HF baseline (skip Unsloth)")
    args = parser.parse_args()

    print("Loading dataset...")
    ds = prepare_dataset()

    results = []
    total = len(args.models) * len(args.types) * len(args.batch_sizes)
    idx = 0

    for model_id in args.models:
        for train_type in args.types:
            for batch_size in args.batch_sizes:
                idx += 1
                name = model_id.split("/")[-1]
                label = f"[{idx}/{total}] {name} {train_type} batch={batch_size}"

                entry = {
                    "model": name,
                    "model_id": model_id,
                    "type": train_type,
                    "batch_size": batch_size,
                    "max_length": args.max_length,
                }

                # Measure HF
                if not args.unsloth_only:
                    print(f"\n{label} — HF ...", end=" ", flush=True)
                    try:
                        hf_peak = measure_hf(model_id, train_type, batch_size, args.max_length, ds)
                        entry["hf_peak_gb"] = round(hf_peak, 2)
                        print(f"{hf_peak:.2f} GB")
                    except Exception as e:
                        entry["hf_peak_gb"] = None
                        entry["hf_error"] = str(e)
                        print(f"FAILED: {e}")
                    cleanup()
                    time.sleep(2)

                # Measure Unsloth
                if not args.hf_only:
                    print(f"{label} — Unsloth ...", end=" ", flush=True)
                    try:
                        us_peak = measure_unsloth(model_id, train_type, batch_size, args.max_length, ds)
                        entry["unsloth_peak_gb"] = round(us_peak, 2)
                        print(f"{us_peak:.2f} GB")
                    except Exception as e:
                        entry["unsloth_peak_gb"] = None
                        entry["unsloth_error"] = str(e)
                        print(f"FAILED: {e}")
                    cleanup()
                    time.sleep(2)

                # Compute discount
                hf_val = entry.get("hf_peak_gb")
                us_val = entry.get("unsloth_peak_gb")
                if hf_val and us_val and hf_val > 0:
                    entry["discount"] = round(us_val / hf_val, 3)

                results.append(entry)

    # Save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Saved {len(results)} results to {args.output}")

    # Print summary table
    print(f"\n{'='*90}")
    print("MEMORY PROFILE: HF vs Unsloth")
    print(f"{'='*90}")
    print(f"{'Model':<16} {'Type':<10} {'Batch':<6} {'HF (GB)':<10} {'Unsloth (GB)':<14} {'Discount':<10}")
    print("-" * 90)
    for r in results:
        hf_str = f"{r['hf_peak_gb']:.1f}" if r.get("hf_peak_gb") else "N/A"
        us_str = f"{r['unsloth_peak_gb']:.1f}" if r.get("unsloth_peak_gb") else "N/A"
        disc_str = f"{r['discount']:.3f}" if r.get("discount") else "N/A"
        print(f"{r['model']:<16} {r['type']:<10} {r['batch_size']:<6} "
              f"{hf_str:<10} {us_str:<14} {disc_str:<10}")

    # Compute average discounts per type
    type_discounts = {}
    for r in results:
        if r.get("discount"):
            tt = r["type"]
            type_discounts.setdefault(tt, []).append(r["discount"])

    if type_discounts:
        print(f"\nAverage discounts (unsloth/hf) per type:")
        for tt, vals in sorted(type_discounts.items()):
            avg = sum(vals) / len(vals)
            print(f"  {tt}: {avg:.3f} (from {len(vals)} measurements)")
        print("\nUse these to update UNSLOTH_DISCOUNT in benchmark_configs.py")


if __name__ == "__main__":
    main()
