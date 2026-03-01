#!/usr/bin/env python3
"""
Memory profiling script for Unsloth vs standard HF training.

Measures peak GPU memory for each (model, type, batch_size) combination using
both standard HF and Unsloth loading. Runs 3 training steps per config.

CRITICAL: Each measurement runs in a SEPARATE SUBPROCESS because unsloth_zoo
globally patches SFTTrainer at import time, which would contaminate HF-only
measurements if both ran in the same process.

Usage:
  python3 measure_unsloth_memory.py
  python3 measure_unsloth_memory.py --models google/gemma-3-1b-it google/gemma-3-4b-it
  python3 measure_unsloth_memory.py --types lora qlora --batch-sizes 1 4

Output:
  Prints a comparison table and saves results to unsloth_memory_profile.json.
"""
import argparse
import json
import os
import subprocess
import sys
import textwrap


# ── Worker script template ────────────────────────────────────────────
# This gets written to a temp file and executed in a clean subprocess
# so that unsloth_zoo patches don't leak between measurements.
WORKER_SCRIPT = textwrap.dedent(r'''
import gc
import json
import os
import sys
import torch

def cleanup():
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
    from datasets import load_dataset
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
            target_modules=["q_proj","k_proj","v_proj","o_proj",
                            "gate_proj","up_proj","down_proj"],
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
            target_modules=["q_proj","k_proj","v_proj","o_proj",
                            "gate_proj","up_proj","down_proj"],
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
            target_modules=["q_proj","k_proj","v_proj","o_proj",
                            "gate_proj","up_proj","down_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        optim = "paged_adamw_8bit"

    use_grad_ckpt = train_type in ("8bit-lora", "qlora")
    training_args = SFTConfig(
        output_dir="/tmp/measure-hf",
        max_length=max_length, packing=False,
        num_train_epochs=1, max_steps=3,
        per_device_train_batch_size=batch_size,
        gradient_checkpointing=use_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_grad_ckpt else None,
        optim=optim, logging_steps=1,
        save_strategy="no", eval_strategy="no",
        bf16=True, fp16=False,
        lr_scheduler_type="constant", report_to="none",
        dataset_kwargs={"add_special_tokens": False, "append_concat_token": True},
    )

    trainer = SFTTrainer(
        model=model, args=training_args,
        train_dataset=ds["train"], processing_class=tokenizer,
    )
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    return get_peak_gb()


def measure_unsloth(model_id, train_type, batch_size, max_length, ds):
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
            target_modules=["q_proj","k_proj","v_proj","o_proj",
                            "gate_proj","up_proj","down_proj"],
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
            target_modules=["q_proj","k_proj","v_proj","o_proj",
                            "gate_proj","up_proj","down_proj"],
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
            target_modules=["q_proj","k_proj","v_proj","o_proj",
                            "gate_proj","up_proj","down_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        optim = "paged_adamw_8bit"

    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
    def apply_template(examples):
        texts = [tokenizer.apply_chat_template(
            m, tokenize=False, add_generation_prompt=False
        ).removeprefix('<bos>') for m in examples["messages"]]
        return {"text": texts}
    ds_formatted = {"train": ds["train"].map(apply_template, batched=True)}

    use_grad_ckpt = train_type in ("8bit-lora", "qlora")
    training_args = SFTConfig(
        output_dir="/tmp/measure-unsloth",
        dataset_text_field="text", dataset_num_proc=1,
        max_length=max_length, packing=False,
        num_train_epochs=1, max_steps=3,
        per_device_train_batch_size=batch_size,
        gradient_checkpointing=use_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_grad_ckpt else None,
        optim=optim, logging_steps=1,
        save_strategy="no", eval_strategy="no",
        bf16=True, fp16=False,
        lr_scheduler_type="constant", report_to="none",
        dataset_kwargs={"add_special_tokens": False, "append_concat_token": True},
    )

    trainer = SFTTrainer(
        model=model, args=training_args,
        train_dataset=ds_formatted["train"], processing_class=tokenizer,
    )
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    return get_peak_gb()


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1])
    ds = prepare_dataset()
    engine = cfg["engine"]
    try:
        if engine == "hf":
            peak = measure_hf(cfg["model_id"], cfg["type"], cfg["batch_size"], cfg["max_length"], ds)
        else:
            peak = measure_unsloth(cfg["model_id"], cfg["type"], cfg["batch_size"], cfg["max_length"], ds)
        print(json.dumps({"peak_gb": round(peak, 2)}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
''')


def run_worker(config: dict) -> dict:
    """Run a single measurement in a clean subprocess."""
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(WORKER_SCRIPT)
        worker_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, worker_path, json.dumps(config)],
            capture_output=True, text=True, timeout=600,
        )
        # Find the JSON output (last line of stdout that looks like JSON)
        for line in reversed(result.stdout.strip().split("\n")):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        # No JSON found — return stderr as error
        return {"error": result.stderr[-500:] if result.stderr else "No output"}
    except subprocess.TimeoutExpired:
        return {"error": "TIMEOUT (>600s)"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        os.unlink(worker_path)


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
                    "model": name, "model_id": model_id,
                    "type": train_type, "batch_size": batch_size,
                    "max_length": args.max_length,
                }

                base_cfg = {"model_id": model_id, "type": train_type,
                            "batch_size": batch_size, "max_length": args.max_length}

                # Measure HF (in clean subprocess — no unsloth_zoo patches)
                if not args.unsloth_only:
                    print(f"\n{label} — HF ...", end=" ", flush=True)
                    result = run_worker({**base_cfg, "engine": "hf"})
                    if "peak_gb" in result:
                        entry["hf_peak_gb"] = result["peak_gb"]
                        print(f"{result['peak_gb']:.2f} GB")
                    else:
                        entry["hf_peak_gb"] = None
                        entry["hf_error"] = result["error"]
                        print(f"FAILED: {result['error'][:200]}")

                # Measure Unsloth (in clean subprocess)
                if not args.hf_only:
                    print(f"{label} — Unsloth ...", end=" ", flush=True)
                    result = run_worker({**base_cfg, "engine": "unsloth"})
                    if "peak_gb" in result:
                        entry["unsloth_peak_gb"] = result["peak_gb"]
                        print(f"{result['peak_gb']:.2f} GB")
                    else:
                        entry["unsloth_peak_gb"] = None
                        entry["unsloth_error"] = result["error"]
                        print(f"FAILED: {result['error'][:200]}")

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

    # Average discounts per type
    type_discounts = {}
    for r in results:
        if r.get("discount"):
            type_discounts.setdefault(r["type"], []).append(r["discount"])

    if type_discounts:
        print(f"\nAverage discounts (unsloth/hf) per type:")
        for tt, vals in sorted(type_discounts.items()):
            avg = sum(vals) / len(vals)
            print(f"  {tt}: {avg:.3f} (from {len(vals)} measurements)")
        print("\nUse these to update UNSLOTH_DISCOUNT in benchmark_configs.py")


if __name__ == "__main__":
    main()
