import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
from accelerate import Accelerator

def reset_peak_mem():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def report_peak_mem(tag: str = ""):
    if torch.cuda.is_available():
        print(f"Peak training memory{(' ' + tag) if tag else ''}: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

def main():
    parser = argparse.ArgumentParser(description="Distributed Fine-Tuning for Strix Halo")
    parser.add_argument("--model", type=str, default="google/gemma-3-1b-it", 
                        help="HuggingFace model ID (e.g. google/gemma-3-1b-it, google/gemma-3-4b-it)")
    parser.add_argument("--type", type=str, choices=["full", "lora", "qlora-8bit", "qlora-4bit"], 
                        default="qlora-4bit", help="Type of training to perform")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output_dir", type=str, default="~/finetuning-workspace",
                        help="Base directory to save the output model checkpoints")
    
    args_cli = parser.parse_args()
    
    # Initialize Accelerate for FSDP
    accelerator = Accelerator()
    
    MODEL = args_cli.model
    model_name = MODEL.split("/")[-1]
    
    # Load dataset
    accelerator.print(f"Loading dataset for {args_cli.type} training on {MODEL}...")
    ds = load_dataset("Abirate/english_quotes", split="train").shuffle(seed=42).select(range(1000))

    def format_chat(ex):
        return {
            "messages": [
                {"role": "user", "content": f"Give me a quote about: {ex['tags']}"},
                {"role": "assistant", "content": f"{ex['quote']} - {ex['author']}"}
            ]
        }
    
    with accelerator.main_process_first():
        ds = ds.map(format_chat, remove_columns=ds.column_names)
        ds = ds.train_test_split(test_size=0.2)
        
    accelerator.print(f"Train: {len(ds['train'])}, Test: {len(ds['test'])}")
    
    # Setup Quantization and Model Loading Arguments
    model_kwargs = {
        "device_map": {"": accelerator.local_process_index},
        "attn_implementation": "eager",
        "torch_dtype": torch.bfloat16
    }
    
    if args_cli.type == "qlora-4bit":
        accelerator.print("Using 4-bit QLoRA...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16 # Required for FSDP
        )
        model_kwargs["quantization_config"] = bnb_config
    elif args_cli.type == "qlora-8bit":
        accelerator.print("Using 8-bit QLoRA...")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        model_kwargs["quantization_config"] = bnb_config
        
    # Load Base Model
    accelerator.print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(MODEL, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    
    # Prepare and Apply PEFT if applicable
    if args_cli.type in ["lora", "qlora-8bit", "qlora-4bit"]:
        if "qlora" in args_cli.type:
            model = prepare_model_for_kbit_training(model)
            
        accelerator.print("Applying LoRA adapters...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        
        # FSDP Dtype Sweep Fix
        for name, param in model.named_parameters():
            if param.dtype == torch.float32:
                param.data = param.data.to(torch.bfloat16)
                
        if accelerator.is_main_process:
            model.print_trainable_parameters()
    else:
        accelerator.print("Performing Full Fine-Tuning (no PEFT)...")
        # For full FT with FSDP, make sure model is strictly bfloat16
        for name, param in model.named_parameters():
            if param.dtype == torch.float32:
                param.data = param.data.to(torch.bfloat16)

    model.config.use_cache = False
    if accelerator.is_main_process:
        print(f"Weights footprint: {model.get_memory_footprint()/1e9:.2f} GB")
        
    from pathlib import Path
    output_base = Path(args_cli.output_dir).expanduser()
    output_path = output_base / f"output-{model_name}-{args_cli.type}-fsdp"
    
    # To match notebook VRAM while taking advantage of distributed FSDP, 
    # we enforce gradient accumulation to keep micro-batches small.
    micro_batch = max(1, args_cli.batch_size // 4)
    grad_accum = max(1, args_cli.batch_size // micro_batch)
    
    args = SFTConfig(
        output_dir=str(output_path),
        max_length=512,
        packing=False,
        num_train_epochs=args_cli.epochs,
        per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True if args_cli.type != "full" else False, # Often disabled for Full FT
        gradient_checkpointing_kwargs={"use_reentrant": False} if args_cli.type != "full" else None,
        optim="paged_adamw_8bit" if "qlora" in args_cli.type else "adamw_torch_fused",
        fp16=False,
        bf16=True,
        lr_scheduler_type="constant",
        report_to="none",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        dataset_kwargs={"add_special_tokens": False, "append_concat_token": True},
        save_total_limit=1,
    )
    
    trainer = SFTTrainer(
        model=model, 
        args=args, 
        train_dataset=ds['train'], 
        eval_dataset=ds['test'], 
        processing_class=tokenizer
    )
    
    accelerator.print("Starting training loop...")
    reset_peak_mem()
    trainer.train()
    
    if accelerator.is_main_process:
        report_peak_mem(f"{args_cli.type}-fsdp")
        accelerator.print("Saving model on main process...")
        trainer.save_model()
        
    accelerator.print("Done.")

if __name__ == "__main__":
    main()
