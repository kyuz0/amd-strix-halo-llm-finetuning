import os
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
    # Initialize Accelerate for FSDP
    accelerator = Accelerator()
    
    MODEL = "google/gemma-3-1b-it" 
    model_name = MODEL.split("/")[-1]
    
    # Training parameters
    LR = 5e-5
    EPOCHS = 2
    BATCH_SIZE = 4
    
    # Load dataset
    accelerator.print("Loading dataset...")
    # Seed fixed for reproducibility across nodes
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
    
    # Load tokenizer and base model using standard Transformers + BitsAndBytes
    accelerator.print("Loading model and tokenizer...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16 # Required parameter for FSDP + QLoRA
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, 
        quantization_config=bnb_config, 
        device_map={"": accelerator.local_process_index}, # Must use LOCAL process index, not global!
        attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    
    # PEFT / LoRA setup
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
    
    if accelerator.is_main_process:
        model.print_trainable_parameters()
        print(f"Weights footprint: {model.get_memory_footprint()/1e9:.2f} GB")
        
    # Set up SFT Trainer
    args = SFTConfig(
        output_dir=f"output-{model_name}-qlora-fsdp",
        max_length=512,
        packing=False,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
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
    
    # SFTTrainer integrates with Accelerate and detects the FSDP environment
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
        report_peak_mem("qlora-fsdp")
        accelerator.print("Saving model on main process...")
        trainer.save_model()
        
    accelerator.print("Done.")

if __name__ == "__main__":
    main()
