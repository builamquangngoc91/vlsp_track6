#!/usr/bin/env python3
"""
Fine-tuning script for VLSP Track 6 - Legal Question Answering
Converted from Jupyter notebook for server deployment
"""

import os
import argparse
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import HfApi, login
import warnings

warnings.filterwarnings("ignore")

def setup_device():
    """Setup device configuration"""
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU. This will be very slow.")
        return "cpu"
    else:
        # Use specific device mapping for 4-bit training
        return {'': torch.cuda.current_device()}

def setup_tokenizer(model_name):
    """Setup and configure tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Before: {tokenizer.pad_token}")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"After: {tokenizer.pad_token}")
    return tokenizer

def format_dolly(sample):
    """Format dataset sample for training"""
    question = sample["question"]
    chosen_answer = (
        f"<think>\\n{sample['chosen_reason']}\\n</think>\\n"
        f"{sample['chosen_answer']}"
    )
    
    prompt = f"### Question:\\n{question}\\n\\n### Answer:\\n{chosen_answer}"
    return {"text": prompt}

def tokenize_function(examples, tokenizer):
    """Tokenize the formatted data"""
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

def prepare_dataset(dataset_name, tokenizer):
    """Load and prepare dataset"""
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")
    print(f"Dataset size: {len(dataset)}")
    
    # Format the dataset
    dataset = dataset.map(format_dolly).filter(lambda x: x is not None and x["text"] is not None)
    
    # Tokenize the formatted data
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=["question", "chosen_answer", "chosen_reason", "rejected_answer", "rejected_reason", "text"],
    )
    
    # Split the tokenized dataset
    train_test_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    return train_test_split["train"], train_test_split["test"]

def setup_lora_config():
    """Configure LoRA parameters"""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
        ],
    )

def setup_quantization_config():
    """Configure 4-bit quantization"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

def load_model(model_name, quantization_config, device_map):
    """Load and prepare the model"""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=True,
    )
    
    # Prepare the model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    return model

def setup_training_args(output_dir, num_epochs=2, learning_rate=1e-4):
    """Setup training arguments"""
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=num_epochs,
        fp16=True,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=False,
        remove_unused_columns=False,
        logging_dir="./logs",
        logging_steps=20,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        max_grad_norm=1.0,
        group_by_length=False,
        length_column_name=None,
        eval_accumulation_steps=1,
    )

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    return {}

def train_model(model, train_dataset, eval_dataset, training_args, tokenizer):
    """Train the model"""
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    # Enable optimizations
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    
    print("Starting training...")
    train_result = trainer.train()
    
    return trainer, train_result

def evaluate_and_save(trainer, model, tokenizer, output_dir, hub_repo_id=None):
    """Evaluate model and save results"""
    # Evaluate the model
    eval_history = trainer.evaluate()
    final_eval_loss = eval_history.get("eval_loss")
    
    if final_eval_loss is not None:
        final_perplexity = torch.exp(torch.tensor(final_eval_loss)).item()
        print(f"Final Evaluation Loss: {final_eval_loss:.4f}")
        print(f"Final Perplexity: {final_perplexity:.2f}")
    
    # Plot training history
    plot_training_history(trainer, output_dir)
    
    # Save the model
    print("Merging LoRA adapters...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    
    # Push to hub if specified
    if hub_repo_id:
        print(f"Pushing to Hugging Face Hub: {hub_repo_id}")
        merged_model.push_to_hub(
            repo_id=hub_repo_id,
            use_temp_dir=False,
            private=False
        )
        tokenizer.push_to_hub(
            repo_id=hub_repo_id,
            use_temp_dir=False
        )

def plot_training_history(trainer, output_dir):
    """Plot and save training history"""
    train_losses = []
    eval_losses = []
    
    # Extract metrics from log history
    for log in trainer.state.log_history:
        if "loss" in log and "learning_rate" in log:
            train_losses.append({"step": log["step"], "loss": log["loss"]})
        if "eval_loss" in log:
            eval_losses.append({"step": log["step"], "loss": log["eval_loss"]})
    
    # Prepare data for plotting
    train_steps_plot = [entry["step"] for entry in train_losses]
    train_values_plot = [entry["loss"] for entry in train_losses]
    
    eval_steps_plot = [entry["step"] for entry in eval_losses]
    eval_values_plot = [entry["loss"] for entry in eval_losses]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(train_steps_plot, train_values_plot, label="Training Loss", marker='.')
    plt.plot(eval_steps_plot, eval_values_plot, label="Evaluation Loss", marker='o', linestyle='--')
    
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Loss Over Steps")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "loss_plot.png")
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune model for VLSP Track 6")
    parser.add_argument("--model_name", default="VLSP2025-LegalSML/qwen3-1.7b-legal-pretrain", help="Base model name")
    parser.add_argument("--dataset_name", default="thailevann/finetune_track6_vlsp", help="Dataset name")
    parser.add_argument("--output_dir", default="./qwen2_5_dolly_qlora", help="Output directory")
    parser.add_argument("--hub_repo_id", help="Hugging Face Hub repository ID")
    parser.add_argument("--hf_token", help="Hugging Face token")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    # Setup Hugging Face authentication
    if args.hf_token:
        login(token=args.hf_token)
    
    # Setup device
    device_map = setup_device()
    
    # Setup tokenizer
    tokenizer = setup_tokenizer(args.model_name)
    
    # Prepare dataset
    train_dataset, eval_dataset = prepare_dataset(args.dataset_name, tokenizer)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    
    # Setup model configuration
    lora_config = setup_lora_config()
    quantization_config = setup_quantization_config()
    
    # Load and setup model
    model = load_model(args.model_name, quantization_config, device_map)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Setup training arguments
    training_args = setup_training_args(
        args.output_dir, 
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate
    )
    
    # Train the model
    trainer, train_result = train_model(model, train_dataset, eval_dataset, training_args, tokenizer)
    
    # Evaluate and save
    evaluate_and_save(trainer, model, tokenizer, args.output_dir, args.hub_repo_id)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()