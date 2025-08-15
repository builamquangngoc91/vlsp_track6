#!/usr/bin/env python3

import torch
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
import matplotlib.pyplot as plt
from evaluate import load
import os
import pandas as pd
from huggingface_hub import HfApi, login

# Configuration
HF_TOKEN = login(token=os.getenv("HF_TOKEN"))  #
model_name = "VLSP2025-LegalSML/qwen3-1.7b-legal-pretrain"
dataset_name = "thailevann/finetune_track6_vlsp"
output_dir = "./adapter1"

def setup_huggingface():
    """Setup HuggingFace API and login"""
    api = HfApi(token=HF_TOKEN) 
    login(token=HF_TOKEN)
    return api

def setup_tokenizer(model_name):
    """Setup tokenizer with proper configuration"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Before : {tokenizer.pad_token}")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"After : {tokenizer.pad_token}")
    return tokenizer

def format_dolly(sample):
    """Format dataset samples for training"""
    question = sample["question"]
    chosen_answer = (
        f"<think>\n{sample['chosen_reason']}\n</think>\n"
        f"{sample['chosen_answer']}"
    )

    user_prompt = question
    
    chat_conversations = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": chosen_answer}
    ]

    return {"conversation": chat_conversations}

def prepare_dataset(dataset_name, tokenizer):
    """Load and prepare dataset for training"""
    # Load dataset
    dataset = load_dataset(dataset_name, split="train")
    print(f"Dataset size: {len(dataset)}")
    
    # Apply formatting
    dataset = dataset.map(format_dolly).filter(lambda x: x is not None and x["conversation"] is not None)
    
    # Apply chat template
    reasoning_conversations = tokenizer.apply_chat_template(
        dataset["conversation"],
        tokenize=False,
    )
    
    # Convert to pandas Series and back to Dataset
    reasoning_conversations = pd.Series(reasoning_conversations)
    reasoning_conversations.name = "text"
    reasoning_conversations = Dataset.from_pandas(pd.DataFrame(reasoning_conversations))
    reasoning_conversations = reasoning_conversations.shuffle(seed=3407)
    
    return reasoning_conversations

def tokenize_function(examples, tokenizer):
    """Tokenize the texts with padding and truncation"""
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=2048)

def setup_model(model_name):
    """Setup the model with LoRA configuration"""
    device_map = "auto"
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        trust_remote_code=True,
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
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
    
    # Add LoRA adapters
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    return {}

def setup_training_args(output_dir):
    """Setup training arguments"""
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=1,
        fp16=True,
        eval_strategy="steps",
        eval_steps=100,
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
        dataloader_num_workers=2,
        max_grad_norm=1.0,
        group_by_length=False,
        length_column_name=None,
        eval_accumulation_steps=1,
    )

def main():
    """Main training function"""
    # Setup HuggingFace
    api = setup_huggingface()
    
    # Setup tokenizer
    tokenizer = setup_tokenizer(model_name)
    
    # Prepare dataset
    reasoning_conversations = prepare_dataset(dataset_name, tokenizer)
    
    # Tokenize dataset
    tokenized_dataset = reasoning_conversations.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=["text"],
    )
    
    # Split dataset
    train_test_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]
    
    print(f"Train dataset: {train_dataset}")
    print(f"Eval dataset: {eval_dataset}")
    
    # Setup model
    model = setup_model(model_name)
    
    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    # Setup training arguments
    training_args = setup_training_args(output_dir)
    
    # Create trainer
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
    
    # Train model
    train_result = trainer.train()
    
    # Save adapter
    model.save_pretrained("output_dir/adapter_a")
    
    print("Training completed!")

if __name__ == "__main__":
    main()