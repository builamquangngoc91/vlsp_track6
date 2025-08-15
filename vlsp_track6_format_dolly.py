import torch
from datasets import load_dataset
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

import os
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
from huggingface_hub import login
HF_TOKEN = os.getenv("HF_TOKEN")

api = HfApi(token=HF_TOKEN) 
login(token=HF_TOKEN) 

# 1. Define the model and tokenizer
model_name = "VLSP2025-LegalSML/qwen3-1.7b-legal-pretrain"
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"Before : {tokenizer.pad_token}")
tokenizer.pad_token = tokenizer.eos_token  # Set pad token if it's missing
tokenizer.padding_side = "left"  # IMPORTANT: Set padding_side to 'left' BEFORE tokenizing
print(f"After : {tokenizer.pad_token}")

# Before : <|endoftext|>
# After : <|endoftext|>

# 2. Load the dataset
dataset_name = "thailevann/finetune_track6_vlsp"
dataset = load_dataset(dataset_name, split="train")

# 1. Format the dataset first
def format_dolly(sample):
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


# Apply formatting
dataset = dataset.map(format_dolly).filter(lambda x: x is not None and x["conversation"] is not None)

reasoning_conversations = tokenizer.apply_chat_template(
    dataset["conversation"],
    tokenize = False,
)

import pandas as pd
reasoning_conversations = pd.Series(reasoning_conversations)

reasoning_conversations.name = "text"

from datasets import Dataset
reasoning_conversations = Dataset.from_pandas(pd.DataFrame(reasoning_conversations))
reasoning_conversations = reasoning_conversations.shuffle(seed = 3407)



# 2. Now tokenize the formatted data
def tokenize_function(examples):
    # Tokenize the texts with padding and truncation
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=2048)



# Apply tokenization to create input_ids, attention_mask, etc.
tokenized_dataset = reasoning_conversations.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)
   

# Split the tokenized dataset
train_test_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]


# 3. Configure QLoRA
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
        # "o_proj",
        # "gate_proj",
        # "up_proj",
        # "down_proj",
    ],
)


device_map = "auto"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    #quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=True,
    # attn_implementation="flash_attention_2" #FlashAttention only supports Ampere GPUs or newer.
)
     



# Load perplexity metric
# perplexity = load("perplexity", module_type="metric")

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
#     return perplexity.compute(predictions=predictions, references=labels)
def compute_metrics(eval_pred):
    return {} # Trainer will automatically log eval_loss for us.

# Add LoRA adapters to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


output_dir = "./adapter1"  # Directory to save fine-tuned model


data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,  # Keep at 1
    gradient_accumulation_steps=4,  # Increased to maintain batch size
    learning_rate=1e-4,  # Slightly reduced learning rate
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    num_train_epochs=1,  # Reduced from 3 to 2
    fp16=True,
    eval_strategy="steps",
    eval_steps=100,  # Increased eval steps to reduce frequency
    save_strategy="steps",
    save_steps=200,
    load_best_model_at_end=False,  # Disabled to save memory
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    push_to_hub=False,
    remove_unused_columns=False,
    logging_dir="./logs",
    logging_steps=20,
    report_to="none",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_pin_memory=False,  # Disable pin memory to save GPU memory
    dataloader_num_workers=2,     # Use single worker to save memory
    max_grad_norm=1.0,           # Add gradient clipping
    group_by_length=False,       # Disable to save memory
    length_column_name=None,
    eval_accumulation_steps=1,   # Process eval in smaller chunks
)


trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    data_collator=data_collator,
    # compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)


torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)
train_result = trainer.train()
from peft import PeftModel

# model_a là model đang train adapter A
model.save_pretrained("output_dir/adapter_a")
