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
login(token=HF_TOKEN)  # Hoặc thay bằng login("your_token")

model_name = "VLSP2025-LegalSML/qwen3-1.7b-legal-pretrain"

# Ensure the 4-bit model is loaded on the same device that training will use
if torch.cuda.is_available():
    device_map = {"": torch.cuda.current_device()}
else:
    device_map = {"": "cpu"}

tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"Before : {tokenizer.pad_token}")
tokenizer.pad_token = tokenizer.eos_token  # Set pad token if it's missing
tokenizer.padding_side = "left"  # IMPORTANT: Set padding_side to 'left' BEFORE tokenizing
print(f"After : {tokenizer.pad_token}")

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

    prompt = f"### Question:\n{question}\n\n### Answer:\n{chosen_answer}"
    return {"text": prompt}
     

# Apply formatting
dataset = dataset.map(format_dolly).filter(lambda x: x is not None and x["text"] is not None)


# 2. Now tokenize the formatted data
def tokenize_function(examples):
    # Tokenize the texts with padding and truncation
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

     

# Apply tokenization to create input_ids, attention_mask, etc.
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["question", "chosen_answer", "chosen_reason", "rejected_answer", "rejected_reason", "text"],
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


def create_bnb_config(quant_mode: str) -> BitsAndBytesConfig | None:
    if quant_mode == "4bit":
        # Use float32 compute to avoid fragile tensor-op paths
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
        )
    if quant_mode == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None

def build_model_and_trainer(quant_mode: str) -> tuple[AutoModelForCausalLM, Trainer]:
    bnb_cfg = create_bnb_config(quant_mode)
    if bnb_cfg is None:
        # Full precision fallback; prefer bf16 if supported else fp16
        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        model_local = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager",
        )
    else:
        model_local = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_cfg,
            device_map=device_map,
            trust_remote_code=True,
            attn_implementation="eager",
        )

    # Prepare and attach LoRA
    model_local = prepare_model_for_kbit_training(model_local)
    model_local = get_peft_model(model_local, lora_config)
    model_local.print_trainable_parameters()

    trainer_local = Trainer(
        model=model_local,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    return model_local, trainer_local



# Load perplexity metric
# perplexity = load("perplexity", module_type="metric")

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
#     return perplexity.compute(predictions=predictions, references=labels)
def compute_metrics(eval_pred):
    return {} # Trainer will automatically log eval_loss for us.


output_dir = "./qwen2_5_dolly_qlora"  # Directory to save fine-tuned model


data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)



training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,  # Keep at 1
    gradient_accumulation_steps=16,  # Increased to maintain batch size
    learning_rate=1e-4,  # Slightly reduced learning rate
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    num_train_epochs=2,  # Reduced from 3 to 2
    fp16=False,
    bf16=False,
    eval_strategy="steps",
    eval_steps=50,  # Increased eval steps to reduce frequency
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
    dataloader_num_workers=0,     # Use single worker to save memory
    max_grad_norm=1.0,           # Add gradient clipping
    group_by_length=False,       # Disable to save memory
    length_column_name=None,
    eval_accumulation_steps=1,   # Process eval in smaller chunks
)

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# Try 4-bit → 8-bit → full-precision fallbacks if CUDA/BnB GEMM errors occur
quant_try_order = ["4bit", "8bit", "full"]
last_error = None
for qmode in quant_try_order:
    try:
        model, trainer = build_model_and_trainer(qmode)
        train_result = trainer.train()
        break
    except RuntimeError as e:
        err_msg = str(e)
        if ("CUBLAS_STATUS" in err_msg or "cublas" in err_msg.lower() or
            "bitsandbytes" in err_msg.lower() or "matmul_4bit" in err_msg or
            "No available kernel" in err_msg):
            last_error = e
            # Cleanup and retry with the next quantization mode
            try:
                del model
            except Exception:
                pass
            torch.cuda.empty_cache()
            continue
        else:
            raise
else:
    # If all modes failed, re-raise the last CUDA/bnb error
    if last_error is not None:
        raise last_error
    raise RuntimeError("Training failed without a captured CUDA/bitsandbytes error.")


# Merge LoRA vào base model
merged_model = model.merge_and_unload()  # Hoặc nếu dùng PEFT mới: model.merge_and_unload()

# Lưu full model
merged_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
merged_model.push_to_hub(
    repo_id="thailevann/qwen3-1.7b-merged",
    use_temp_dir=False,
    private=False  # True nếu muốn private repo
)

tokenizer.push_to_hub(
    repo_id="thailevann/qwen3-1.7b-merged",
    use_temp_dir=False
)

# Store training and evaluation metrics
train_history = train_result.metrics
eval_history = trainer.evaluate()
final_eval_loss = eval_history.get("eval_loss")

if final_eval_loss is not None:
    final_perplexity = torch.exp(torch.tensor(final_eval_loss)).item()
    print(f"Final Evaluation Loss: {final_eval_loss:.4f}")
    print(f"Final Perplexity: {final_perplexity:.2f}")

# Final Evaluation Loss: 2.2326
# Final Perplexity: 9.32