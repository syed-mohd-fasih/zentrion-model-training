import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer
import gc

# -----------------------
# Model Path
# -----------------------
MODEL_NAME = "../qwen2-7b-model/"

# -----------------------
# Tokenizer
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# -----------------------
# Dataset
# -----------------------
dataset = load_dataset("json", data_files={
    "train": [
        "../data/processed/logs.jsonl",
        # "../data/processed/policies.jsonl",
        # "../data/processed/manifests.jsonl",
        # "../data/processed/qa_reasoning.jsonl"
    ]
})
dataset = dataset.shuffle(seed=42)

# -----------------------
# Quantization (VERY IMPORTANT)
# -----------------------
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=False,
)

# -----------------------
# Load base model ================> (Change to checkpoints)
# -----------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    max_memory={0: "19GB"},
)

# Prepare model for k-bit training (CRITICAL for memory)
model = prepare_model_for_kbit_training(model)

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# -----------------------
# Load previous LoRA adapter (resume training)
# -----------------------
# model = PeftModel.from_pretrained(
#     model,
#     "../checkpoints/logs-1",
#     is_trainable=True
# )

# -----------------------
# LoRA Config
# -----------------------
lora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    target_modules=[
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()

# -----------------------
# Training Arguments
# -----------------------
training_args = TrainingArguments(
    output_dir="../checkpoints/",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=5,
    save_steps=250,
    save_total_limit=2,
    learning_rate=2e-4,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_pin_memory=False,
    remove_unused_columns=True,
    report_to="none",
)

# -----------------------
# Data Formatter
# -----------------------
def format_example(example):
    """Format a single example into a prompt string."""
    # Handle both single examples and batched examples
    if isinstance(example.get("instruction"), list):
        # Batched processing
        formatted_texts = []
        for i in range(len(example["instruction"])):
            instruction = example["instruction"][i].strip()
            input_text = example.get("input", [""])[i].strip()
            output = example["output"][i].strip()
            
            if input_text:
                prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse: {output}"
            else:
                prompt = f"Instruction: {instruction}\nResponse: {output}"
            formatted_texts.append(prompt)
        return formatted_texts
    else:
        # Single example processing
        instruction = example.get("instruction", "").strip()
        input_text = example.get("input", "").strip()
        output = example.get("output", "").strip()

        if input_text:
            prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse: {output}"
        else:
            prompt = f"Instruction: {instruction}\nResponse: {output}"
        return prompt

# -----------------------
# Clear any cached memory
# -----------------------
gc.collect()
torch.cuda.empty_cache()

# -----------------------
# Trainer
# -----------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
    args=training_args,
    max_seq_length=2048,
    packing=False,
    formatting_func=format_example,
    dataset_text_field=None,
)

# -----------------------
# Start Training
# -----------------------
print("Starting training...")
print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print(f"GPU Memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
trainer.train()

# -----------------------
# Save LoRA (Change accordingly)
# -----------------------
model.save_pretrained("../checkpoints/logs-1")
tokenizer.save_pretrained("../checkpoints/logs-1")

# model.save_pretrained("../checkpoints/policies-1")
# tokenizer.save_pretrained("../checkpoints/policies-1")

# model.save_pretrained("../checkpoints/manifests-1")
# tokenizer.save_pretrained("../checkpoints/manifests-1")

# model.save_pretrained("../checkpoints/qa-1")
# tokenizer.save_pretrained("../checkpoints/qa-1")


print("Training complete!")
print(f"Final GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")