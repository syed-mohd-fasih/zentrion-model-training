import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer
from peft import PeftModel


MODEL_NAME = "./qwen2-7b-model"

# -----------------------
# Tokenizer
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# -----------------------
# Dataset
# -----------------------
dataset = load_dataset("json", data_files={
    "train": [
        "../data/logs.jsonl",
        # "../data/policies.jsonl",
        # "../data/manifests.jsonl",
        # "../data/qa_reasoning.jsonl"
    ]
})

# -----------------------
# Quantization (VERY IMPORTANT)
# -----------------------
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# -----------------------
# Load base model ================> (Change to checkpoints)
# -----------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16
)

### For continuous training
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     quantization_config=quant_config,
#     device_map="auto"
# )

# model = PeftModel.from_pretrained(
#     model,
#     "../checkpoints/lora-final"
# )

# -----------------------
# LoRA Config
# -----------------------
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=[
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# -----------------------
# Training Arguments
# -----------------------
training_args = TrainingArguments(
    output_dir="../checkpoints/",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    learning_rate=2e-4,
    fp16=True,
    bf16=False,
    optim="paged_adamw_8bit",
    max_grad_norm=1.0,
)

# -----------------------
# Trainer
# -----------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
    args=training_args,
    max_seq_length=4096
)

trainer.train()

# -----------------------
# Save LoRA
# -----------------------
model.save_pretrained("../checkpoints/lora-final")
