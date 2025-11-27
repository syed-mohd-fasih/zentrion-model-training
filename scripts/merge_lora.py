from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model = AutoPeftModelForCausalLM.from_pretrained(
    "../checkpoints/lora-final",
    device_map="auto"
).merge_and_unload()

model.save_pretrained("../checkpoints/merged-model")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
tokenizer.save_pretrained("../checkpoints/merged-model")
