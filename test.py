from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "./qwen2-7b-model"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=60,
        temperature=0.8
    )


print(tokenizer.decode(output[0], skip_special_tokens=True))
