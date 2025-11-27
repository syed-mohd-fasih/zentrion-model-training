from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "../checkpoints/merged-model",
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("../checkpoints/merged-model")

def ask(query, context=None):
    input_text = f"Instruction: {query}\nInput: {context or ''}\nOutput:"
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=300)
    print(tokenizer.decode(output[0], skip_special_tokens=True))

ask("Review this policy", "spec:\n  rules:\n  - to:\n    - operation: {}")
