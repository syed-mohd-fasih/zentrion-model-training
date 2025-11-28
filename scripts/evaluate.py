import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import json
from tqdm import tqdm
import os

# -----------------------
# Configuration
# -----------------------
BASE_MODEL_PATH = "../qwen2-7b-model/"
LORA_ADAPTER_PATH = "../checkpoints/logs-1"  # Your trained LoRA adapter
TEST_DATA_PATH = "../data/processed/logs.jsonl"  # Can use same data or create test split

# Set to True if you want to test on a separate test file
USE_SEPARATE_TEST_FILE = False
TEST_FILE_PATH = "../data/processed/test_logs.jsonl"

# -----------------------
# Load Tokenizer
# -----------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# -----------------------
# Quantization Config
# -----------------------
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# -----------------------
# Load Base Model
# -----------------------
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

# -----------------------
# Load LoRA Adapter
# -----------------------
print(f"Loading LoRA adapter from {LORA_ADAPTER_PATH}...")
model = PeftModel.from_pretrained(
    model,
    LORA_ADAPTER_PATH,
    torch_dtype=torch.float16,
)

# Set to evaluation mode
model.eval()

print("Model loaded successfully!")
print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

# -----------------------
# Load Test Data
# -----------------------
def load_test_data(file_path, num_samples=None):
    """Load test data from JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if num_samples and i >= num_samples:
                break
            data.append(json.loads(line))
    return data

# -----------------------
# Generate Predictions
# -----------------------
def generate_prediction(instruction, input_text, max_new_tokens=128):
    """Generate a prediction for a single input."""
    # Format the prompt
    if input_text:
        prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse:"
    else:
        prompt = f"Instruction: {instruction}\nResponse:"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response part
    response = generated_text.split("Response:")[-1].strip()
    
    return response

# -----------------------
# Evaluate Model
# -----------------------
def evaluate_model(test_data, num_samples=None):
    """Evaluate the model on test data."""
    if num_samples:
        test_data = test_data[:num_samples]
    
    print(f"\nEvaluating on {len(test_data)} samples...\n")
    
    correct = 0
    total = 0
    results = []
    
    for example in tqdm(test_data, desc="Evaluating"):
        instruction = example.get("instruction", "").strip()
        input_text = example.get("input", "").strip()
        expected_output = example.get("output", "").strip()
        
        # Generate prediction
        predicted_output = generate_prediction(instruction, input_text)
        
        # Simple exact match evaluation (you can make this more sophisticated)
        is_correct = predicted_output.lower().strip() == expected_output.lower().strip()
        
        if is_correct:
            correct += 1
        total += 1
        
        # Store result
        results.append({
            "instruction": instruction,
            "input": input_text,
            "expected": expected_output,
            "predicted": predicted_output,
            "correct": is_correct
        })
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    return results, accuracy

# -----------------------
# Interactive Testing
# -----------------------
def interactive_test():
    """Interactive mode for testing custom inputs."""
    print("\n" + "="*80)
    print("INTERACTIVE TESTING MODE")
    print("="*80)
    print("Enter your Kubernetes logs to analyze (or 'quit' to exit)")
    print("="*80 + "\n")
    
    while True:
        log_input = input("\n📋 Enter log message: ").strip()
        
        if log_input.lower() in ['quit', 'exit', 'q']:
            print("Exiting interactive mode...")
            break
        
        if not log_input:
            continue
        
        instruction = "Analyze the following Kubernetes logs and detect anomalies."
        
        print("\n🤖 Analyzing...")
        prediction = generate_prediction(instruction, log_input, max_new_tokens=256)
        
        print(f"\n✅ Model Response: {prediction}\n")
        print("-" * 80)

# -----------------------
# Save Results
# -----------------------
def save_results(results, accuracy, output_path="../checkpoints/evaluation_results.json"):
    """Save evaluation results to a JSON file."""
    output_data = {
        "accuracy": accuracy,
        "total_samples": len(results),
        "correct_predictions": sum(1 for r in results if r["correct"]),
        "results": results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")

# -----------------------
# Main Execution
# -----------------------
if __name__ == "__main__":
    print("\n" + "="*80)
    print("KUBERNETES LOG ANOMALY DETECTION - MODEL EVALUATION")
    print("="*80 + "\n")
    
    # Choose evaluation mode
    print("Select evaluation mode:")
    print("1. Automated evaluation on test set")
    print("2. Interactive testing (manual log input)")
    print("3. Both")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        # Load test data
        if USE_SEPARATE_TEST_FILE and os.path.exists(TEST_FILE_PATH):
            print(f"\nLoading test data from {TEST_FILE_PATH}...")
            test_data = load_test_data(TEST_FILE_PATH)
        else:
            print(f"\nLoading data from {TEST_DATA_PATH}...")
            # Use last 20% as test set if no separate test file
            all_data = load_test_data(TEST_DATA_PATH)
            split_idx = int(len(all_data) * 0.8)
            test_data = all_data[split_idx:]
            print(f"Using last {len(test_data)} samples as test set (20% of data)")
        
        # Ask how many samples to evaluate
        num_samples_input = input(f"\nHow many samples to evaluate? (Enter number or 'all' for all {len(test_data)}): ").strip()
        
        if num_samples_input.lower() == 'all':
            num_samples = None
        else:
            try:
                num_samples = int(num_samples_input)
            except:
                print("Invalid input, using 10 samples...")
                num_samples = 10
        
        # Run evaluation
        results, accuracy = evaluate_model(test_data, num_samples)
        
        # Print results
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Correct: {sum(1 for r in results if r['correct'])}/{len(results)}")
        print("="*80 + "\n")
        
        # Show some examples
        print("Sample predictions:\n")
        for i, result in enumerate(results[:5]):
            print(f"Example {i+1}:")
            print(f"  Input: {result['input'][:100]}...")
            print(f"  Expected: {result['expected']}")
            print(f"  Predicted: {result['predicted']}")
            print(f"  ✓ Correct" if result['correct'] else "  ✗ Incorrect")
            print()
        
        # Save results
        save_results(results, accuracy)
    
    if choice in ['2', '3']:
        # Interactive mode
        interactive_test()
    
    print("\n✅ Evaluation complete!")