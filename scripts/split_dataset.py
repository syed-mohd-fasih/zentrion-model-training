import json
import random

# Set seed for reproducibility
random.seed(42)

# Load all data
input_file = "../data/processed/logs.jsonl"
train_file = "../data/processed/train_logs.jsonl"
test_file = "../data/processed/test_logs.jsonl"

print("Loading data...")
with open(input_file, 'r') as f:
    data = [json.loads(line) for line in f]

print(f"Total samples: {len(data)}")

# Shuffle
random.shuffle(data)

# Split 80/20
split_idx = int(len(data) * 0.8)
train_data = data[:split_idx]
test_data = data[split_idx:]

print(f"Train samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")

# Save splits
with open(train_file, 'w') as f:
    for item in train_data:
        f.write(json.dumps(item) + '\n')

with open(test_file, 'w') as f:
    for item in test_data:
        f.write(json.dumps(item) + '\n')

print(f"\n✅ Saved to:")
print(f"   {train_file}")
print(f"   {test_file}")