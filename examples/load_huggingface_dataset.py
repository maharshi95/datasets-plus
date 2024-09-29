from datasets_plus.core import load_dataset

# Load a HuggingFace dataset
dataset = load_dataset("squad:v1.1:train")

print(f"Loaded dataset with {len(dataset)} examples")
print("First example:")
print(dataset[0])
