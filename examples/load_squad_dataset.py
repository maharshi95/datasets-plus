from hf_datasets import load_dataset

# Load SQuAD v2.0 validation set
squad_v2_val = load_dataset("squad_v2::validation")
print(f"\nLoaded SQuAD v2.0 validation set with {len(squad_v2_val)} examples")
print("First example:")
print(squad_v2_val[0])
