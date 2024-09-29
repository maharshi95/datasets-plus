from datasets_plus.core import load_dataset

try:
    # Attempt to load a BeIR dataset
    dataset = load_dataset("beir/nfcorpus:train")

    print(f"Loaded dataset with {len(dataset)} examples")
    print("First example:")
    print(dataset[0])
except NotImplementedError:
    print("BeIR dataset loading is not yet implemented")
