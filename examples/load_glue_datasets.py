from datasets_plus import load_dataset

dataset_names = [
    "glue:mnli:validation_matched",
    "glue:qnli:validation",
    "glue:qqp:validation",
    "glue:sst2:validation",
    "glue:cola:validation",
]

for name in dataset_names:
    # Load the training set for each GLUE task
    dataset = load_dataset(name)
    print(f"Loaded GLUE {name.upper()} training set with {len(dataset)} examples")
    print("First example:")
    print(dataset[0])
    print("\n" + "=" * 50 + "\n")
