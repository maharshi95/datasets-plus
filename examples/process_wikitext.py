from datasets_plus import load_dataset

# Load WikiText-2 raw dataset
wikitext = load_dataset("wikitext:wikitext-2-raw-v1:train")
print(f"Loaded WikiText-2 raw training set with {len(wikitext)} examples")


# Process the dataset to create a language modeling dataset
def preprocess_function(examples):
    return {"input_ids": [text.split() for text in examples["text"]]}


tokenized_wikitext = wikitext.map(
    preprocess_function,
    batched=True,
    remove_columns=wikitext.column_names,
    desc="Running tokenizer on dataset",
)

print("\nProcessed WikiText-2 dataset:")
print(f"Number of examples: {len(tokenized_wikitext)}")
print("First example:")
print(tokenized_wikitext[0])

# Calculate vocabulary size
vocab = set(word for example in tokenized_wikitext["input_ids"] for word in example)
print(f"\nVocabulary size: {len(vocab)}")
