from transformers import AutoTokenizer
from dataset import ParquetDataset
import torch

# 1. Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("unsloth/Mistral-Nemo-Base-2407-bnb-4bit")

# 2. Define dataset path for cluster
dataset_path = "/capstor/store/cscs/ethz/large-sc/datasets/train_data.parquet"

# 3. Define the sequence length
sequence_length = 4096

# 4. Define the number of training samples
training_samples = 1
dataset = ParquetDataset(dataset_path, tokenizer, sequence_length, training_samples)

# 5. Get the first sample
sample = dataset[0]

# 6. Print the sample
input_ids = sample["input_ids"]
first_200_tokens = input_ids[:200]   # slice the first 200 token IDs
decoded_200_tokens = tokenizer.decode(first_200_tokens)

with open("decoded_200_tokens.txt", "w") as f:
    f.write(decoded_200_tokens)

print(f"Decoded first 200 tokens: {decoded_200_tokens}")