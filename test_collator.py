from dataset import ParquetDataset, CollatorForCLM
from iterable_dataset import IterableParquetDataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import argparse

def main(use_iterable_dset: bool = False):
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Mistral-Nemo-Base-2407-bnb-4bit")
    dataset_path = "/capstor/store/cscs/ethz/large-sc/datasets/train_data.parquet"
    
    sequence_length = 4096
    training_samples = 32
    batch_size = 32

    if use_iterable_dset:
        dataset = IterableParquetDataset(dataset_path, tokenizer, sequence_length)
    else:
        dataset = ParquetDataset(dataset_path, tokenizer, sequence_length, training_samples)

    # Setup collator and dataloader
    collator = CollatorForCLM(sequence_length=sequence_length, pad_token_id=tokenizer.pad_token_id)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    
    for batch_inputs, batch_labels in dataloader:
        
        print(f"Input shape: {batch_inputs.shape}")
        print(f"Labels shape: {batch_labels.shape}")

        ignored_count = (batch_labels == -100).sum().item()
        total_label_tokens = batch_labels.numel()
        ignored_pct = (ignored_count / total_label_tokens) * 100

        print(f"Ignored tokens in loss: {ignored_count} out of {total_label_tokens} "
              f"({ignored_pct:.2f}%)")

        break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterable_dset", "-id", action="store_true", help="Use IterableDataset")
    args = parser.parse_args()
    main(args.iterable_dset)