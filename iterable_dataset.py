import pyarrow.parquet as pq
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
import torch

class IterableParquetDataset(IterableDataset):
    def __init__(self, parquet_file: str, tokenizer: str, sequence_length: int, bos_token_id: int = 1):
        
        self.parquet_ds = pq.read_table(parquet_file, memory_map=True)
        self.real_length = len(self.parquet_ds)
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        
        self.bos_token_id = bos_token_id
        self.token_buffer = []
        self.current_index = 0
        

    def __iter__(self):
        self.current_index = 0
        self.token_buffer = []
        return self
    
    def __next__(self):
        
        while len(self.token_buffer) < (self.sequence_length + 1):
            if self.current_index >= self.real_length:
                raise StopIteration
        
            sample_str = str(self.parquet_ds["text"][self.current_index])
            encoded_sample = self.tokenizer.encode(sample_str, add_special_tokens=False)
            self.token_buffer.append(self.bos_token_id)
            self.token_buffer.extend(encoded_sample)
            self.current_index += 1
            
        chunk = self.token_buffer[:(self.sequence_length + 1)]
        self.token_buffer = self.token_buffer[(self.sequence_length+1):]
        
        inputs = torch.LongTensor(chunk[:-1])
        labels = torch.LongTensor(chunk[1:])

        bos_positions = (inputs == self.bos_token_id).nonzero(as_tuple=True)[0]
        for pos in bos_positions:
            if pos <= (self.sequence_length-1):
                labels[pos] = -100
        
        yield inputs, labels

