# Detailed Script Explanations

This section provides a comprehensive explanation of the core scripts in PyRecover.

---

## 1. `dataset.py`

### Purpose

`dataset.py` provides a PyTorch-compatible dataset class for reading and tokenizing text data from Parquet files, as well as a collator for preparing batches for causal language modeling (CLM). This is essential for training transformer models on large text datasets.

### Key Components

#### - `ParquetDataset` class

- **Inherits from:** `torch.utils.data.Dataset`
- **Purpose:** Reads text samples from a Parquet file, tokenizes them, and prepares them for training.
- **Constructor Arguments:**
  - `parquet_file`: Path to the Parquet file containing text data.
  - `tokenizer`: A tokenizer object (e.g., from HuggingFace Transformers).
  - `sequence_length`: The length of each training sequence (number of tokens).
  - `training_samples`: The number of samples to provide (can be larger than the number of rows in the Parquet file, in which case it wraps around).
- **How it works:**
  - Loads the Parquet file into memory using `pyarrow`.
  - Stores the tokenizer and sequence length.
  - Implements `__len__` to return the number of training samples.
  - Implements `__getitem__` to:
    - Fetch a text sample by index (wrapping around if needed).
    - Tokenize the sample using the tokenizer's `encode_plus` method, which:
      - Converts text to a list of token IDs.
      - Pads or truncates to `sequence_length + 1` (the extra token is for next-token prediction).
      - Returns a dictionary with keys like `"input_ids"`.

#### - `CollatorForCLM` dataclass

- **Purpose:** Prepares batches of tokenized data for training a causal language model.
- **Constructor Arguments:**
  - `sequence_length`: The length of each sequence (number of tokens).
  - `pad_token_id`: The token ID used for padding.
- **How it works:**
  - Implements `__call__`, which takes a list of tokenized examples and:
    - Stacks them into a tensor of shape `(batch_size, sequence_length + 1)`.
    - Splits each sequence into:
      - `inputs`: All tokens except the last one.
      - `labels`: All tokens except the first one.
    - Sets label positions corresponding to padding tokens to `-100` (ignored in loss computation).
    - Returns a tuple `(inputs, labels)`.

### How to Use

- Instantiate a `ParquetDataset` with your Parquet file, tokenizer, sequence length, and desired number of samples.
- Use `CollatorForCLM` as the `collate_fn` in a PyTorch `DataLoader` to automatically batch and prepare data for training.

---

## 2. `iterable_dataset.py`

### Purpose

`iterable_dataset.py` provides an alternative dataset class, `IterableParquetDataset`, which is an `IterableDataset` for streaming data from a Parquet file. This is useful for very large datasets that don't fit in memory or when you want to process data sequentially.

### Key Components

#### - `IterableParquetDataset` class

- **Inherits from:** `torch.utils.data.IterableDataset`
- **Purpose:** Streams tokenized text samples from a Parquet file, yielding input-label pairs for language modeling.
- **Constructor Arguments:**
  - `parquet_file`: Path to the Parquet file.
  - `tokenizer`: A tokenizer object.
  - `sequence_length`: Number of tokens per sequence.
  - `bos_token_id`: Token ID for the beginning-of-sequence token (default: 1).
- **How it works:**
  - Loads the Parquet file into memory.
  - Initializes a buffer for tokens and an index pointer.
  - Implements `__iter__` to reset the buffer and index for each new iterator.
  - Implements `__next__` to:
    - Fill the token buffer with enough tokens to create a sequence of length `sequence_length + 1`.
    - For each new sample:
      - Reads the next text entry.
      - Tokenizes it (without adding special tokens).
      - Prepends the BOS token.
      - Adds tokens to the buffer.
    - Once enough tokens are available:
      - Extracts a chunk of `sequence_length + 1` tokens.
      - Converts the first `sequence_length` tokens to `inputs`.
      - Converts the next `sequence_length` tokens to `labels`.
      - Sets label positions corresponding to BOS tokens to `-100`.
      - Yields `(inputs, labels)` as tensors.
    - Raises `StopIteration` when the end of the dataset is reached.

### How to Use

- Instantiate `IterableParquetDataset` with your Parquet file, tokenizer, and sequence length.
- Use it with a PyTorch `DataLoader` (with `batch_size=None` or `batch_size=1` for streaming).
- Each iteration yields a tuple `(inputs, labels)` ready for training.

---

## 3. `model.py`

### Purpose

`model.py` defines the architecture of the transformer model used for language modeling. It includes all the building blocks: normalization, attention, feedforward layers, and the overall transformer stack.

### Key Components

#### - `TransformerModelArgs` dataclass

- **Purpose:** Stores all configuration parameters for the transformer model (dimensions, number of layers, etc.).
- **Fields:** Includes `dim`, `n_layers`, `n_heads`, `n_kv_heads`, `multiple_of`, `ffn_dim_multiplier`, `norm_eps`, `rope_theta`, `norm_type`, `seq_len`, `vocab_size`.

#### - `RMSNorm` class

- **Purpose:** Implements Root Mean Square Layer Normalization, a variant of layer normalization.
- **How it works:**
  - Computes the RMS of the input tensor along the last dimension.
  - Scales the input by the inverse RMS and a learnable weight.

#### - Rotary Embedding Functions

- **`precompute_freqs_cis`**: Precomputes complex exponentials for rotary positional embeddings.
- **`reshape_for_broadcast`**: Reshapes frequency tensors for broadcasting.
- **`apply_rotary_emb`**: Applies rotary embeddings to query and key tensors.

#### - `Attention` class

- **Purpose:** Implements multi-head self-attention with rotary embeddings.
- **How it works:**
  - Projects input to queries, keys, and values.
  - Applies rotary embeddings.
  - Handles key/value head repetition if needed.
  - Computes scaled dot-product attention with causal masking.
  - Projects output back to model dimension.

#### - `FeedForward` class

- **Purpose:** Implements the feedforward (MLP) block of the transformer.
- **How it works:**
  - Projects input to a hidden dimension (with optional multiplier and rounding).
  - Applies SiLU activation and elementwise multiplication.
  - Projects back to model dimension.

#### - `TransformerBlock` class

- **Purpose:** Represents a single transformer block (attention + feedforward + normalization).
- **How it works:**
  - Applies attention (with normalization and residual connection).
  - Applies feedforward (with normalization and residual connection).

#### - `Transformer` class

- **Purpose:** The full transformer model for language modeling.
- **How it works:**
  - Embeds input tokens.
  - Applies a stack of `TransformerBlock`s.
  - Normalizes the output.
  - Projects to vocabulary size for logits.

### How to Use

- Instantiate `TransformerModelArgs` with desired configuration.
- Create a `Transformer` model with these arguments.
- Pass tokenized input tensors to the model to get logits for language modeling.

---

## 4. `train.py`

### Purpose

`train.py` is the main training script. It orchestrates data loading, model creation, optimizer and scheduler setup, and the training loop.

### Key Components

#### - Imports

- Loads all necessary modules, including PyTorch, data utilities, model, and helper functions from `utils.py`.

#### - `train(args)` function

- **Purpose:** Runs the training loop for the transformer model.
- **How it works:**
  - Logs experiment arguments.
  - Sets up the device (GPU).
  - Determines model data type (e.g., bf16, fp16).
  - Loads the tokenizer.
  - Creates the training dataset and data loader.
  - Instantiates the model with the specified configuration and moves it to the device.
  - Optionally compiles the model with `torch.compile` for performance.
  - Sets up the optimizer (AdamW) and learning rate scheduler.
  - Computes FLOPs per token for logging efficiency metrics.
  - Initializes counters for tokens and timing.
  - Enters the training loop:
    - Optionally starts profiling (for performance analysis).
    - Fetches the next batch of data.
    - Moves data to the device.
    - Zeroes gradients.
    - Computes logits and loss (cross-entropy).
    - Normalizes loss by number of training tokens.
    - Backpropagates gradients.
    - (Optionally) clips gradients.
    - Steps the optimizer and scheduler.
    - Logs metrics (loss, tokens/sec, MFU, TFLOPs) at specified intervals.
    - Optionally stops profiling.
  - Logs completion.

#### - Main Block

- Initializes the logger.
- Parses command-line arguments.
- Calls `train(args)`.

### How to Use

- Run `python train.py` from the command line.
- Pass arguments as needed (see `utils.py` for available arguments).
- The script will train the transformer model on your dataset, logging progress and metrics.

---

## 5. `utils.py`

### Purpose

`utils.py` provides utility functions and argument parsing for the training pipeline. These functions handle logging, learning rate scheduling, gradient clipping, argument parsing, and more.

### Key Components

#### - Logging Utilities

- **`init_logger()`**: Configures the global logger to print messages with timestamps and levels.

#### - Precision Mapping

- **`PRECISION_STR_TO_DTYPE`**: Maps string names (e.g., "fp16", "bf16") to PyTorch data types.

#### - Model Utilities

- **`get_num_params(model, exclude_embedding=False)`**: Returns the number of parameters in a model, optionally excluding embedding layers.
- **`get_num_flop_per_token(num_params, model_config)`**: Estimates the number of floating-point operations per token for the model, based on its configuration.

#### - Learning Rate Scheduler

- **`build_lr_scheduler(optimizer, warmup_steps)`**: Returns a PyTorch LambdaLR scheduler that linearly warms up the learning rate for a specified number of steps, then keeps it constant.

#### - Gradient Clipping

- **`clip_grad_norm_(parameters, grad_max_norm)`**: Clips gradients to a maximum norm (uses PyTorch utilities).

#### - Context Managers

- **`set_default_dtype(dtype)`**: Temporarily sets the default PyTorch data type within a context (useful for mixed precision training).

#### - Argument Parsing

- **`get_args()`**: Defines and parses command-line arguments for the training script. Arguments include dataset path, tokenizer, sequence length, batch size, optimizer options, learning rate, warmup steps, training steps, logging frequency, profiling options, gradient clipping, model dtype, and whether to use `torch.compile`.

### How to Use

- Import and use these utilities in your training scripts.
- Use `get_args()` to parse command-line arguments.
- Use `init_logger()` to set up logging.
- Use `build_lr_scheduler()` and `clip_grad_norm_()` in your training loop as needed.

---

# Summary Table

| Script                | Purpose                                                    | How to Use                                                                            |
| --------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `dataset.py`          | Loads and tokenizes data from Parquet files for training   | Instantiate `ParquetDataset` and use with `CollatorForCLM` in a DataLoader            |
| `iterable_dataset.py` | Streams tokenized data from Parquet files (for large data) | Instantiate `IterableParquetDataset` and use with a DataLoader (batch_size=1 or None) |
| `model.py`            | Defines the transformer model architecture                 | Instantiate `TransformerModelArgs` and `Transformer`, then call with input tensors    |
| `train.py`            | Main training script                                       | Run `python train.py` with appropriate arguments                                      |
| `utils.py`            | Utility functions for logging, scheduling, parsing, etc.   | Import and use functions as needed in your scripts                                    |

---

# Example Workflow

1. **Prepare your dataset** as a Parquet file with a `"text"` column.
2. **Set up your environment** using the provided `env.yml`.
3. **Run training**:
   ```
   python train.py --dataset /path/to/your_data.parquet --tokenizer-name-or-path your_tokenizer
   ```
   - Adjust other arguments as needed (see `utils.py` for all options).
4. **Monitor logs** for training progress, loss, and efficiency metrics.

---

# Additional Notes

- All scripts are designed to be modular and extensible.
- You can swap out the dataset or model with your own implementations if needed.
- For distributed or multi-GPU training, integrate with SLURM or other job schedulers as appropriate.
- Profiling options are available for performance tuning.

---

# Troubleshooting

- **Out of Memory:** Reduce `batch_size` or `sequence_length`.
- **Tokenizer Errors:** Ensure the tokenizer path or name is correct and compatible with your dataset.
- **Parquet Errors:** Verify your Parquet file has a `"text"` column and is readable by `pyarrow`.

---

# References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [PyArrow Documentation](https://arrow.apache.org/docs/python/)

---

# Additional Details from Source Code

## `utils.py` (Additional Details)

- **Gradient Clipping**: The function `clip_grad_norm_()` uses PyTorch's `get_total_norm` and `clip_grads_with_norm_` to clip gradients, ensuring training stability.
- **Precision Mapping**: The dictionary `PRECISION_STR_TO_DTYPE` allows flexible selection of model precision (fp16, bf16, fp32, fp64) via command-line arguments.
- **Context Manager**: `set_default_dtype()` is a context manager to temporarily set the default PyTorch dtype, useful for mixed-precision training.
- **Argument Parsing**: `get_args()` provides a comprehensive set of command-line arguments, including options for distributed training, checkpointing, profiling, logging, and model configuration.

## `model.py` (Additional Details)

- **TransformerModelArgs**: A dataclass for all model hyperparameters, including support for rotary embeddings and flash attention.
- **RMSNorm**: Implements Root Mean Square Layer Normalization for stable training.
- **Rotary Embedding Functions**: Includes `precompute_freqs_cis`, `reshape_for_broadcast`, and `apply_rotary_emb` for efficient rotary positional encoding.
- **Attention**: Multi-head self-attention with support for rotary embeddings and optional flash attention. Handles key/value head repetition for multi-query attention.
- **FeedForward**: Implements the MLP block with configurable hidden dimension and activation.
- **TransformerBlock**: Stacks attention and feedforward layers with normalization and residual connections.
- **Transformer**: The main model class, stacks multiple `TransformerBlock`s, applies token embeddings, normalization, and projects to vocabulary logits.

## `dist_utils.py` (Additional Details)

- **Distributed Setup**: Functions to initialize and clean up PyTorch DDP using SLURM environment variables. Handles rank, world size, and device assignment.
- **Logging**: `log_rank0()` and `log_rank()` ensure only rank 0 (or a specific rank) logs messages, avoiding clutter in distributed runs.
- **SLURM Integration**: `get_slurm_job_end_time_env()` fetches the SLURM job end time for time-aware checkpointing and graceful shutdown.

## `train.py` (Additional Details)

- **Training Loop**: Handles distributed and non-distributed training, model/optimizer/scheduler setup, and checkpointing (vanilla or torch.distributed.checkpoint).
- **Profiling**: Supports NSYS profiling via command-line flags, with start/end step control.
- **Loss Logging**: Optionally logs loss to CSV for later analysis.
- **Time-Aware Checkpointing**: Optionally monitors SLURM job end time to save a final checkpoint before the job is killed.
- **Resuming**: Can resume from latest or specified checkpoint, restoring model, optimizer, and scheduler state.
- **Device Management**: Automatically selects the correct CUDA device for each process in distributed mode.

---

# Quick Reference: Key Features

- **Flexible Model Configuration**: All model and training hyperparameters are configurable via command-line arguments.
- **Distributed Training**: Seamless support for single-node and multi-node distributed training with SLURM and PyTorch DDP.
- **Checkpointing**: Supports both vanilla PyTorch and torch.distributed.checkpoint for efficient and robust checkpointing.
- **Profiling**: Integrated NSYS profiling for performance analysis, with step-based control.
- **Logging**: Detailed logging to console and optional CSV for loss tracking.
- **Time-Aware Checkpointing**: Prevents job preemption by saving a final checkpoint before SLURM timeouts.
- **Rotary Embeddings & Flash Attention**: Modern transformer features for improved efficiency and scalability.

---

# For More Information

- See code comments in each file for further details and usage examples.
- For troubleshooting and advanced usage, consult the function docstrings and argument help messages (`python train.py --help`).

# End of Documentation