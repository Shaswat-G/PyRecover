# PyRecover

PyRecover is a robust distributed checkpointing and job management system for multi-GPU SLURM workloads. The project offers efficient training with time-aware checkpointing to maximize cluster utilization.

## Table of Contents
- [PyRecover](#pyrecover)
  - [Table of Contents](#table-of-contents)
  - [Environment Setup](#environment-setup)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Training](#training)
    - [Command Line Arguments](#command-line-arguments)
    - [SLURM Submission Script](#slurm-submission-script)
      - [Key Parameters](#key-parameters)
      - [Script Arguments](#script-arguments)
      - [Time-Aware Job Management](#time-aware-job-management)
      - [Example Usage](#example-usage)
  - [Checkpointing](#checkpointing)
  - [Time-Aware Checkpointing](#time-aware-checkpointing)
  - [Distributed Training](#distributed-training)

## Environment Setup

Shows environment creation with conda, but principally also other tools such as venv can be used.

### Prerequisites

- Miniconda or Anaconda

### Installation

1. Clone the repository
2. Set up the environment with miniconda:

```bash
conda env create -f env.yml
```

This will create an environment called "pyrecover" based on python 3.10.
Use this environment for development. Activate it by calling:

```
conda activate pyrecover
```

#### Installation with Flash Attention

To install with Flash Attention support, ensure you have the following prerequisites:
- CUDA toolkit (compatible with your PyTorch installation)
- C++ compiler (gcc/g++)
- Python development headers

Then install with:
```
./setup_flashattention.sh
```
or
```
pip install ".[flash-attention]"
```

After this you can activate flash attention as argument. The isntalletion can take un to 2h.
In the slurm script flash attention is attempted to be installed if its activated. This is to make slurm runs as stateless as possible without needing this installation in a container or environment before.

## Training

The codebase contains example code for training a Transformer model on a parquet dataset. It's designed to work with SLURM, automatically detecting when multiple GPUs are available and enabling distributed training via DDP (DistributedDataParallel).


### Command Line Arguments

The training script (`train.py`) accepts various arguments to customize the training process. Here are the key parameters:

| Argument                       | Description                                   | Default                                                         |
|--------------------------------|-----------------------------------------------|-----------------------------------------------------------------|
| `--dataset`                    | Path to parquet file with text data           | `/capstor/store/cscs/ethz/large-sc/datasets/train_data.parquet` |
| `--sequence-length`            | Maximum sequence length                       | 2048                                                            |
| `--batch-size`                 | Batch size per GPU                            | 1                                                               |
| `--learning-rate`              | Learning rate                                 | 1e-5                                                            |
| `--training-steps`             | Number of training steps                      | 1000                                                            |
| `--distributed`                | Enable distributed training                   | False                                                           |
| `--model-dtype`                | Model precision (fp16/bf16/fp32/fp64)         | "bf16"                                                          |
| `--checkpoint-dir`             | Directory for checkpoints                     | "checkpoints/"                                                  |
| `--checkpoint-frequency`       | Save checkpoint every N steps                 | 10                                                              |
| `--resume-from-checkpoint`     | Path to checkpoint or "latest"                | None                                                            |
| `--profile`                    | activates profiling support for nsys          | False                                                           |
| `--experiment_name`            | Name of experiment (for checkpoint subfolder) | "default-exp"                                                   |
| `--use-torch-distributed-ckpt` | Use distributed checkpointing                 | False                                                           |
| `--compile`                    | Compile model with torch.compile              | False                                                           |
| `--fused-optimizer`            | Use fused optimizer                           | False                                                           |
| `--use_flash_attention`        | Use flash-attention in the model              | False                                                           |
| `--log-loss-to-csv`            | Log loss to a csv for plots/comparison        | False                                                           |
| `--timeaware-checkpointing`    | Activates time aware checkpointing            | False                                                           |

For a complete list of arguments, run:
```bash
python train.py --help
```

### Command Line Arguments

The training script accepts various arguments to customize the training process. Here are the key parameters: `train.py`

### Running non distributes training

Make sure to set `#SBATCH --ntasks-per-node=1` this way only one process is spawned on a node. The code uses DDP and one process will only make use of one gpu.

```bash
python train.py --help
```

### SLURM Submission Script

The script is provided for launching training jobs on SLURM clusters. `submit-training-simple.sh`

#### Key Parameters

These key parameters can be adapted by editing the script.

| SLURM Parameter     | Description                          |
|---------------------|--------------------------------------|
| `--nodes`           | Number of nodes to allocate          |
| `--ntasks-per-node` | Tasks per node (typically 1 per GPU) |
| `--gpus-per-node`   | GPUs to use per node                 |
| `--time`            | Time limit for the job               |
| `--partition`       | SLURM partition to use               |

#### Script Arguments

The submission script supports the following arguments:

| Argument                       | Description                                        |
|--------------------------------|----------------------------------------------------|
| `--distributed`                | Enable distributed training                        |
| `--exp_name=NAME`              | Set experiment name (affects checkpoint subfolder) |
| `--continue`                   | Resume from latest checkpoint                      |
| `--use_torch_distributed_ckpt` | Use torch distributed checkpointing                |
| `--use_flash_attention`        | Use and install flash-attention in the model       |
| `--log-loss-to-csv`            | Log the loss of the training to a csv file         |
| `--timeaware-checkpointing`    | Activate time-aware checkpointing in train script  |
| `--fused-optimizer`            | Activate using the fused optimizer for training    |
| `--profile-nsys`               | Run the nsys profiling. Only support run with one GPU, so adapt batch script accordingly  |

#### Time-Aware Job Management

The script automatically computes the job end time based on the SLURM time limit and makes it available to the training script. This enables graceful stopping and checkpointing as the job approaches its time limit.

#### Example Usage

```bash
# Non-distributed training
sbatch submit-training-simple.sh --exp_name=my_experiment

# Distributed training on multiple GPUs
sbatch submit-training-simple.sh --distributed --exp_name=distributed_exp

# Resume from checkpoint with distributed checkpointing
sbatch submit-training-simple.sh --distributed --continue --use_torch_distributed_ckpt
```

## Checkpointing

PyRecover offers two checkpointing methods:

1. **Vanilla Checkpointing**: Standard PyTorch checkpointing (default)

   - Use with standard submission script without flags

2. **Distributed Checkpointing**: Faster loading/saving for large models (45+ GB)
   - Enable with flag `--use_torch_distributed_ckpt`

Checkpoints are automatically organized by experiment name, allowing you to run multiple experiments without overwriting previous results.

## Time-Aware Checkpointing

Time-aware checkpointing enables the training script to monitor the remaining SLURM job time and automatically trigger a final checkpoint and graceful exit before the job ends. This prevents loss of training progress due to walltime limits.

- Activate by adding the `--timeaware-checkpointing` flag to your training command or SLURM script.
- The script dynamically tracks iteration and checkpoint durations to calculate a safe stopping threshold.
- As the job nears its end, a final checkpoint is saved and the process exits cleanly, allowing seamless resumption.

**Example usage:**

```bash
sbatch submit-training-simple.sh --distributed --timeaware-checkpointing
```

## Distributed Training

For distributed training across multiple GPUs and nodes:

1. Set SLURM parameters in the submission script:

   - (for 4 GPUs per node) `--ntasks-per-node=4`
   - `--nodes=X` (where X is the number of nodes)

2. Launch with the distributed flag:

```bash
   sbatch submit-training-simple.sh --distributed
```
This will automatically:
- Initialize process groups
- Set up data parallelism with DistributedDataParallel
- Configure distributed samplers for the dataset

## Benchmarks

To test the checkpointing we employ different benchmark possibilities. This is either enabled by separate scripts or by setting cmd args.
For some it is even enough to look at the output.

### Check equality of weights
With and without checkpointing or continue from checkpoint we can reach two final checkpoints.
Make sure training is done with same hyperparams and training-args and use the same fixed seed.
Then use the script `tests/check_weights_equality.py` and give the path to two checkpoints as arguments.

#### Usage
``` bash
python check_weights_equality.py <checkpoint1> <checkpoint2> [--distributed] [--tolerance 1e-7] [--verbose]
```

#### Arguments
- `checkpoint1`: Path to the first checkpoint
- `checkpoint2`: Path to the second checkpoint
- : Use this flag if the checkpoints were saved using distributed checkpointing `--distributed`
- `--tolerance`: Floating point tolerance for comparison (default: 1e-7)
- `--verbose`: Enable detailed output of differences

### Loss convergence
To compare loss convergence with and without checkpointing, we add the possibility to log loss values for each step to a csv file that will be stored in the experiment folder.
Just add the parameter: `--log-loss-to-csv`.