# PyRecover

PyRecover is a robust distributed checkpointing and job management system for multi-GPU SLURM workloads. The project offers efficient training with time-aware checkpointing to maximize cluster utilization.

## Table of Contents
- [Environment Setup](#environment-setup)
- [Training](#training)
  - [Command Line Arguments](#command-line-arguments)
  - [SLURM Submission Script](#slurm-submission-script)
- [Checkpointing](#checkpointing)
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

## Training
The codebase contains example code for training a Transformer model on a parquet dataset. It's designed to work with SLURM, automatically detecting when multiple GPUs are available and enabling distributed training via DDP (DistributedDataParallel).
### Command Line Arguments
The training script () accepts various arguments to customize the training process. Here are the key parameters: `train.py`

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
| `--experiment_name`            | Name of experiment (for checkpoint subfolder) | "default-exp"                                                   |
| `--use-torch-distributed-ckpt` | Use distributed checkpointing                 | False                                                           |
| `--compile`                    | Compile model with torch.compile              | False                                                           |
| `--fused-optimizer`            | Use fused optimizer                           | False                                                           |

For a complete list of arguments, run:
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

#### Time-Aware Job Management
The script automatically computes the job end time based on the SLURM time limit and makes it available to the training script. This enables graceful stopping and checkpointing as the job approaches its time limit.

#### Example Usage
``` bash
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
## Distributed Training
For distributed training across multiple GPUs and nodes:
1. Set SLURM parameters in the submission script:
    - (for 4 GPUs per node) `--ntasks-per-node=4`
    - `--nodes=X` (where X is the number of nodes)

2. Launch with the distributed flag:
``` bash
   sbatch submit-training-simple.sh --distributed
```
This will automatically:
- Initialize process groups
- Set up data parallelism with DistributedDataParallel
- Configure distributed samplers for the dataset
