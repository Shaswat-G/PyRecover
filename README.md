# PyRecover

Robust distributed checkpointing and job management system for multi-GPU SLURM workloads

## Environment

For setting up environment with miniconda:

```
conda env create -f env.yml
```

This will create an environment called "pyrecover" based on python 3.10.
Use this environment for development. Activate it by calling:

```
conda activate pyrecover
```

## Training details

The codebase contains example code for training a Transformer on a parquet dataset given to us by the lecturers.
The code is setup to work with slurm. It detects whether the slurm environment has multiple GPU's available and if 
distributed training is activated: activated DDP in this case.