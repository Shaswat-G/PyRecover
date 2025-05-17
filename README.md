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

In general we provide the script `submit-training-simple.sh` to lauch a training on slurm. You can go in there and change the the cmd-line params for the python script. Please look at utils.py for run `python3 train.py --help` to get an overview of options. 

### Running non distributes training

Make sure to set `#SBATCH --ntasks-per-node=1` this way only one process is spawned on a node. The code uses DDP and one process will only make use of one gpu.

```
sbatch submit-training-simple.sh
```

### Running distributed training

Make sure to set `#SBATCH --ntasks-per-node=4` (on the cluster we use, one node has 4 gpus) so DDP can make use of all gpus (one per process). Also set `#SBATCH --nodes=X` with X > 1 if you want train on multiple nodes!

```
sbatch submit-training-simple.sh --distributed
```