#!/bin/bash
#SBATCH --job-name=llm_benchmark  # A name for your job. Visible in squeue.
#SBATCH --account=a-large-sc
#SBATCH --nodes=1 # Number of compute nodes to request.
#SBATCH --ntasks-per-node=1 # Tasks (processes) per node
#SBATCH --time=00:44:00 # HH:MM:SS, set a time limit for this job (here 4 hours)
#SBATCH --partition=debug # Partition to use; "debug" is usually for quick tests
#SBATCH --mem=460000 # Memory needed (simply set the mem of a node)
#SBATCH --cpus-per-task=288 # CPU cores per task (simply set the number of cpus a node has)
#SBATCH --environment=/users/rkreft/scratch/ngc_pt_jan.toml # the environment to use
#SBATCH --output=/iopsstor/scratch/cscs/%u/llm_benchmark_%j.out # log file for stdout / prints etc
#SBATCH --error=/iopsstor/scratch/cscs/%u/llm_benchmark_%j.err # log file for stderr / errors

# Exit immediately if a command exits with a non-zero status (good practice)
set -eo pipefail
# Print SLURM variables so you see how your resources are allocated
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated Node(s): $SLURM_NODELIST"
echo "Number of Tasks: $SLURM_NTASKS"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "Current path: $(pwd)"
echo "Current user: $(whoami)"

# Change to the working directory
cd /users/$(whoami)/scratch/PyRecover

# Create log directory
LOG_DIR="/users/$(whoami)/scratch/PyRecover/logs"
mkdir -p $LOG_DIR

# Set common parameters
TRAINING_STEPS=200
LOGGING_FREQ=10
BASE_CMD="python3 train.py --training-steps $TRAINING_STEPS --logging-frequency $LOGGING_FREQ" #--iterable_dset"

# Benchmarking configurations
echo "=== Starting Simple Training ==="
echo "Will run for $TRAINING_STEPS steps"

# 1. Baseline (default settings: seq_len=2048, no fused optimizer, no compile)
$BASE_CMD > $LOG_DIR/simple_train.log 2>&1
echo "Training completed"

