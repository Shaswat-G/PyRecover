#!/bin/bash
#SBATCH --job-name=pyrecover_run  # A name for your job. Visible in squeue.
#SBATCH --account=a-large-sc
#SBATCH --nodes=2 # On clariden we can only get resources in full node pieces (thus its not needed to set memory or cpus)
#SBATCH --ntasks-per-node=4      # 4 tasks per node (1 per GPU) (with torchrun this would be 1)
#SBATCH --gpus-per-node=4        # with our setup, set to 1 if train non parallel and 4 if training parallel 
#SBATCH --time=00:15:00 # HH:MM:SS, set a time limit for this job
#SBATCH --partition=debug # "normal"(24h max runtime) or "debug"(30min max runtime)
#SBATCH --environment=/users/rkreft/scratch/ngc_pt_jan.toml # the environment to use
#SBATCH --output=/iopsstor/scratch/cscs/%u/llm_benchmark_%j.out # log file for stdout / prints etc
#SBATCH --error=/iopsstor/scratch/cscs/%u/llm_benchmark_%j.err # log file for stderr / errors

# Exit immediately if a command exits with a non-zero status (good practice)
set -eo pipefail
# Print SLURM variables so you see how your resources are allocated
echo "[sbatch-master] Job Name: $SLURM_JOB_NAME"
echo "[sbatch-master] Job ID: $SLURM_JOB_ID"
echo "[sbatch-master] Num Nodes: $SLURM_NNODES"
echo "[sbatch-master] Allocated Node(s): $SLURM_NODELIST"
echo "[sbatch-master] Number of Tasks(worldsize): $SLURM_NTASKS"
echo "[sbatch-master] MasterNodeID: $SLURM_NODEID"
echo "Current path: $(pwd)"
echo "Current user: $(whoami)"

# Parse command line arguments
DISTRIBUTED_FLAG=""
EXPERIMENT_NAME="default_exp"
RESUME_FLAG=""
for arg in "$@"; do
  if [ "$arg" == "--distributed" ]; then
    DISTRIBUTED_FLAG="--distributed"
    echo "LAUNCHING WITH DISTRIBUTED MODE"
  fi
  if [[ "$arg" == --exp_name=* ]]; then
    # Extract the value after the equals sign
    EXPERIMENT_NAME="${arg#*=}"
    echo "Experiment name: $EXPERIMENT_NAME"
  fi
  if [[ "$arg" == "--continue" ]]; then
    RESUME_FLAG="--resume-from-checkpoint=latest"
    echo "Resuming from latest checkpoint!"
  fi
done

# The defined environment vars will be shared with the other compute nodes.
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=12345 # Choose an unused port
export WORLD_SIZE=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
echo "[sbatch-master] execute command on compute nodes"

# Change to the working directory
cd /users/$(whoami)/scratch/PyRecover
echo "cd to: $(pwd)"

# Set common parameters
TRAINING_STEPS=600
LOGGING_FREQ=10
CHECKPOINT_FREQ=150
GLOBAL_BATCH_SIZE=8

# Benchmarking configurations
echo "=== Starting Simple Training ==="
echo "Will run for $TRAINING_STEPS steps (that's not epochs!) With global batch size $GLOBAL_BATCH_SIZE"
echo "Checkpoint every $CHECKPOINT_FREQ train steps. Log every $LOGGING_FREQ train steps"

# PREPARE-TRAIN-CMD
CMD="
# print current environment variables
echo \"[srun] rank=\$SLURM_PROCID host=\$(hostname) noderank=\$SLURM_NODEID localrank=\$SLURM_LOCALID\"
# Need to change directory again as bash -c starts from base dir
cd /users/$USER/scratch/PyRecover
# run the script
python3 train.py --training-steps $TRAINING_STEPS --logging-frequency $LOGGING_FREQ $DISTRIBUTED_FLAG --checkpoint-frequency $CHECKPOINT_FREQ --verify-checkpoints --batch-size=$GLOBAL_BATCH_SIZE --experiment_name=$EXPERIMENT_NAME $RESUME_FLAG
"

# 1. Baseline (default settings: seq_len=2048, no fused optimizer, no compile)
srun bash -c "$CMD"
echo "[sbatch-master] task finished"

