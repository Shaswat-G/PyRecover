#!/bin/bash
#SBATCH --job-name=dist_ckpt_freq500
#SBATCH --account=a-large-sc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=00:40:00
#SBATCH --partition=normal # "normal"(24h max runtime) or "debug"(30min max runtime)
#SBATCH --environment=/users/shagupta/scratch/ngc_pt_jan.toml # the environment to use
#SBATCH --output=/iopsstor/scratch/cscs/%u/dist_ckpt_freq500_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/dist_ckpt_freq500_%j.err

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

# Compute job end time in UNIX timestamp
if [ -n "$SLURM_JOB_START_TIME" ]; then
  # Check if SLURM_JOB_START_TIME is a number (UNIX timestamp)
  if [[ "$SLURM_JOB_START_TIME" =~ ^[0-9]+$ ]]; then
    start_epoch=$SLURM_JOB_START_TIME
  else
    # Assume it's a date string (format: YYYY-MM-DDTHH:MM:SS)
    start_epoch=$(date -d "$SLURM_JOB_START_TIME" +%s)
  fi
else
  # Fallback: use current time as start
  start_epoch=$(date +%s)
fi
# SLURM_TIMELIMIT is in minutes, convert to seconds
# Remove leading zeros to avoid octal interpretation
SLURM_TIMELIMIT=$((10#$SLURM_TIMELIMIT))
timelimit_sec=$((SLURM_TIMELIMIT * 60))
export SLURM_JOB_END_TIME=$((start_epoch + timelimit_sec))
echo "SLURM_JOB_END_TIME set to $SLURM_JOB_END_TIME"

# Parse command line arguments
DISTRIBUTED_FLAG=""
EXPERIMENT_NAME="default_exp"
RESUME_FLAG=""
TORCH_DIST_CKPT_FLAG=""
TIMEAWARE_CKPT_FLAG=""
LOG_LOSS_FLAG=""

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
  if [[ "$arg" == "--use_torch_distributed_ckpt" ]]; then
    TORCH_DIST_CKPT_FLAG="--use-torch-distributed-ckpt"
    echo "Using torch.distributed.checkpoint for checkpointing!"
  fi
  if [[ "$arg" == "--timeaware-checkpointing" ]]; then
    TIMEAWARE_CKPT_FLAG="--timeaware-checkpointing"
    echo "Time-aware checkpointing enabled!"
  fi
  if [[ "$arg" == "--log-loss-to-csv" ]]; then
    LOG_LOSS_FLAG="--log-loss-to-csv"
    echo "Logging loss to file!"
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
TRAINING_STEPS=3000
LOGGING_FREQ=10
CHECKPOINT_FREQ=500
GLOBAL_BATCH_SIZE=4
ITER_TIME=1
CKPT_TIME=10

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
python3 train.py --distributed --use-torch-distributed-ckpt --log-loss-to-csv --experiment_name=dist_ckpt_freq500 --checkpoint-frequency $CHECKPOINT_FREQ --training-steps $TRAINING_STEPS --logging-frequency $LOGGING_FREQ --batch-size=$GLOBAL_BATCH_SIZE
"
# 1. Baseline (default settings: seq_len=2048, no fused optimizer, no compile)
srun bash -c "$CMD"
echo "[sbatch-master] task finished"
