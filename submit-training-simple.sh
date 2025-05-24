#!/bin/bash
#SBATCH --job-name=pyrecover_run  # A name for your job. Visible in squeue.
#SBATCH --account=a-large-sc
#SBATCH --nodes=1 # On clariden we can only get resources in full node pieces (thus its not needed to set memory or cpus)
#SBATCH --ntasks-per-node=4      # 4 tasks per node (1 per GPU) (with torchrun this would be 1)
#SBATCH --gpus-per-node=4        # with our setup, set to 1 if train non parallel and 4 if training parallel 
#SBATCH --time=00:40:00 # HH:MM:SS, set a time limit for this job
#SBATCH --partition=normal # "normal"(24h max runtime) or "debug"(30min max runtime)
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

# Change to the working directory
cd /users/$(whoami)/scratch/PyRecover
echo "cd to: $(pwd)"

# Compute job end time in UNIX timestamp (FOR TIME-AWARE-CHECKPPOINTING)
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
echo "(FOR TIME-AWARE-CHECKPPOINTING) SLURM_JOB_END_TIME set to $SLURM_JOB_END_TIME"

# Parse command line arguments
DISTRIBUTED_FLAG=""
EXPERIMENT_NAME="default_exp"
RESUME_FLAG=""
TORCH_DIST_CKPT_FLAG=""
TIMEAWARE_CKPT_FLAG=""
USE_FLASH_ATTENTION_FLAG=""
LOG_LOSS_FLAG=""
# Assignment 2 flags for fused optimization sequence length and model compilation
FUSED_FLAG=""
SEQ_LEN_ARG=""
COMPILE_FLAG=""
PROFILING=""

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

  if [[ "$arg" == "--use_flash_attention" ]]; then
    USE_FLASH_ATTENTION_FLAG="--use_flash_attention"
    echo "Enable Flash-Attention! Make sure its installed for slurm job..."
    ./setup_flashattention.sh
    echo "Installed Flash-Attention!"
  fi
  if [[ "$arg" == "--log-loss-to-csv" ]]; then
    LOG_LOSS_FLAG="--log-loss-to-csv"
    echo "Logging loss to file!"
  fi
  if [[ "$arg" == "--fused-optimizer" ]]; then
    FUSED_FLAG="--fused-optimizer"
    echo "Use fused optimizer"
  fi
  if [[ "$arg" == "--compile" ]]; then
    COMPILE_FLAG="--compile"
    echo "Use compiled model!"
  fi
  if [[ "$arg" == --sequence-length=* ]]; then
    # Extract the value after the equals sign
    SEQ_LEN_ARG="${arg#*=}"
    echo "Seq-Length: $SEQ_LEN_ARG"
  fi
  if [[ "$arg" == "--profile-nsys" ]]; then
    PROFILING="--profile"
    echo "Use Nsys to trace..."
  fi
done

# The defined environment vars will be shared with the other compute nodes.
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=12345 # Choose an unused port
export WORLD_SIZE=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
echo "[sbatch-master] execute command on compute nodes"

# Set common parameters
TRAINING_STEPS=3000
LOGGING_FREQ=10
CHECKPOINT_FREQ=1000
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

python3 train.py --training-steps $TRAINING_STEPS --logging-frequency $LOGGING_FREQ $DISTRIBUTED_FLAG --checkpoint-frequency $CHECKPOINT_FREQ --verify-checkpoints --batch-size=$GLOBAL_BATCH_SIZE --experiment_name=$EXPERIMENT_NAME --default-iter-time=$ITER_TIME --default-ckpt-time=$CKPT_TIME $RESUME_FLAG $TORCH_DIST_CKPT_FLAG $TIMEAWARE_CKPT_FLAG $USE_FLASH_ATTENTION_FLAG $LOG_LOSS_FLAG $FUSED_FLAG $COMPILE_FLAG $SEQ_LEN_ARG $PROFILE
"

if [[ "$PROFILING" == "--profile" ]]; then
    echo "Running nsys..."
    nsys profile -s none -w true \
        --trace="nvtx,cudnn,cublas,cuda" \
        --output="${NSYS_OUT}/trace.nsys-rep" \
        --force-overwrite true \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop -x true \
        numactl --membind=0-3 \
        $CMD
    else
    # 1. Baseline (default settings: seq_len=2048, no fused optimizer, no compile)
    srun bash -c "$CMD"
fi

echo "[sbatch-master] task finished"

