#!/bin/bash
#SBATCH --job-name=llm_profile  # A name for your job. Visible in squeue.
#SBATCH --account=a-large-sc
#SBATCH --nodes=1 # Number of compute nodes to request.
#SBATCH --ntasks-per-node=1 # Tasks (processes) per node
#SBATCH --time=00:44:00 # HH:MM:SS, set a time limit for this job (here 4 hours)
#SBATCH --partition=debug # Partition to use; "debug" is usually for quick tests
#SBATCH --mem=460000 # Memory needed (simply set the mem of a node)
#SBATCH --cpus-per-task=288 # CPU cores per task (simply set the number of cpus a node has)
#SBATCH --environment=/users/shagupta/scratch/ngc_pt_jan.toml # the environment to use
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
cd /users/shagupta/scratch/assignment-2

# Create log directory
LOG_DIR="/users/shagupta/scratch/assignment-2/benchmark_logs_profiling"
mkdir -p $LOG_DIR

# Set common parameters
TRAINING_STEPS=1000
LOGGING_FREQ=10
BASE_CMD="python3 train.py --training-steps $TRAINING_STEPS --logging-frequency $LOGGING_FREQ"

echo "=== Starting Profiling with NSYS ==="
# Run with profiling enabled using NSYS
nsys profile -s none -w true \
    --trace='nvtx,cudnn,cublas,cuda' \
    --output=/iopsstor/scratch/cscs/$USER/assignment-2/nsys-trace.nsys-rep \
    --force-overwrite true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop -x true \
    numactl --membind=0-3 \
    python3 train.py --profile > $LOG_DIR/nsys_profile_trace.log 2>&1

echo "NSYS profiling completed. Trace file available at: /iopsstor/scratch/cscs/$USER/assignment-2/nsys-trace.nsys-rep"

# Report generation
echo "=== Benchmarking Complete ==="
echo "Extracting throughput metrics from logs..."

# Extract TPS, TFLOPs and MFU metrics from logs
echo -e "Configuration\tTokens per second\tTFLOPs\tMFU (%)" > $LOG_DIR/benchmark_results.tsv

for config in seq_len_4096_fused_optimizer_compile; do
    # Get the last logged metrics
    last_metrics=$(grep "Tokens per second" $LOG_DIR/${config}.log | tail -1)
    tps=$(echo $last_metrics | grep -oP "Tokens per second: \K[0-9.]+")
    tflops=$(echo $last_metrics | grep -oP "TFLOPs: \K[0-9.]+")
    mfu=$(echo $last_metrics | grep -oP "MFU \(%\): \K[0-9.]+")
    
    echo -e "${config}\t${tps}\t${tflops}\t${mfu}" >> $LOG_DIR/benchmark_results.tsv
done

echo "Results saved to $LOG_DIR/benchmark_results.tsv"
