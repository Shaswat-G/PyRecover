#!/bin/bash
#SBATCH --job-name=llm_benchmark  # A name for your job. Visible in squeue.
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
LOG_DIR="/users/shagupta/scratch/assignment-2/benchmark_logs"
mkdir -p $LOG_DIR

# Set common parameters
TRAINING_STEPS=200
LOGGING_FREQ=10
BASE_CMD="python3 train.py --training-steps $TRAINING_STEPS --logging-frequency $LOGGING_FREQ"

# Benchmarking configurations
echo "=== Starting Benchmarking ==="
echo "Each configuration will run for $TRAINING_STEPS steps"

# 1. Baseline (default settings: seq_len=2048, no fused optimizer, no compile)
echo "Running Baseline (seq_len=2048)"
$BASE_CMD > $LOG_DIR/baseline.log 2>&1
echo "Baseline completed"

# 2. With fused optimizer
echo "Running with Fused Optimizer (seq_len=2048)"
$BASE_CMD --fused-optimizer > $LOG_DIR/fused_optimizer.log 2>&1
echo "Fused Optimizer benchmark completed"

# 3. With compilation
echo "Running with Compile (seq_len=2048)"
$BASE_CMD --compile > $LOG_DIR/compile.log 2>&1
echo "Compile benchmark completed"

# 4. With fused optimizer and compilation
echo "Running with Fused Optimizer + Compile (seq_len=2048)"
$BASE_CMD --fused-optimizer --compile > $LOG_DIR/fused_optimizer_compile.log 2>&1
echo "Fused Optimizer + Compile benchmark completed"

# 5. Double sequence length (seq_len=4096)
echo "Running with Double Sequence Length (seq_len=4096)"
$BASE_CMD --sequence-length 4096 > $LOG_DIR/seq_len_4096.log 2>&1
echo "Double Sequence Length benchmark completed"

# 6. Double sequence length with fused optimizer
echo "Running with Double Sequence Length + Fused Optimizer (seq_len=4096)"
$BASE_CMD --sequence-length 4096 --fused-optimizer > $LOG_DIR/seq_len_4096_fused_optimizer.log 2>&1
echo "Double Sequence Length + Fused Optimizer benchmark completed"

# 7. Double sequence length with compilation
echo "Running with Double Sequence Length + Compile (seq_len=4096)"
$BASE_CMD --sequence-length 4096 --compile > $LOG_DIR/seq_len_4096_compile.log 2>&1
echo "Double Sequence Length + Compile benchmark completed"

# 8. Double sequence length with fused optimizer and compilation
echo "Running with Double Sequence Length + Fused Optimizer + Compile (seq_len=4096)"
$BASE_CMD --sequence-length 4096 --fused-optimizer --compile > $LOG_DIR/seq_len_4096_fused_optimizer_compile.log 2>&1
echo "Double Sequence Length + Fused Optimizer + Compile benchmark completed"

# Report generation
echo "=== Benchmarking Complete ==="
echo "Extracting throughput metrics from logs..."

# Extract TPS, TFLOPs and MFU metrics from logs
echo -e "Configuration\tTokens per second\tTFLOPs\tMFU (%)" > $LOG_DIR/benchmark_results.tsv

for config in baseline fused_optimizer compile fused_optimizer_compile seq_len_4096 seq_len_4096_fused_optimizer seq_len_4096_compile seq_len_4096_fused_optimizer_compile; do
    # Get the last logged metrics
    last_metrics=$(grep "Tokens per second" $LOG_DIR/${config}.log | tail -1)
    tps=$(echo $last_metrics | grep -oP "Tokens per second: \K[0-9.]+")
    tflops=$(echo $last_metrics | grep -oP "TFLOPs: \K[0-9.]+")
    mfu=$(echo $last_metrics | grep -oP "MFU \(%\): \K[0-9.]+")
    
    echo -e "${config}\t${tps}\t${tflops}\t${mfu}" >> $LOG_DIR/benchmark_results.tsv
done

# Generate a more comprehensive report
echo -e "\n=== Detailed Benchmark Report ===\n" > $LOG_DIR/benchmark_report.txt
echo -e "This report compares the effects of different optimizations:" >> $LOG_DIR/benchmark_report.txt
echo -e "1. --fused-optimizer: Enables fused Adam optimizer for faster training" >> $LOG_DIR/benchmark_report.txt
echo -e "2. --compile: Uses torch.compile to optimize model execution" >> $LOG_DIR/benchmark_report.txt
echo -e "3. Sequence length: Compares default (2048) vs doubled (4096)\n" >> $LOG_DIR/benchmark_report.txt

echo -e "Performance metrics explanation:" >> $LOG_DIR/benchmark_report.txt
echo -e "- Tokens per second: Raw throughput of tokens processed" >> $LOG_DIR/benchmark_report.txt
echo -e "- TFLOPs: Hardware throughput (trillion floating point operations per second)" >> $LOG_DIR/benchmark_report.txt
echo -e "- MFU (%): Model FLOPS Utilization - how efficiently the GPU is utilized\n" >> $LOG_DIR/benchmark_report.txt
echo -e "   High MFU (40-50%): Good efficiency" >> $LOG_DIR/benchmark_report.txt
echo -e "   Low MFU (<20%): Potential bottlenecks (memory, kernels, communication)\n" >> $LOG_DIR/benchmark_report.txt

echo -e "Results:\n" >> $LOG_DIR/benchmark_report.txt
cat $LOG_DIR/benchmark_results.tsv >> $LOG_DIR/benchmark_report.txt

# Calculate percentage improvements relative to baseline
echo -e "\nPerformance Comparison (relative to baseline):" >> $LOG_DIR/benchmark_report.txt

baseline_tps=$(grep "baseline" $LOG_DIR/benchmark_results.tsv | awk '{print $2}')
if [[ ! -z "$baseline_tps" ]]; then
    for config in fused_optimizer compile fused_optimizer_compile seq_len_4096 seq_len_4096_fused_optimizer seq_len_4096_compile seq_len_4096_fused_optimizer_compile; do
        config_tps=$(grep "$config" $LOG_DIR/benchmark_results.tsv | awk '{print $2}')
        if [[ ! -z "$config_tps" ]]; then
            percent_change=$(awk "BEGIN {printf \"%.2f\", (($config_tps - $baseline_tps) / $baseline_tps) * 100}")
            echo -e "$config: $percent_change% throughput change" >> $LOG_DIR/benchmark_report.txt
        fi
    done
fi

echo "Results saved to $LOG_DIR/benchmark_results.tsv"
echo "Detailed report saved to $LOG_DIR/benchmark_report.txt"

# Show results
cat $LOG_DIR/benchmark_results.tsv
echo ""
echo "See detailed report in $LOG_DIR/benchmark_report.txt"