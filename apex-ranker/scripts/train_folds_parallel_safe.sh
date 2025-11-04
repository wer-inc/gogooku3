#!/bin/bash
# Memory-safe parallel fold training (2 parallel max)
#
# This script runs folds in groups of 2 to balance speed and memory safety.
# Ideal for systems with 2TiB RAM but want to avoid OOM from 5x parallel execution.
#
# Usage:
#   bash apex-ranker/scripts/train_folds_parallel_safe.sh [start_fold] [end_fold]
#
# Example:
#   bash apex-ranker/scripts/train_folds_parallel_safe.sh 2 5

set -euo pipefail

# Configuration
CONFIG_FILE="${CONFIG_FILE:-apex-ranker/configs/v0_base.yaml}"
CV_TYPE="${CV_TYPE:-purged_kfold}"
CV_N_SPLITS="${CV_N_SPLITS:-5}"
EMBARGO_DAYS="${EMBARGO_DAYS:-5}"
MAX_EPOCHS="${MAX_EPOCHS:-12}"
EMA_EPOCHS="${EMA_EPOCHS:-3,6,10}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-models/shortfocus}"
LOG_DIR="${LOG_DIR:-logs}"
PARALLEL_JOBS="${PARALLEL_JOBS:-2}"  # Max 2 concurrent jobs

# Parse arguments
START_FOLD="${1:-2}"
END_FOLD="${2:-5}"

echo "========================================="
echo "Parallel-Safe Fold Training"
echo "========================================="
echo "Config: ${CONFIG_FILE}"
echo "Folds: ${START_FOLD} to ${END_FOLD}"
echo "Parallel jobs: ${PARALLEL_JOBS} at a time"
echo "Output: ${OUTPUT_PREFIX}_foldN.pt"
echo "Logs: ${LOG_DIR}/shortfocus_foldN.log"
echo "========================================="
echo ""

# Create log directory
mkdir -p "${LOG_DIR}"

# Track start time
OVERALL_START=$(date +%s)

# Function to train a single fold
train_fold() {
    local fold=$1
    local output_file="${OUTPUT_PREFIX}_fold${fold}.pt"
    local log_file="${LOG_DIR}/shortfocus_fold${fold}.log"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Fold ${fold} (PID: $$)" | tee -a "${log_file}"

    # Convert comma-separated EMA_EPOCHS to space-separated for argument list
    EMA_EPOCHS_ARGS=$(echo "${EMA_EPOCHS}" | tr ',' ' ')
    python apex-ranker/scripts/train_v0.py \
        --config "${CONFIG_FILE}" \
        --cv-type "${CV_TYPE}" \
        --cv-n-splits "${CV_N_SPLITS}" \
        --cv-fold "${fold}" \
        --embargo-days "${EMBARGO_DAYS}" \
        --output "${output_file}" \
        --ema-snapshot-epochs ${EMA_EPOCHS_ARGS} \
        --max-epochs "${MAX_EPOCHS}" \
        >> "${log_file}" 2>&1

    local exit_code=$?

    if [ "${exit_code}" -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Fold ${fold} completed successfully" | tee -a "${log_file}"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ Fold ${fold} FAILED (exit code: ${exit_code})" | tee -a "${log_file}"
    fi

    return "${exit_code}"
}

# Export function for parallel execution
export -f train_fold
export CONFIG_FILE CV_TYPE CV_N_SPLITS EMBARGO_DAYS MAX_EPOCHS EMA_EPOCHS OUTPUT_PREFIX LOG_DIR

# Show initial memory
echo "Memory before training:"
free -h | grep Mem
echo ""

# Run folds in batches of PARALLEL_JOBS
folds_array=($(seq "${START_FOLD}" "${END_FOLD}"))
total_folds=${#folds_array[@]}
batch_count=$(( (total_folds + PARALLEL_JOBS - 1) / PARALLEL_JOBS ))

echo "Running ${total_folds} folds in ${batch_count} batches of ${PARALLEL_JOBS}"
echo ""

failed_folds=()

for ((batch=0; batch<batch_count; batch++)); do
    batch_start=$((batch * PARALLEL_JOBS))
    batch_end=$((batch_start + PARALLEL_JOBS))
    [ "${batch_end}" -gt "${total_folds}" ] && batch_end="${total_folds}"

    batch_folds=("${folds_array[@]:batch_start:$((batch_end - batch_start))}")

    echo "========================================="
    echo "Batch $((batch + 1))/${batch_count}: Folds ${batch_folds[*]}"
    echo "========================================="

    # Launch parallel jobs
    pids=()
    for fold in "${batch_folds[@]}"; do
        train_fold "${fold}" &
        pids+=($!)
        echo "  Launched Fold ${fold} (PID: ${pids[-1]})"
    done

    echo ""
    echo "Waiting for batch $((batch + 1)) to complete..."

    # Wait for all jobs in this batch
    for i in "${!pids[@]}"; do
        fold="${batch_folds[$i]}"
        pid="${pids[$i]}"

        if wait "${pid}"; then
            echo "  ✅ Fold ${fold} finished (PID: ${pid})"
        else
            echo "  ❌ Fold ${fold} FAILED (PID: ${pid})"
            failed_folds+=("${fold}")
        fi
    done

    echo ""
    echo "Batch $((batch + 1)) complete. Memory status:"
    free -h | grep Mem
    echo ""

    # Small delay between batches to allow memory cleanup
    if [ $((batch + 1)) -lt "${batch_count}" ]; then
        echo "Waiting 5 seconds before next batch..."
        sleep 5
        echo ""
    fi
done

# Summary
OVERALL_END=$(date +%s)
OVERALL_DURATION=$((OVERALL_END - OVERALL_START))

echo "========================================="
echo "Parallel Training Complete!"
echo "========================================="
echo "Folds requested: ${START_FOLD} to ${END_FOLD}"
echo "Total duration: $((OVERALL_DURATION / 60))m $((OVERALL_DURATION % 60))s"

if [ "${#failed_folds[@]}" -eq 0 ]; then
    echo "Status: ✅ All folds completed successfully"
    echo ""
    echo "Next steps:"
    echo "1. Check logs: ${LOG_DIR}/shortfocus_fold*.log"
    echo "2. Blend checkpoints: python apex-ranker/scripts/average_checkpoints.py"
    echo "3. Run backtest: python apex-ranker/scripts/backtest_smoke_test.py"
else
    echo "Status: ❌ ${#failed_folds[@]} fold(s) failed: ${failed_folds[*]}"
    echo ""
    echo "Check failed logs:"
    for fold in "${failed_folds[@]}"; do
        echo "  - ${LOG_DIR}/shortfocus_fold${fold}.log"
    done
    exit 1
fi

echo "========================================="
