#!/bin/bash
# Safe sequential fold training to avoid OOM from parallel execution
#
# Usage:
#   bash apex-ranker/scripts/train_folds_sequential.sh [start_fold] [end_fold]
#
# Example:
#   bash apex-ranker/scripts/train_folds_sequential.sh 2 5  # Run folds 2-5

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

# Parse arguments
START_FOLD="${1:-2}"
END_FOLD="${2:-5}"

echo "========================================="
echo "Sequential Fold Training (OOM-Safe)"
echo "========================================="
echo "Config: ${CONFIG_FILE}"
echo "Folds: ${START_FOLD} to ${END_FOLD}"
echo "Output: ${OUTPUT_PREFIX}_foldN.pt"
echo "Logs: ${LOG_DIR}/shortfocus_foldN.log"
echo "========================================="
echo ""

# Create log directory
mkdir -p "${LOG_DIR}"

# Track start time
OVERALL_START=$(date +%s)

# Sequential execution (memory-safe)
for fold in $(seq "${START_FOLD}" "${END_FOLD}"); do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Fold ${fold}/${CV_N_SPLITS}"
    echo "---"

    FOLD_START=$(date +%s)
    OUTPUT_FILE="${OUTPUT_PREFIX}_fold${fold}.pt"
    LOG_FILE="${LOG_DIR}/shortfocus_fold${fold}.log"

    # Show memory before training
    echo "Memory before Fold ${fold}:"
    free -h | grep Mem | awk '{print "  Used: "$3" / "$2" ("$3/$2*100"%)"}'

    # Run training (foreground to ensure completion)
    # Convert comma-separated EMA_EPOCHS to space-separated for argument list
    EMA_EPOCHS_ARGS=$(echo "${EMA_EPOCHS}" | tr ',' ' ')
    python apex-ranker/scripts/train_v0.py \
        --config "${CONFIG_FILE}" \
        --cv-type "${CV_TYPE}" \
        --cv-n-splits "${CV_N_SPLITS}" \
        --cv-fold "${fold}" \
        --embargo-days "${EMBARGO_DAYS}" \
        --output "${OUTPUT_FILE}" \
        --ema-snapshot-epochs ${EMA_EPOCHS_ARGS} \
        --max-epochs "${MAX_EPOCHS}" \
        2>&1 | tee "${LOG_FILE}"

    # Check if training succeeded
    if [ "${PIPESTATUS[0]}" -eq 0 ]; then
        FOLD_END=$(date +%s)
        FOLD_DURATION=$((FOLD_END - FOLD_START))

        echo ""
        echo "✅ Fold ${fold} completed successfully"
        echo "   Duration: $((FOLD_DURATION / 60))m $((FOLD_DURATION % 60))s"
        echo "   Output: ${OUTPUT_FILE}"
        echo "   Log: ${LOG_FILE}"

        # Show memory after training
        echo "   Memory after: $(free -h | grep Mem | awk '{print $3" / "$2}')"

        # Force garbage collection (Python cleans up between folds)
        sleep 2
    else
        echo ""
        echo "❌ Fold ${fold} FAILED!"
        echo "   Check log: ${LOG_FILE}"
        exit 1
    fi

    echo ""
    echo "---"
    echo ""
done

# Summary
OVERALL_END=$(date +%s)
OVERALL_DURATION=$((OVERALL_END - OVERALL_START))

echo "========================================="
echo "Sequential Training Complete!"
echo "========================================="
echo "Folds completed: ${START_FOLD} to ${END_FOLD}"
echo "Total duration: $((OVERALL_DURATION / 60))m $((OVERALL_DURATION % 60))s"
echo "Average per fold: $((OVERALL_DURATION / (END_FOLD - START_FOLD + 1) / 60))m"
echo ""
echo "Next steps:"
echo "1. Check logs: ${LOG_DIR}/shortfocus_fold*.log"
echo "2. Blend checkpoints: python apex-ranker/scripts/average_checkpoints.py"
echo "3. Run backtest: python apex-ranker/scripts/backtest_smoke_test.py"
echo "========================================="
