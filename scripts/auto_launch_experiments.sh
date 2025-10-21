#!/bin/bash
# Auto-launch experiment queue
# Waits for 1.1a to complete, then launches 1.1b

set -e

WORKSPACE="/workspace/gogooku3"
LOG_DIR="/tmp"

echo "================================================================================"
echo "Automatic Experiment Queue Manager"
echo "================================================================================"
echo "Started: $(date)"
echo ""

# Function to check if process is running
is_running() {
    local pid=$1
    ps -p $pid > /dev/null 2>&1
}

# Function to extract final Sharpe from log
extract_sharpe() {
    local log_file=$1
    grep -E "Achieved Sharpe|Final.*Sharpe" "$log_file" | tail -1 || echo "Not found"
}

# Monitor Experiment 1.1a
echo "Step 1: Monitoring Experiment 1.1a (Sharpe 0.8)"
echo "----------------------------------------"

EXP_1_1A_LOG="${LOG_DIR}/experiment_1_1a_sharpe08.log"

# Find the actual python process (not the bash wrapper)
EXP_1_1A_PID=$(pgrep -f "python.*integrated_ml_training_pipeline.*sharpe08" || echo "")

if [ -z "$EXP_1_1A_PID" ]; then
    echo "⚠️  Experiment 1.1a not found running. Checking log for completion..."
    if grep -q "Achieved Sharpe" "$EXP_1_1A_LOG" 2>/dev/null; then
        echo "✅ Experiment 1.1a already completed!"
        extract_sharpe "$EXP_1_1A_LOG"
    else
        echo "❌ Experiment 1.1a not running and not completed. Please check manually."
        exit 1
    fi
else
    echo "📊 Experiment 1.1a running (PID: $EXP_1_1A_PID)"
    echo "⏱️  Waiting for completion..."

    # Wait for completion (check every 60 seconds)
    while is_running $EXP_1_1A_PID; do
        CURRENT_EPOCH=$(tail -100 "$EXP_1_1A_LOG" 2>/dev/null | grep "Epoch [0-9]*/[0-9]*:" | tail -1 || echo "Unknown")
        echo "   $(date +%H:%M:%S) - Still running... $CURRENT_EPOCH"
        sleep 60
    done

    echo ""
    echo "✅ Experiment 1.1a COMPLETED at $(date)"
    echo "📊 Final result:"
    extract_sharpe "$EXP_1_1A_LOG"
fi

echo ""
echo "================================================================================"
echo "Step 2: Launching Experiment 1.1b (Sharpe 1.0)"
echo "================================================================================"
echo ""

# Launch Experiment 1.1b
cd "$WORKSPACE"

export SHARPE_WEIGHT=1.0
export RANKIC_WEIGHT=0.0
export CS_IC_WEIGHT=0.0

EXP_1_1B_LOG="${LOG_DIR}/experiment_1_1b_sharpe10.log"

echo "🚀 Starting Experiment 1.1b..."
echo "   Configuration:"
echo "   - Sharpe weight: 1.0 (pure Sharpe optimization)"
echo "   - RankIC weight: 0.0"
echo "   - CS-IC weight: 0.0"
echo "   - Max epochs: 30"
echo "   - Log: $EXP_1_1B_LOG"
echo ""

nohup python scripts/integrated_ml_training_pipeline.py \
    --max-epochs 30 \
    --data-path output/ml_dataset_latest_full.parquet \
    > "$EXP_1_1B_LOG" 2>&1 &

EXP_1_1B_PID=$!

echo "✅ Experiment 1.1b launched!"
echo "   PID: $EXP_1_1B_PID"
echo "   Monitor: tail -f $EXP_1_1B_LOG"
echo ""

# Wait a few seconds and verify it started
sleep 10

if is_running $EXP_1_1B_PID; then
    echo "✅ Experiment 1.1b confirmed running"
    echo ""
    echo "Latest output:"
    tail -20 "$EXP_1_1B_LOG"
else
    echo "❌ Experiment 1.1b failed to start. Check log:"
    tail -50 "$EXP_1_1B_LOG"
    exit 1
fi

echo ""
echo "================================================================================"
echo "Experiment Queue Status"
echo "================================================================================"
echo "✅ Experiment 1.1a: COMPLETED"
echo "🚀 Experiment 1.1b: RUNNING (PID: $EXP_1_1B_PID)"
echo ""
echo "To monitor 1.1b:"
echo "  tail -f $EXP_1_1B_LOG | grep -E \"Epoch|Sharpe|Phase\""
echo ""
echo "Script completed at: $(date)"
echo "================================================================================"
