#!/bin/bash
# Phase 2 Long Training: 50 epochs with full monitoring

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="_logs/training/phase2_long_${TIMESTAMP}.log"
PID_FILE="_logs/training/phase2_long.pid"

echo "🚀 Starting Phase 2 Long Training (50 epochs)"
echo "Log: $LOG_FILE"

# Enable all monitoring
export FORCE_PHASE2=1
export MONITOR_GAT_GRADIENTS=1
export DEGENERACY_GUARD=1

# Run training in background
nohup python scripts/integrated_ml_training_pipeline.py \
  --max-epochs 50 \
  --data-path output/ml_dataset_latest_full.parquet \
  --batch-size 2048 \
  --lr 5e-4 \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo $PID > "$PID_FILE"

echo "✅ Training started with PID: $PID"
echo "Monitor: tail -f $LOG_FILE"
echo "Check progress: grep -E 'Epoch|RankIC|GAT-GRAD' $LOG_FILE"
echo "Stop: kill $(cat $PID_FILE)"
