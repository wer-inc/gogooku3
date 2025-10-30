#!/bin/bash
# Quick validation test with NaN fixes

LOG_FILE="_logs/training/validation_test_$(date +%Y%m%d_%H%M%S).log"

echo "🧪 Testing validation with NaN fixes..."

FORCE_PHASE2=1 MONITOR_GAT_GRADIENTS=1 \
python scripts/integrated_ml_training_pipeline.py \
  --max-epochs 3 \
  --data-path output/ml_dataset_latest_full.parquet \
  --batch-size 2048 \
  --lr 5e-4 \
  2>&1 | tee "$LOG_FILE"

echo
echo "📊 Extracting validation metrics..."
grep -E "val/loss|val.*rank_ic|Validation.*IC" "$LOG_FILE" | tail -10
