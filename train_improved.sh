#!/bin/bash
# Improved ATFT-GAT-FAN Training with Better Hyperparameters

echo "================================"
echo "🚀 IMPROVED TRAINING CONFIGURATION"
echo "================================"
echo "Key Improvements:"
echo "  ✅ Increased hidden_size: 64 → 256 (4x capacity)"
echo "  ✅ Higher learning rate: 1e-4 → 2e-4"
echo "  ✅ Optimized batch size: 1024 → 512"
echo "  ✅ Extended training: 60 → 120 epochs"
echo "  ✅ RankIC optimization: weight=0.3"
echo "  ✅ Early stopping: patience=15"
echo ""

# Environment variables for optimization
export HIDDEN_SIZE=256
export BATCH_SIZE=512
export LEARNING_RATE=2e-4
export MAX_EPOCHS=120
export USE_RANKIC=1
export RANKIC_WEIGHT=0.3
export SHARPE_WEIGHT=0.4
export CS_IC_WEIGHT=0.15
export USE_HUBER=1
export HUBER_WEIGHT=0.1
export FEATURE_CLIP_VALUE=10.0
export MIN_TRAINING_DATE="2018-01-01"
export NUM_WORKERS=0
export OUTPUT_BASE=/home/ubuntu/gogooku3-standalone/output
export DATA_PATH=/home/ubuntu/gogooku3-standalone/output/ml_dataset_latest_full.parquet
export ALLOW_UNSAFE_DATALOADER=0
export FORCE_SINGLE_PROCESS=1
export ENABLE_FEATURE_NORM=1
export EARLY_STOP_PATIENCE=15
export REDUCE_LR_PATIENCE=7

echo "Configuration:"
echo "  Hidden size: $HIDDEN_SIZE"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Max epochs: $MAX_EPOCHS"
echo "  Feature clipping: $FEATURE_CLIP_VALUE"
echo ""

cd /home/ubuntu/gogooku3-standalone

# Run improved training with proper model capacity
python scripts/integrated_ml_training_pipeline.py \
  --data-path /home/ubuntu/gogooku3-standalone/output/ml_dataset_latest_full.parquet \
  --batch-size 512 \
  --lr 0.0002 \
  --max-epochs 120 \
  --hidden-size 256 \
  --early-stopping \
  --patience 15

echo ""
echo "✅ Improved training started!"