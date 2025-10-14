#!/bin/bash
# Optimized ATFT-GAT-FAN Training Configuration
# Based on log analysis and performance issues

echo "================================"
echo "🚀 OPTIMIZED PRODUCTION TRAINING"
echo "================================"
echo "Improvements:"
echo "  ✅ Increased hidden_size: 64 → 256"
echo "  ✅ Feature clipping enabled: 10.0"
echo "  ✅ Better learning rate: 1e-4"
echo "  ✅ Larger batch size: 1024"
echo "  ✅ Early stopping enabled"
echo "  ✅ RankIC/Sharpe optimization"
echo ""

# Core stability settings
export NUM_WORKERS=0
export ALLOW_UNSAFE_DATALOADER=0
export FORCE_SINGLE_PROCESS=1

# Data quality settings
export MIN_TRAINING_DATE="2018-01-01"
export FEATURE_CLIP_VALUE=10.0
export ENABLE_FEATURE_NORM=1

# Loss optimization (from PDF analysis)
export USE_RANKIC=1
export RANKIC_WEIGHT=0.3
export CS_IC_WEIGHT=0.2
export SHARPE_WEIGHT=0.4
export USE_HUBER=1
export HUBER_WEIGHT=0.1

# Model improvements
export HIDDEN_SIZE=256
export BATCH_SIZE=1024
export LEARNING_RATE=1e-4

# Training settings
export MAX_EPOCHS=60  # Reduced since model plateaus early
export EARLY_STOP_PATIENCE=10
export REDUCE_LR_PATIENCE=5

# Debug settings
export DEBUG_TARGETS=0
export LOG_ZERO_BATCHES=0
export DEGENERACY_ABORT=1

echo "Configuration:"
echo "  Hidden size: $HIDDEN_SIZE"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Max epochs: $MAX_EPOCHS"
echo "  Feature clipping: $FEATURE_CLIP_VALUE"
echo ""

cd /home/ubuntu/gogooku3-standalone

# Run optimized training
python scripts/train_atft.py \
    --config-path ../configs/atft \
    --config-name config_production_optimized \
    data.source.data_dir=/home/ubuntu/gogooku3-standalone/output/atft_data \
    model.hidden_size=${HIDDEN_SIZE} \
    train.trainer.max_epochs=${MAX_EPOCHS} \
    train.batch.batch_size=${BATCH_SIZE} \
    train.optimizer.lr=${LEARNING_RATE} \
    train.optimizer.weight_decay=1e-5 \
    train.scheduler.factor=0.5 \
    train.dataloader.num_workers=0 \
    train.trainer.gradient_clip_val=1.0

echo ""
echo "✅ Training started with optimized configuration!"