#!/bin/bash
# Stable training configuration for ATFT-GAT-FAN

echo "Starting stable ATFT-GAT-FAN training with optimized parameters..."

# Core settings for stability
export PYTHONPATH=/home/ubuntu/gogooku3-standalone:$PYTHONPATH

# DataLoader settings (single worker for stability)
export NUM_WORKERS=0
export PERSISTENT_WORKERS=0
export PREFETCH_FACTOR=0
export PIN_MEMORY=0
export ALLOW_UNSAFE_DATALOADER=0

# Date filtering to ensure valid targets
export MIN_TRAINING_DATE="2016-01-01"

# Batch size (conservative for stability)
export BATCH_SIZE=256
export VAL_BATCH_SIZE=512

# Training settings
export MAX_EPOCHS=120
export LEARNING_RATE=2e-4
export GRADIENT_CLIP_VAL=1.0

# Model settings
export MODEL_HIDDEN_SIZE=256
export MODEL_DROPOUT=0.1

# Loss function settings (from PDF optimization)
export USE_RANKIC=1
export RANKIC_WEIGHT=0.2
export CS_IC_WEIGHT=0.15
export SHARPE_WEIGHT=0.3

# Horizon weights
export HORIZON_WEIGHT_1D=1.0
export HORIZON_WEIGHT_5D=0.6
export HORIZON_WEIGHT_10D=0.3
export HORIZON_WEIGHT_20D=0.2

# GPU settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USE_AMP=1
export AMP_DTYPE=bf16

# Feature normalization
export ENABLE_FEATURE_NORM=1
export FEATURE_CLIP_VALUE=10.0

# Debugging
export VERBOSE=1

echo "Configuration:"
echo "  MIN_TRAINING_DATE: $MIN_TRAINING_DATE (filters out early data without valid targets)"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  NUM_WORKERS: $NUM_WORKERS (single-process for stability)"
echo "  MAX_EPOCHS: $MAX_EPOCHS"
echo "  Loss weights: RankIC=$RANKIC_WEIGHT, CS-IC=$CS_IC_WEIGHT, Sharpe=$SHARPE_WEIGHT"
echo ""

# Run training
python scripts/train_atft.py \
    --config-path ../configs/atft \
    --config-name config_production_optimized \
    data=jpx_large_scale \
    train=production_improved \
    model=atft_gat_fan \
    train.trainer.max_epochs=${MAX_EPOCHS} \
    train.batch.train_batch_size=${BATCH_SIZE} \
    train.batch.val_batch_size=${VAL_BATCH_SIZE} \
    train.optimizer.lr=${LEARNING_RATE} \
    train.trainer.gradient_clip_val=${GRADIENT_CLIP_VAL} \
    model.hidden_size=${MODEL_HIDDEN_SIZE}

echo "Training completed!"
