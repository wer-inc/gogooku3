#!/bin/bash
# Force single-process training to avoid DataLoader crashes

echo "==================================================="
echo "SINGLE-PROCESS STABLE TRAINING"
echo "==================================================="

# Force single-process operation
export NUM_WORKERS=0
export PERSISTENT_WORKERS=0
export PREFETCH_FACTOR=0
export PIN_MEMORY=0
export ALLOW_UNSAFE_DATALOADER=0
export FORCE_SINGLE_PROCESS=1
export USE_DAY_BATCH=0

# Date filtering for valid targets
export MIN_TRAINING_DATE="2016-01-01"

# Conservative batch sizes
export BATCH_SIZE=128
export VAL_BATCH_SIZE=256

# Disable multiprocessing completely
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Training settings
export MAX_EPOCHS=10  # Start with few epochs for testing
export LEARNING_RATE=1e-4

# Loss settings from PDF
export USE_RANKIC=1
export RANKIC_WEIGHT=0.2
export CS_IC_WEIGHT=0.15
export SHARPE_WEIGHT=0.3

# GPU settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USE_AMP=1
export AMP_DTYPE=bf16

# Feature normalization
export ENABLE_FEATURE_NORM=1
export FEATURE_CLIP_VALUE=10.0

# Debug
export PYTHONPATH=/home/ubuntu/gogooku3-standalone:$PYTHONPATH

echo "Configuration:"
echo "  NUM_WORKERS: $NUM_WORKERS (forced single-process)"
echo "  BATCH_SIZE: $BATCH_SIZE (conservative)"
echo "  MIN_TRAINING_DATE: $MIN_TRAINING_DATE"
echo "  MAX_EPOCHS: $MAX_EPOCHS (limited for testing)"
echo ""

# Run training directly without make (to avoid ALLOW_UNSAFE_DATALOADER override)
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
    model.hidden_size=256 \
    improvements.compile_model=false

echo "Training completed!"
