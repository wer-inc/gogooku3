#!/bin/bash
# Simple stable training script with minimal overrides

echo "🛡️ Starting stable single-process training..."

# Core stability settings
export NUM_WORKERS=0
export ALLOW_UNSAFE_DATALOADER=0
export FORCE_SINGLE_PROCESS=1
export MIN_TRAINING_DATE="2016-01-01"
export BATCH_SIZE=128

# Loss weights from PDF
export USE_RANKIC=1
export RANKIC_WEIGHT=0.2
export CS_IC_WEIGHT=0.15
export SHARPE_WEIGHT=0.3

# Run training with minimal Hydra overrides
cd /home/ubuntu/gogooku3-standalone
python scripts/train_atft.py \
    --config-path ../configs/atft \
    --config-name config_production_optimized \
    train.trainer.max_epochs=120 \
    model.hidden_size=256

echo "✅ Training completed!"
