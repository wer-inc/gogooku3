#!/bin/bash
# Fix zero loss issue by enforcing minimum date and debugging

echo "================================"
echo "FIXING ZERO LOSS ISSUE"
echo "================================"

# Force minimum date to ensure valid targets
export MIN_TRAINING_DATE="2017-01-01"  # 2年分のデータが蓄積された時点から
export MIN_VALID_RATIO=0.5  # 少なくとも50%の有効なターゲットが必要

# Debugging flags
export DEBUG_TARGETS=1
export LOG_ZERO_BATCHES=1
export SKIP_ZERO_BATCHES=1

# Conservative settings
export BATCH_SIZE=64
export NUM_WORKERS=0
export ALLOW_UNSAFE_DATALOADER=0

# Feature normalization
export ENABLE_FEATURE_NORM=1
export FEATURE_CLIP_VALUE=5.0

echo "Settings:"
echo "  MIN_TRAINING_DATE: $MIN_TRAINING_DATE"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  Skipping batches with zero loss"
echo ""

# Run with minimal config for testing
cd /home/ubuntu/gogooku3-standalone
python scripts/train_atft.py \
    --config-path ../configs/atft \
    --config-name config_production \
    data.source.data_dir=/home/ubuntu/gogooku3-standalone/output/atft_data \
    model.hidden_size=64 \
    train.trainer.max_epochs=1 \
    train.batch.train_batch_size=64 \
    train.optimizer.lr=1e-4

echo "Test completed!"
