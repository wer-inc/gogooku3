#!/bin/bash
# Production training script with proper date filtering
# Ensures we use data with valid forward-looking targets

echo "================================"
echo "PRODUCTION TRAINING WITH DATE FILTERING"
echo "================================"

# Core stability settings
export NUM_WORKERS=0
export ALLOW_UNSAFE_DATALOADER=0
export FORCE_SINGLE_PROCESS=1

# CRITICAL: Use recent data with valid forward targets
# 2018年以降のデータを使用（3年分の履歴で有効なターゲットを保証）
export MIN_TRAINING_DATE="2018-01-01"

# Conservative batch size for stability
export BATCH_SIZE=512

# Enable debugging to understand data flow
export DEBUG_TARGETS=1
export LOG_ZERO_BATCHES=1

# Loss weights from PDF analysis
export USE_RANKIC=1
export RANKIC_WEIGHT=0.2
export CS_IC_WEIGHT=0.15
export SHARPE_WEIGHT=0.3

# Model configuration
export HIDDEN_SIZE=256

echo "Configuration:"
echo "  MIN_TRAINING_DATE: $MIN_TRAINING_DATE (ensures valid forward targets)"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  NUM_WORKERS: $NUM_WORKERS (single-process for stability)"
echo "  Model hidden size: $HIDDEN_SIZE"
echo ""

# Run training with the latest dataset
cd /home/ubuntu/gogooku3-standalone

# Use the existing dataset (symlink points to the latest)
DATASET_PATH="output/ml_dataset_latest_full.parquet"

if [ ! -f "$DATASET_PATH" ]; then
    echo "❌ Dataset not found at $DATASET_PATH"
    echo "Please run: make dataset-full START=2020-09-06 END=2025-09-06"
    exit 1
fi

echo "📊 Using dataset: $DATASET_PATH"

# Get dataset info
python -c "
import pandas as pd
df = pd.read_parquet('$DATASET_PATH')
print(f'Dataset shape: {df.shape}')
print(f'Date range: {df[\"date\"].min()} to {df[\"date\"].max()}')

# Check target columns
target_cols = [col for col in df.columns if col.startswith('feat_ret_') and col.endswith('d')]
if target_cols:
    print(f'Target columns found: {target_cols}')

    # Filter by MIN_TRAINING_DATE
    df_filtered = df[df['date'] >= '$MIN_TRAINING_DATE']
    print(f'After date filter: {df_filtered.shape[0]} samples ({100*df_filtered.shape[0]/df.shape[0]:.1f}% of data)')

    # Check valid target ratio
    for col in target_cols[:2]:  # Check first 2 horizons
        valid = df_filtered[col].notna() & (df_filtered[col] != 0)
        print(f'  {col}: {valid.sum()} valid ({100*valid.sum()/len(df_filtered):.1f}%)')
else:
    print('⚠️ No feat_ret_*d target columns found!')
"

echo ""
echo "Starting training..."

# Run training with proper configuration
python scripts/train_atft.py \
    --config-path ../configs/atft \
    --config-name config_production_optimized \
    data.source.data_dir=/home/ubuntu/gogooku3-standalone/output/atft_data \
    data.source.min_date="${MIN_TRAINING_DATE}" \
    model.hidden_size=${HIDDEN_SIZE} \
    train.trainer.max_epochs=120 \
    train.batch.batch_size=${BATCH_SIZE} \
    train.optimizer.lr=2e-4 \
    train.dataloader.num_workers=0 \
    train.dataloader.persistent_workers=false \
    train.dataloader.pin_memory=false

echo "✅ Training completed!"