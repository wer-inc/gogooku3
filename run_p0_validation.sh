#!/bin/bash

# P0 Fix 5-Epoch Validation
# Based on successful test_fix.sh configuration

# 環境変数設定（P0 Fix）
export USE_RANKIC=1
export RANKIC_WEIGHT=0.5
export CS_IC_WEIGHT=0.3
export SHARPE_WEIGHT=0.1

echo "=== 環境変数確認 ==="
env | grep -E "USE_RANKIC|RANKIC_WEIGHT|CS_IC_WEIGHT|SHARPE_WEIGHT"

echo ""
echo "=== 訓練開始（5 epochs, P0 Fix適用） ==="
echo "Dataset: output/atft_data (99 features)"
echo "Batch size: 64 (from config)"
echo "Expected: Val RankIC improvement with P0 fix"
echo ""

./venv/bin/python scripts/train_atft.py \
  data.source.data_dir=output/atft_data \
  train.trainer.max_epochs=5
