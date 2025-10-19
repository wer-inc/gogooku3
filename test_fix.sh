#!/bin/bash

# 環境変数設定
export USE_RANKIC=1
export RANKIC_WEIGHT=0.5
export CS_IC_WEIGHT=0.3
export SHARPE_WEIGHT=0.1

echo "=== 環境変数確認 ==="
env | grep -E "USE_RANKIC|RANKIC_WEIGHT|CS_IC_WEIGHT|SHARPE_WEIGHT"

echo ""
echo "=== 訓練開始（修正版コード + 環境変数） ==="
./venv/bin/python scripts/train_atft.py \
  data.source.data_dir=output/atft_data \
  train.trainer.max_epochs=1
