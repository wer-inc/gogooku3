#!/bin/bash

# P0 Fix検証スクリプト（修正版）

echo "=== P0 Fix Validation with Phase 2 Dataset (Corrected) ==="
echo ""
echo "Settings:"
echo "- Dataset: output/phase2_data (112 features, 9M samples)"
echo "- Mode: Safe (FORCE_SINGLE_PROCESS=1, num_workers=0)"
echo "- Batch size: 256"
echo "- Max epochs: 10 (Early stopping expected at 6-7)"
echo "- P0+P1 Fixes: RANKIC_WEIGHT=0.5, CS_IC_WEIGHT=0.3, Normalization 5→50"
echo ""
echo "Expected improvement:"
echo "- Baseline (99-feat): Val RankIC -0.054486"
echo "- Target (112-feat + fixes): Val RankIC > 0"
echo ""

# P0 Fix環境変数
export USE_RANKIC=1
export RANKIC_WEIGHT=0.5
export CS_IC_WEIGHT=0.3
export SHARPE_WEIGHT=0.1

# Safe mode設定
export FORCE_SINGLE_PROCESS=1

echo "=== Environment Variables ==="
env | grep -E "USE_RANKIC|RANKIC_WEIGHT|CS_IC_WEIGHT|SHARPE_WEIGHT|FORCE_SINGLE_PROCESS"
echo ""

# ログファイル準備
LOG_FILE="_logs/training/validate_p0_fix_112feat_$(date +%Y%m%d_%H%M%S).log"
mkdir -p _logs/training

echo "=== Starting Training ==="
echo "Log file: $LOG_FILE"
echo ""

# 訓練実行
./venv/bin/python scripts/train_atft.py \
  data.source.data_dir=output/phase2_data \
  train.trainer.max_epochs=10 \
  train.batch.train_batch_size=256 \
  2>&1 | tee "$LOG_FILE"

# 結果確認
echo ""
echo "=== Training Completed ==="
grep -E "Val Metrics" "$LOG_FILE" | tail -10
