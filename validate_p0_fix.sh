#!/bin/bash

# P0 Fix 検証スクリプト（Phase 2成功時の設定を再現）
#
# 期待結果:
# - Val RankIC: 0.0205以上（Phase 2ベースライン）
# - P0 fixにより0.025-0.030への改善を期待
#
# 実行時間: 約6-8時間（Safe mode）

echo "=== P0 Fix Validation with Phase 2 Dataset ==="
echo ""
echo "Settings (reproducing Phase 2 success):"
echo "- Dataset: ml_dataset_phase2_enriched.parquet (3.6GB, 306 features)"
echo "- Mode: Safe (FORCE_SINGLE_PROCESS=1, num_workers=0)"
echo "- Batch size: 256 (Phase 2 baseline)"
echo "- Max epochs: 10 (Early stopping expected at 6-7)"
echo "- P0 Fix: RANKIC_WEIGHT=0.5, CS_IC_WEIGHT=0.3"
echo ""
echo "Expected improvement:"
echo "- Baseline (Phase 2): Val RankIC 0.0205"
echo "- Target (with P0 fix): Val RankIC 0.025-0.030 (+22-46%)"
echo ""

# P0 Fix環境変数
export USE_RANKIC=1
export RANKIC_WEIGHT=0.5
export CS_IC_WEIGHT=0.3
export SHARPE_WEIGHT=0.1

# Safe mode設定（Phase 2成功時と同じ）
export FORCE_SINGLE_PROCESS=1

echo "=== Environment Variables ==="
env | grep -E "USE_RANKIC|RANKIC_WEIGHT|CS_IC_WEIGHT|SHARPE_WEIGHT|FORCE_SINGLE_PROCESS"
echo ""

# データセット確認
if [ ! -f "output/ml_dataset_phase2_enriched.parquet" ]; then
    echo "ERROR: Dataset not found: output/ml_dataset_phase2_enriched.parquet"
    exit 1
fi

echo "=== Dataset Verification ==="
ls -lh output/ml_dataset_phase2_enriched.parquet
echo ""

# ログファイル準備
LOG_FILE="_logs/training/validate_p0_fix_$(date +%Y%m%d_%H%M%S).log"
mkdir -p _logs/training

echo "=== Starting Training ==="
echo "Log file: $LOG_FILE"
echo ""
echo "Command:"
echo "./venv/bin/python scripts/train_atft.py \\"
echo "  data.source.data_dir=output/ml_dataset_phase2_enriched.parquet \\"
echo "  train.trainer.max_epochs=10 \\"
echo "  train.batch.train_batch_size=256"
echo ""

# 訓練実行
./venv/bin/python scripts/train_atft.py \
  data.source.data_dir=output/ml_dataset_phase2_enriched.parquet \
  train.trainer.max_epochs=10 \
  train.batch.train_batch_size=256 \
  2>&1 | tee "$LOG_FILE"

# 結果確認
echo ""
echo "=== Training Completed ==="
echo "Extracting final metrics..."

# 最終エポックのメトリクスを抽出
grep "Val Metrics" "$LOG_FILE" | tail -5
echo ""

# ベストモデルの保存確認
grep "Saved best model" "$LOG_FILE" | tail -3
echo ""

echo "Full log: $LOG_FILE"
echo ""
echo "Next steps:"
echo "1. Compare Val RankIC with baseline (0.0205)"
echo "2. If Val RankIC >= 0.025: P0 fix is effective"
echo "3. If Val RankIC < 0.020: Investigate other bottlenecks"
