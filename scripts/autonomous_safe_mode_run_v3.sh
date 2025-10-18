#!/bin/bash
set -e

echo "[$(date)] ========================================" | tee -a logs/autonomous_session/session_log.md
echo "[$(date)] Safe Mode v3: Minimal Test (10 batches)" | tee -a logs/autonomous_session/session_log.md
echo "[$(date)] ========================================" | tee -a logs/autonomous_session/session_log.md

# ========================================
# Safe Mode v3 設定
# ========================================

# ✅ FIX 1: グラフ構築キャッシュ有効化
export GRAPH_REBUILD_INTERVAL=0
echo "[$(date)] ✅ GRAPH_REBUILD_INTERVAL=0 (グラフキャッシュ有効)" | tee -a logs/autonomous_session/session_log.md

# ✅ FIX 2: PERSISTENT_WORKERS無効化（NUM_WORKERS=0との不整合解消）
export PERSISTENT_WORKERS=0
echo "[$(date)] ✅ PERSISTENT_WORKERS=0 (DataLoaderハング対策)" | tee -a logs/autonomous_session/session_log.md

# Safe Mode基本設定
export FORCE_SINGLE_PROCESS=1
export NUM_WORKERS=0
export ALLOW_UNSAFE_DATALOADER=0

# ✅ FIX 3: 極小テスト（10バッチのみ）
export PHASE_MAX_BATCHES=10
echo "[$(date)] ✅ PHASE_MAX_BATCHES=10 (極小テストで早期検証)" | tee -a logs/autonomous_session/session_log.md

# Loss weights最適化（Phase 2設定維持）
export USE_RANKIC=1
export RANKIC_WEIGHT=0.5
export CS_IC_WEIGHT=0.3
export SHARPE_WEIGHT=0.1

# Validation logging抑制
export VAL_DEBUG_LOGGING=0

echo "[$(date)] 環境変数設定完了" | tee -a logs/autonomous_session/session_log.md
echo "[$(date)]" | tee -a logs/autonomous_session/session_log.md

# ========================================
# 実行
# ========================================

source venv/bin/activate

echo "[$(date)] トレーニング開始..." | tee -a logs/autonomous_session/session_log.md

python scripts/hpo/run_optuna_atft.py \
  --data-path output/ml_dataset_latest_full.parquet \
  --n-trials 1 \
  --max-epochs 2 \
  --study-name safe_mode_v3_minimal \
  --output-dir output/hpo_safe_mode_v3 > logs/autonomous_session/safe_mode_v3.log 2>&1

EXIT_CODE=$?

# ========================================
# 結果判定
# ========================================

echo "[$(date)]" | tee -a logs/autonomous_session/session_log.md

if [ $EXIT_CODE -eq 0 ]; then
  echo "[$(date)] ✅✅✅ Safe Mode v3 SUCCEEDED ✅✅✅" | tee -a logs/autonomous_session/session_log.md
  echo "[$(date)] グラフキャッシュ効果を確認してください:" | tee -a logs/autonomous_session/session_log.md
  echo "[$(date)]   grep 'edges-' logs/autonomous_session/safe_mode_v3.log" | tee -a logs/autonomous_session/session_log.md
else
  echo "[$(date)] ❌ Safe Mode v3 FAILED (exit code: $EXIT_CODE)" | tee -a logs/autonomous_session/session_log.md
  echo "[$(date)] ログ確認: tail -100 logs/autonomous_session/safe_mode_v3.log" | tee -a logs/autonomous_session/session_log.md
fi

echo "[$(date)] ========================================" | tee -a logs/autonomous_session/session_log.md

exit $EXIT_CODE
