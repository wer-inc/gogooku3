#!/bin/bash
set -e

echo "[$(date)] Phase A-1 (FIXED): Safe Mode with Graph Caching" | tee -a logs/autonomous_session/session_log.md

# ✅ FIX: Enable graph caching to avoid 78-hour epoch time
export GRAPH_REBUILD_INTERVAL=0  # Build once per epoch, cache for all batches

# Safe Mode settings
export FORCE_SINGLE_PROCESS=1
export NUM_WORKERS=0
export ALLOW_UNSAFE_DATALOADER=0

# Other optimizations still apply
export USE_RANKIC=1
export RANKIC_WEIGHT=0.5
export CS_IC_WEIGHT=0.3
export SHARPE_WEIGHT=0.1

source venv/bin/activate

echo "[$(date)] 🔧 GRAPH_REBUILD_INTERVAL=0 (cache graph for entire epoch)" | tee -a logs/autonomous_session/session_log.md

python scripts/hpo/run_optuna_atft.py \
  --data-path output/ml_dataset_latest_full.parquet \
  --n-trials 1 \
  --max-epochs 2 \
  --study-name safe_mode_graph_cached \
  --output-dir output/hpo_safe_mode_fixed > logs/autonomous_session/safe_mode_fixed.log 2>&1

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo "[$(date)] ✅ Safe Mode SUCCEEDED (graph caching enabled)" | tee -a logs/autonomous_session/session_log.md
else
  echo "[$(date)] ❌ Safe Mode FAILED (exit code: $EXIT_CODE)" | tee -a logs/autonomous_session/session_log.md
fi

exit $EXIT_CODE
