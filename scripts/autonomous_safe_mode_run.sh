#!/bin/bash
set -e

echo "[$(date)] Phase A-1: Safe Mode Dry Run Starting" | tee -a logs/autonomous_session/session_log.md

export FORCE_SINGLE_PROCESS=1
export NUM_WORKERS=0
export ALLOW_UNSAFE_DATALOADER=0

source venv/bin/activate

echo "[$(date)] Running Safe Mode (NUM_WORKERS=0)..." | tee -a logs/autonomous_session/session_log.md

python scripts/hpo/run_optuna_atft.py \
  --data-path output/ml_dataset_latest_full.parquet \
  --n-trials 1 \
  --max-epochs 2 \
  --study-name safe_mode_verify \
  --output-dir output/hpo_safe_mode > logs/autonomous_session/safe_mode_run.log 2>&1

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo "[$(date)] ✅ Safe Mode SUCCEEDED" | tee -a logs/autonomous_session/session_log.md
  cat output/hpo_safe_mode/trial_0/metrics.json | jq '.' >> logs/autonomous_session/session_log.md 2>/dev/null || echo "No metrics yet"
else
  echo "[$(date)] ❌ Safe Mode FAILED (exit code: $EXIT_CODE)" | tee -a logs/autonomous_session/session_log.md
  tail -50 logs/ml_training.log >> logs/autonomous_session/session_log.md
fi

exit $EXIT_CODE
