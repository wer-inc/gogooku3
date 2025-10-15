#!/bin/bash
# Test single configuration to verify training pipeline works

set -x

# Configuration
SWEEP_DIR="output/sweep_results"
LOG_DIR="$SWEEP_DIR/logs"
mkdir -p "$LOG_DIR"

# Test config
export FEATURE_CLIP_VALUE=10
export DEGENERACY_GUARD=1
export HEAD_NOISE_STD=0.02
export PRED_VAR_MIN=0.012
export USE_RANKIC=1
export USE_CS_IC=1
export CS_IC_WEIGHT=0.25
export USE_TURNOVER_PENALTY=1
export ALLOW_UNSAFE_DATALOADER=1
export FORCE_SINGLE_PROCESS=0
export NUM_WORKERS=4
export PERSISTENT_WORKERS=1

export TURNOVER_WEIGHT=0.0
export PRED_VAR_WEIGHT=0.8
export OUTPUT_NOISE_STD=0.03
export RANKIC_WEIGHT=0.3

config_id="test_tw0p0_pvw0p8"

echo "Starting test training: $config_id"
echo "Log: $LOG_DIR/${config_id}.log"

python scripts/integrated_ml_training_pipeline.py \
    --config configs/atft/config_sharpe_optimized.yaml \
    --max-epochs 5 \
    --batch-size 2048 \
    --data-path output/ml_dataset_latest_full.parquet \
    > "$LOG_DIR/${config_id}.log" 2>&1

exit_code=$?
echo "Training exit code: $exit_code"

if [ $exit_code -eq 0 ]; then
    echo "✅ Test training completed successfully"
    tail -50 "$LOG_DIR/${config_id}.log"
else
    echo "❌ Test training failed"
    tail -100 "$LOG_DIR/${config_id}.log"
fi
