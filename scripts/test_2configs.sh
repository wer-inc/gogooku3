#!/bin/bash
# Test 2 configurations in parallel

SWEEP_DIR="output/sweep_test"
LOG_DIR="$SWEEP_DIR/logs"
mkdir -p "$LOG_DIR"

# Common settings
export FEATURE_CLIP_VALUE=10 DEGENERACY_GUARD=1 HEAD_NOISE_STD=0.02 PRED_VAR_MIN=0.012
export USE_RANKIC=1 USE_CS_IC=1 CS_IC_WEIGHT=0.25 USE_TURNOVER_PENALTY=1
export ALLOW_UNSAFE_DATALOADER=1 FORCE_SINGLE_PROCESS=0 NUM_WORKERS=4 PERSISTENT_WORKERS=1

echo "Testing 2 configurations in parallel..."

# Config 1
(
    export TURNOVER_WEIGHT=0.0 PRED_VAR_WEIGHT=0.8 OUTPUT_NOISE_STD=0.03 RANKIC_WEIGHT=0.3
    python scripts/integrated_ml_training_pipeline.py \
        --config configs/atft/config_sharpe_optimized.yaml \
        --max-epochs 2 --batch-size 2048 \
        --data-path output/ml_dataset_latest_full.parquet \
        > "$LOG_DIR/test_config1.log" 2>&1
    echo "Config 1 done (exit $?)"
) &
pid1=$!

# Config 2
(
    export TURNOVER_WEIGHT=0.05 PRED_VAR_WEIGHT=0.5 OUTPUT_NOISE_STD=0.02 RANKIC_WEIGHT=0.2
    python scripts/integrated_ml_training_pipeline.py \
        --config configs/atft/config_sharpe_optimized.yaml \
        --max-epochs 2 --batch-size 2048 \
        --data-path output/ml_dataset_latest_full.parquet \
        > "$LOG_DIR/test_config2.log" 2>&1
    echo "Config 2 done (exit $?)"
) &
pid2=$!

echo "Started: PID1=$pid1, PID2=$pid2"
echo "Waiting for completion..."

wait $pid1
wait $pid2

echo "Both configs completed!"
ls -lh $LOG_DIR/
