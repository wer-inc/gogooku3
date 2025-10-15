#!/bin/bash
# Fixed Parallel Stability Sweep
# Uses simpler background job management

# Configuration
SWEEP_DIR="output/sweep_results"
LOG_DIR="$SWEEP_DIR/logs"
MAX_EPOCHS=5
DATA_PATH="output/ml_dataset_latest_full.parquet"
CONFIG="configs/atft/config_sharpe_optimized.yaml"
MAX_PARALLEL_JOBS=8

# Create directories
mkdir -p "$LOG_DIR"

echo "================================================================================"
echo "PARALLEL STABILITY SWEEP (FIXED)"
echo "================================================================================"
echo "Sweep directory: $SWEEP_DIR"
echo "Max parallel jobs: $MAX_PARALLEL_JOBS"
echo "================================================================================"

# Grid parameters
TURNOVER_WEIGHTS=(0.0 0.025 0.05)
PRED_VAR_WEIGHTS=(0.5 0.8 1.0)
OUTPUT_NOISE_STDS=(0.02 0.03)
RANKIC_WEIGHTS=(0.2 0.3)

# Common settings
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

# Generate all config combinations
configs=()
for tw in "${TURNOVER_WEIGHTS[@]}"; do
    for pvw in "${PRED_VAR_WEIGHTS[@]}"; do
        for onu in "${OUTPUT_NOISE_STDS[@]}"; do
            for rw in "${RANKIC_WEIGHTS[@]}"; do
                config_id="tw${tw}_pvw${pvw}_onu${onu}_rw${rw}"
                config_id="${config_id//./p}"
                configs+=("$tw,$pvw,$onu,$rw,$config_id")
            done
        done
    done
done

total_configs=${#configs[@]}
echo "Total configurations: $total_configs"
echo "Starting sweep at $(date)"
echo "================================================================================"

# Function to run single training
run_training() {
    local config_str=$1
    IFS=',' read -r tw pvw onu rw config_id <<< "$config_str"

    echo "[INFO] Starting: $config_id"

    # Set environment
    export TURNOVER_WEIGHT=$tw
    export PRED_VAR_WEIGHT=$pvw
    export OUTPUT_NOISE_STD=$onu
    export RANKIC_WEIGHT=$rw

    # Run training
    python scripts/integrated_ml_training_pipeline.py \
        --config "$CONFIG" \
        --max-epochs $MAX_EPOCHS \
        --batch-size 2048 \
        --data-path "$DATA_PATH" \
        > "$LOG_DIR/${config_id}.log" 2>&1

    local exit_code=$?

    # Save config metadata
    cat > "$LOG_DIR/${config_id}.meta" <<EOF
TURNOVER_WEIGHT=$tw
PRED_VAR_WEIGHT=$pvw
OUTPUT_NOISE_STD=$onu
RANKIC_WEIGHT=$rw
FEATURE_CLIP_VALUE=$FEATURE_CLIP_VALUE
DEGENERACY_GUARD=$DEGENERACY_GUARD
HEAD_NOISE_STD=$HEAD_NOISE_STD
PRED_VAR_MIN=$PRED_VAR_MIN
CS_IC_WEIGHT=$CS_IC_WEIGHT
EXIT_CODE=$exit_code
EOF

    if [ $exit_code -eq 0 ]; then
        echo "[DONE] ✅ $config_id"
    else
        echo "[DONE] ❌ $config_id (exit $exit_code)"
    fi

    return $exit_code
}

export -f run_training
export CONFIG MAX_EPOCHS DATA_PATH LOG_DIR
export FEATURE_CLIP_VALUE DEGENERACY_GUARD HEAD_NOISE_STD PRED_VAR_MIN
export USE_RANKIC USE_CS_IC CS_IC_WEIGHT USE_TURNOVER_PENALTY
export ALLOW_UNSAFE_DATALOADER FORCE_SINGLE_PROCESS NUM_WORKERS PERSISTENT_WORKERS

# Run in parallel using GNU parallel if available, otherwise xargs
if command -v parallel &> /dev/null; then
    echo "Using GNU Parallel for parallel execution"
    printf '%s\n' "${configs[@]}" | parallel -j $MAX_PARALLEL_JOBS run_training {}
elif command -v xargs &> /dev/null; then
    echo "Using xargs for parallel execution"
    printf '%s\n' "${configs[@]}" | xargs -P $MAX_PARALLEL_JOBS -I {} bash -c 'run_training "$@"' _ {}
else
    echo "No parallel executor found, running sequentially"
    for config in "${configs[@]}"; do
        run_training "$config"
    done
fi

echo ""
echo "================================================================================"
echo "SWEEP COMPLETED at $(date)"
echo "================================================================================"
echo "Total configurations: $total_configs"
echo "Results: $SWEEP_DIR"
echo "Logs: $LOG_DIR"
echo ""
echo "Next: python scripts/evaluate_sweep_results.py --sweep-dir $SWEEP_DIR"
echo "================================================================================"
