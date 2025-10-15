#!/bin/bash
# Parallel Stability Sweep for Variance Collapse Avoidance
# Runs 5-epoch tests with multiple hyperparameter combinations in parallel

set -e

# Configuration
SWEEP_DIR="output/sweep_results"
LOG_DIR="$SWEEP_DIR/logs"
CHECKPOINT_DIR="$SWEEP_DIR/checkpoints"
MAX_EPOCHS=5
DATA_PATH="output/ml_dataset_latest_full.parquet"
CONFIG="configs/atft/config_sharpe_optimized.yaml"

# Parallel control
MAX_PARALLEL_JOBS=8  # Adjust based on GPU memory

# Create directories
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"

echo "================================================================================"
echo "PARALLEL STABILITY SWEEP"
echo "================================================================================"
echo "Sweep directory: $SWEEP_DIR"
echo "Max parallel jobs: $MAX_PARALLEL_JOBS"
echo "Epochs per run: $MAX_EPOCHS"
echo "================================================================================"

# Grid parameters
TURNOVER_WEIGHTS=(0.0 0.025 0.05)
PRED_VAR_WEIGHTS=(0.5 0.8 1.0)
OUTPUT_NOISE_STDS=(0.02 0.03)
RANKIC_WEIGHTS=(0.2 0.3)

# Common settings (Collapse prevention)
export FEATURE_CLIP_VALUE=10
export DEGENERACY_GUARD=1
export HEAD_NOISE_STD=0.02
export PRED_VAR_MIN=0.012
export USE_RANKIC=1
export USE_CS_IC=1
export CS_IC_WEIGHT=0.25
export USE_TURNOVER_PENALTY=1
export ALLOW_UNSAFE_DATALOADER=1
export FORCE_SINGLE_PROCESS=0  # Enable multi-worker
export NUM_WORKERS=4
export PERSISTENT_WORKERS=1

# Job counter
job_count=0
total_jobs=0

# Calculate total jobs
for tw in "${TURNOVER_WEIGHTS[@]}"; do
    for pvw in "${PRED_VAR_WEIGHTS[@]}"; do
        for onu in "${OUTPUT_NOISE_STDS[@]}"; do
            for rw in "${RANKIC_WEIGHTS[@]}"; do
                ((total_jobs++))
            done
        done
    done
done

echo "Total configurations to test: $total_jobs"
echo "Starting sweep at $(date)"
echo "================================================================================"

# Track PIDs and config IDs
declare -a pids=()
declare -a config_ids=()

# Launch jobs
for tw in "${TURNOVER_WEIGHTS[@]}"; do
    for pvw in "${PRED_VAR_WEIGHTS[@]}"; do
        for onu in "${OUTPUT_NOISE_STDS[@]}"; do
            for rw in "${RANKIC_WEIGHTS[@]}"; do
                # Generate config ID
                config_id="tw${tw}_pvw${pvw}_onu${onu}_rw${rw}"
                config_id="${config_id//./p}"  # Replace . with p

                ((job_count++))

                echo "[$job_count/$total_jobs] Launching: $config_id"

                # Set config-specific environment
                export TURNOVER_WEIGHT=$tw
                export PRED_VAR_WEIGHT=$pvw
                export OUTPUT_NOISE_STD=$onu
                export RANKIC_WEIGHT=$rw

                # Launch training in background
                (
                    python scripts/integrated_ml_training_pipeline.py \
                        --config "$CONFIG" \
                        --max-epochs $MAX_EPOCHS \
                        --batch-size 2048 \
                        --data-path "$DATA_PATH" \
                        > "$LOG_DIR/${config_id}.log" 2>&1

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
EOF
                ) &

                pid=$!
                pids+=($pid)
                config_ids+=($config_id)

                # Wait if we hit max parallel jobs
                if [ ${#pids[@]} -ge $MAX_PARALLEL_JOBS ]; then
                    echo "Waiting for batch to complete (${#pids[@]} jobs running)..."
                    for i in "${!pids[@]}"; do
                        wait ${pids[$i]}
                        exit_code=$?
                        if [ $exit_code -eq 0 ]; then
                            echo "✅ ${config_ids[$i]} completed successfully"
                        else
                            echo "❌ ${config_ids[$i]} failed (exit code: $exit_code)"
                        fi
                    done
                    pids=()
                    config_ids=()
                fi
            done
        done
    done
done

# Wait for remaining jobs
if [ ${#pids[@]} -gt 0 ]; then
    echo "Waiting for final batch (${#pids[@]} jobs)..."
    for i in "${!pids[@]}"; do
        wait ${pids[$i]}
        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "✅ ${config_ids[$i]} completed successfully"
        else
            echo "❌ ${config_ids[$i]} failed (exit code: $exit_code)"
        fi
    done
fi

echo "================================================================================"
echo "SWEEP COMPLETED at $(date)"
echo "================================================================================"
echo "Results directory: $SWEEP_DIR"
echo "Next step: Run evaluation script to select top configurations"
echo ""
echo "  python scripts/evaluate_sweep_results.py --sweep-dir $SWEEP_DIR"
echo ""
echo "================================================================================"
