#!/bin/bash
# Production-grade Parallel Stability Sweep
# Incorporates all robustness improvements

# Strict error handling
set -Eeuo pipefail
IFS=$'\n\t'

# Trap handlers
cleanup() {
    echo "[CLEANUP] Terminating all child processes..."
    kill 0 2>/dev/null || true
}
trap cleanup INT TERM ERR
trap 'wait' EXIT

# Configuration
SWEEP_DIR="output/sweep_results"
LOG_DIR="$SWEEP_DIR/logs"
MAX_EPOCHS=5
DATA_PATH="output/ml_dataset_latest_full.parquet"
CONFIG="configs/atft/config_sharpe_optimized.yaml"
MAX_PARALLEL_JOBS=8
TIMEOUT_PER_JOB="2h"  # Prevent hangs
RUN_ID=$(date +%s%N)  # Unique run identifier

# System limits
ulimit -n 65535 2>/dev/null || echo "[WARN] Could not set file limit"

# Thread control (prevent CPU over-subscription)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Create directories
mkdir -p "$LOG_DIR"

echo "================================================================================"
echo "PRODUCTION PARALLEL STABILITY SWEEP"
echo "================================================================================"
echo "Run ID: $RUN_ID"
echo "Sweep directory: $SWEEP_DIR"
echo "Max parallel jobs: $MAX_PARALLEL_JOBS"
echo "Timeout per job: $TIMEOUT_PER_JOB"
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
export FORCE_SINGLE_PROCESS=1
export NUM_WORKERS=0
export PERSISTENT_WORKERS=1

# GPU assignment function (round-robin if multiple GPUs)
get_gpu_id() {
    local job_idx=$1
    local num_gpus=$(nvidia-smi --list-gpus | wc -l)
    echo $((job_idx % num_gpus))
}

# Generate all configurations
declare -a configs=()
job_count=0
for tw in "${TURNOVER_WEIGHTS[@]}"; do
    for pvw in "${PRED_VAR_WEIGHTS[@]}"; do
        for onu in "${OUTPUT_NOISE_STDS[@]}"; do
            for rw in "${RANKIC_WEIGHTS[@]}"; do
                config_id="tw${tw}_pvw${pvw}_onu${onu}_rw${rw}"
                config_id="${config_id//./p}"
                configs+=("$tw|$pvw|$onu|$rw|$config_id|$job_count")
                ((job_count++))
            done
        done
    done
done

total_configs=${#configs[@]}
echo "Total configurations: $total_configs"
echo "Starting sweep at $(date)"
echo "================================================================================"

# Status tracking
declare -A job_status=()
successful=0
failed=0
running=0

# Launch jobs in batches
batch_start=0
while [ $batch_start -lt $total_configs ]; do
    batch_end=$((batch_start + MAX_PARALLEL_JOBS))
    [ $batch_end -gt $total_configs ] && batch_end=$total_configs

    echo ""
    echo "[BATCH] Launching jobs $((batch_start+1))-$batch_end of $total_configs"

    declare -a pids=()
    declare -a config_ids=()

    for ((i=batch_start; i<batch_end; i++)); do
        IFS='|' read -r tw pvw onu rw config_id job_idx <<< "${configs[$i]}"

        # GPU assignment
        gpu_id=$(get_gpu_id $job_idx)

        # Set config-specific environment
        (
            export CUDA_VISIBLE_DEVICES=$gpu_id
            export TURNOVER_WEIGHT=$tw
            export PRED_VAR_WEIGHT=$pvw
            export OUTPUT_NOISE_STD=$onu
            export RANKIC_WEIGHT=$rw

            log_file="$LOG_DIR/${config_id}.log"
            meta_file="$LOG_DIR/${config_id}.meta"

            echo "[START] GPU$gpu_id: $config_id" >> "$LOG_DIR/sweep_progress.log"

            # Run with timeout
            timeout $TIMEOUT_PER_JOB python scripts/integrated_ml_training_pipeline.py \
                --config "$CONFIG" \
                --max-epochs $MAX_EPOCHS \
                --batch-size 2048 \
                --data-path "$DATA_PATH" \
                > "$log_file" 2>&1

            exit_code=$?

            # Save metadata
            cat > "$meta_file" <<EOF
TURNOVER_WEIGHT=$tw
PRED_VAR_WEIGHT=$pvw
OUTPUT_NOISE_STD=$onu
RANKIC_WEIGHT=$rw
FEATURE_CLIP_VALUE=$FEATURE_CLIP_VALUE
DEGENERACY_GUARD=$DEGENERACY_GUARD
HEAD_NOISE_STD=$HEAD_NOISE_STD
PRED_VAR_MIN=$PRED_VAR_MIN
CS_IC_WEIGHT=$CS_IC_WEIGHT
GPU_ID=$gpu_id
RUN_ID=$RUN_ID
EXIT_CODE=$exit_code
START_TIME=$(date -r "$log_file" +%s 2>/dev/null || echo "unknown")
END_TIME=$(date +%s)
EOF

            if [ $exit_code -eq 0 ]; then
                echo "[DONE] ✅ GPU$gpu_id: $config_id" >> "$LOG_DIR/sweep_progress.log"
            elif [ $exit_code -eq 124 ]; then
                echo "[TIMEOUT] ⏱️ GPU$gpu_id: $config_id" >> "$LOG_DIR/sweep_progress.log"
            else
                echo "[FAIL] ❌ GPU$gpu_id: $config_id (exit $exit_code)" >> "$LOG_DIR/sweep_progress.log"
            fi

            exit $exit_code
        ) &

        pid=$!
        pids+=($pid)
        config_ids+=($config_id)

        echo "  [$((i+1))/$total_configs] Launched: $config_id (PID=$pid, GPU=$gpu_id)"
    done

    # Wait for batch to complete
    echo ""
    echo "[WAIT] Waiting for batch to complete..."
    for idx in "${!pids[@]}"; do
        pid=${pids[$idx]}
        config_id=${config_ids[$idx]}

        if wait $pid; then
            echo "  ✅ $config_id completed successfully"
            ((successful++))
            job_status[$config_id]="success"
        else
            exit_code=$?
            if [ $exit_code -eq 124 ]; then
                echo "  ⏱️ $config_id timed out"
            else
                echo "  ❌ $config_id failed (exit $exit_code)"
            fi
            ((failed++))
            job_status[$config_id]="failed"
        fi
    done

    echo ""
    echo "[PROGRESS] Completed: $((successful+failed))/$total_configs (✅$successful ❌$failed)"

    batch_start=$batch_end
done

# Final summary
echo ""
echo "================================================================================"
echo "SWEEP COMPLETED at $(date)"
echo "================================================================================"
echo "Total configurations: $total_configs"
echo "Successful: $successful"
echo "Failed: $failed"
echo "Success rate: $(awk "BEGIN {printf \"%.1f\", 100*$successful/$total_configs}")%"
echo ""
echo "Results: $SWEEP_DIR"
echo "Logs: $LOG_DIR"
echo "Progress log: $LOG_DIR/sweep_progress.log"
echo ""
echo "Next step: python scripts/evaluate_sweep_results.py --sweep-dir $SWEEP_DIR"
echo "================================================================================"

# Generate summary report
cat > "$SWEEP_DIR/sweep_summary.txt" <<EOF
Sweep Summary
=============
Run ID: $RUN_ID
Start: $(date -d @$((RUN_ID/1000000000)) 2>/dev/null || echo "N/A")
End: $(date)
Total configs: $total_configs
Successful: $successful
Failed: $failed
Success rate: $(awk "BEGIN {printf \"%.1f\", 100*$successful/$total_configs}")%

Configuration Grid:
- TURNOVER_WEIGHT: ${TURNOVER_WEIGHTS[@]}
- PRED_VAR_WEIGHT: ${PRED_VAR_WEIGHTS[@]}
- OUTPUT_NOISE_STD: ${OUTPUT_NOISE_STDS[@]}
- RANKIC_WEIGHT: ${RANKIC_WEIGHTS[@]}

Settings:
- Max parallel jobs: $MAX_PARALLEL_JOBS
- Timeout per job: $TIMEOUT_PER_JOB
- Max epochs: $MAX_EPOCHS
- Batch size: 2048
EOF

echo ""
echo "📄 Summary report: $SWEEP_DIR/sweep_summary.txt"
echo ""

# Exit with appropriate code
if [ $failed -eq 0 ]; then
    echo "🎉 All jobs completed successfully!"
    exit 0
elif [ $successful -gt 0 ]; then
    echo "⚠️ Some jobs failed, but $successful completed successfully"
    exit 0
else
    echo "❌ All jobs failed!"
    exit 1
fi
