#!/bin/bash
# Simple and robust parallel sweep - minimal complexity

SWEEP_DIR="output/sweep_results"
LOG_DIR="$SWEEP_DIR/logs"
MAX_EPOCHS=5
MAX_PARALLEL=8
TIMEOUT="2h"

mkdir -p "$LOG_DIR"

echo "==========================================================================="
echo "PARALLEL SWEEP (SIMPLE & ROBUST)"
echo "==========================================================================="
echo "Directory: $SWEEP_DIR"
echo "Parallel jobs: $MAX_PARALLEL"
echo "Started: $(date)"
echo "==========================================================================="

# System optimizations
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
ulimit -n 65535 2>/dev/null || true

# Common settings (Variance Collapse prevention)
export FEATURE_CLIP_VALUE=10 DEGENERACY_GUARD=1 HEAD_NOISE_STD=0.02 PRED_VAR_MIN=0.012
export USE_RANKIC=1 USE_CS_IC=1 CS_IC_WEIGHT=0.25 USE_TURNOVER_PENALTY=1
export ALLOW_UNSAFE_DATALOADER=1 FORCE_SINGLE_PROCESS=1 NUM_WORKERS=0

# Grid
TW_VALS=(0.0 0.025 0.05)
PV_VALS=(0.5 0.8 1.0)
ON_VALS=(0.02 0.03)
RW_VALS=(0.2 0.3)

# Generate configs
configs=()
for tw in "${TW_VALS[@]}"; do for pv in "${PV_VALS[@]}"; do for on in "${ON_VALS[@]}"; do for rw in "${RW_VALS[@]}"; do
    id="tw${tw//./-}_pv${pv//./-}_on${on//./-}_rw${rw//./-}"
    configs+=("$tw:$pv:$on:$rw:$id")
done; done; done; done

echo "Total configs: ${#configs[@]}"
echo ""

# Process in batches
idx=0
for config in "${configs[@]}"; do
    IFS=':' read -r tw pv on rw id <<< "${configs[$idx]}"

    # Launch job
    (
        export CUDA_VISIBLE_DEVICES=$((idx % 1))  # Single GPU
        export TURNOVER_WEIGHT=$tw PRED_VAR_WEIGHT=$pv OUTPUT_NOISE_STD=$on RANKIC_WEIGHT=$rw

        timeout $TIMEOUT python scripts/integrated_ml_training_pipeline.py \
            --config configs/atft/config_sharpe_optimized.yaml \
            --max-epochs $MAX_EPOCHS --batch-size 2048 \
            --data-path output/ml_dataset_latest_full.parquet \
            > "$LOG_DIR/${id}.log" 2>&1

        ec=$?
        echo "$tw,$pv,$on,$rw,$id,$ec,$(date +%s)" >> "$LOG_DIR/results.csv"

        [ $ec -eq 0 ] && echo "✅ $id" || echo "❌ $id (exit $ec)"
    ) &

    ((idx++))

    # Wait when batch full
    if [ $((idx % MAX_PARALLEL)) -eq 0 ]; then
        echo "Waiting for batch..."
        wait
    fi
done

# Wait for remaining
wait

echo ""
echo "==========================================================================="
echo "COMPLETED: $(date)"
echo "==========================================================================="
ls -lh "$LOG_DIR"/*.log | wc -l | xargs echo "Log files:"
grep -c "✅" "$LOG_DIR/results.csv" 2>/dev/null | xargs echo "Successful:"
echo ""
echo "Next: python scripts/evaluate_sweep_results.py --sweep-dir $SWEEP_DIR"
