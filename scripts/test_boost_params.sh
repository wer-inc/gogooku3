#!/bin/bash
# Phase 1: Boost parameters test to increase pred_std above 0.010 threshold

# Boosted variance preservation parameters
export PRED_VAR_WEIGHT=1.0          # Increased from 0.5 - maximize variance
export OUTPUT_NOISE_STD=0.03        # Increased from 0.02 - more output noise
export HEAD_NOISE_STD=0.02
export PRED_VAR_MIN=0.012
export RANKIC_WEIGHT=0.3            # Increased from 0.2 - emphasize RankIC
export CS_IC_WEIGHT=0.25

# Loss configuration
export USE_RANKIC=1
export USE_CS_IC=1
export USE_TURNOVER_PENALTY=1
export TURNOVER_WEIGHT=0.0          # No turnover penalty in bootstrap phase

# Collapse prevention (common settings)
export DEGENERACY_GUARD=1
export FEATURE_CLIP_VALUE=10

# DataLoader settings (safe mode)
export ALLOW_UNSAFE_DATALOADER=1
export FORCE_SINGLE_PROCESS=1
export NUM_WORKERS=0

# System optimizations
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
ulimit -n 65535 2>/dev/null || true

echo "=========================================="
echo "Phase 1: Boost Parameters Test"
echo "=========================================="
echo "Config: config_sharpe_optimized.yaml"
echo "Epochs: 3"
echo "Batch size: 1024"
echo "Expected features: 365"
echo ""
echo "Boost Parameters:"
echo "  PRED_VAR_WEIGHT: 1.0 (from 0.5)"
echo "  OUTPUT_NOISE_STD: 0.03 (from 0.02)"
echo "  RANKIC_WEIGHT: 0.3 (from 0.2)"
echo ""
echo "Gate Criteria:"
echo "  pred_std(h1) > 0.010"
echo "=========================================="

python scripts/integrated_ml_training_pipeline.py \
    --config configs/atft/config_sharpe_optimized.yaml \
    --max-epochs 3 \
    --batch-size 1024 \
    --data-path output/ml_dataset_latest_full.parquet

exit_code=$?
echo ""
echo "=========================================="
echo "Test completed with exit code: $exit_code"
echo "=========================================="

# Extract and display pred_std from logs if available
if [ -f /tmp/boost_test.log ]; then
    echo ""
    echo "Prediction Variance Results:"
    grep -E "pred_1d.*std:" /tmp/boost_test.log | tail -3 || echo "No pred_std data found in logs"
fi

exit $exit_code
