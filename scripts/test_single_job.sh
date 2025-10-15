#!/bin/bash
# Step B: Single job test for configuration validation

# Environment variables
export TURNOVER_WEIGHT=0.0
export PRED_VAR_WEIGHT=0.5
export OUTPUT_NOISE_STD=0.02
export RANKIC_WEIGHT=0.2

# Common settings (Collapse prevention)
export FEATURE_CLIP_VALUE=10
export DEGENERACY_GUARD=1
export HEAD_NOISE_STD=0.02
export PRED_VAR_MIN=0.012

# Loss configuration
export USE_RANKIC=1
export USE_CS_IC=1
export CS_IC_WEIGHT=0.25
export USE_TURNOVER_PENALTY=1

# DataLoader settings (safe mode)
export ALLOW_UNSAFE_DATALOADER=1
export FORCE_SINGLE_PROCESS=1
export NUM_WORKERS=0

# System optimizations
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
ulimit -n 65535 2>/dev/null || true

echo "=========================================="
echo "Step B: Single Job Test"
echo "=========================================="
echo "Config: config_sharpe_optimized.yaml"
echo "Epochs: 1"
echo "Batch size: 2048"
echo "Expected features: 365"
echo "=========================================="

python scripts/integrated_ml_training_pipeline.py \
    --config configs/atft/config_sharpe_optimized.yaml \
    --max-epochs 1 \
    --batch-size 2048 \
    --data-path output/ml_dataset_latest_full.parquet

exit_code=$?
echo ""
echo "=========================================="
echo "Test completed with exit code: $exit_code"
echo "=========================================="
exit $exit_code
