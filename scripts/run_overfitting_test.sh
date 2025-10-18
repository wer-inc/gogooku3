#!/bin/bash
# Overfitting prevention validation test
# Expected runtime: 10-15 minutes (5 epochs with optimized settings)

set -e

echo "🚀 Starting Overfitting Prevention Validation"
echo "=============================================="
echo "Configuration:"
echo "  - Epochs: 5"
echo "  - PHASE_MAX_BATCHES: 50 (quick test)"
echo "  - Early Stop Patience: 5"
echo "  - Early Stop Min Delta: 0.002"
echo "  - Weight Decay: 2e-5"
echo "  - Dropout Rate: 0.25"
echo "  - Gradient Clip Norm: 0.5"
echo ""

# Export environment variables
export FORCE_SINGLE_PROCESS=1
export GRAPH_REBUILD_INTERVAL=0
export USE_RANKIC=1
export RANKIC_WEIGHT=0.5
export CS_IC_WEIGHT=0.3
export SHARPE_WEIGHT=0.1
export VAL_DEBUG_LOGGING=0
export PHASE_MAX_BATCHES=50

# Overfitting prevention settings (from .env)
export EARLY_STOP_METRIC=val_rankic
export EARLY_STOP_MAXIMIZE=1
export EARLY_STOP_PATIENCE=5
export EARLY_STOP_MIN_DELTA=0.002
export WEIGHT_DECAY=2e-5
export DROPOUT_RATE=0.25
export GRADIENT_CLIP_NORM=0.5

# Create output directory
OUTPUT_DIR="output/overfitting_test"
mkdir -p "$OUTPUT_DIR"

# Run validation test
echo "📊 Running 5-epoch validation test..."
python scripts/hpo/run_optuna_atft.py \
  --data-path output/ml_dataset_latest_full.parquet \
  --n-trials 1 \
  --max-epochs 5 \
  --study-name overfitting_test \
  --output-dir "$OUTPUT_DIR"

echo ""
echo "✅ Overfitting test completed!"
echo "📁 Results saved to: $OUTPUT_DIR"
echo ""
echo "Success criteria:"
echo "  ✓ Val RankIC stable across epochs (no degradation after Epoch 2)"
echo "  ✓ Val RankIC > 0.040 maintained"
echo "  ✓ No overfitting symptoms (train/val loss divergence)"
echo ""
