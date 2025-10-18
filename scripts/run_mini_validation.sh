#!/bin/bash
# Mini run validation for loss weight optimization
# Expected runtime: 5-7 minutes

set -e

echo "🚀 Starting Mini Run Validation"
echo "================================"
echo "Configuration:"
echo "  - Epochs: 2"
echo "  - PHASE_MAX_BATCHES: 50"
echo "  - RANKIC_WEIGHT: 0.5"
echo "  - CS_IC_WEIGHT: 0.3"
echo "  - SHARPE_WEIGHT: 0.1"
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

# Create output directory
OUTPUT_DIR="output/loss_weight_test"
mkdir -p "$OUTPUT_DIR"

# Run mini validation
echo "📊 Running Optuna trial..."
python scripts/hpo/run_optuna_atft.py \
  --data-path output/ml_dataset_latest_full.parquet \
  --n-trials 1 \
  --max-epochs 2 \
  --study-name loss_weight_test \
  --output-dir "$OUTPUT_DIR"

echo ""
echo "✅ Mini run completed!"
echo "📁 Results saved to: $OUTPUT_DIR"
echo ""
echo "Success criteria:"
echo "  ✓ Val RankIC > 0.020 (target: 14x improvement)"
echo "  ✓ Val IC > 0.015 (target: 2x improvement)"
echo "  ✓ Epoch time ≤ 1.5 min"
echo ""
