#!/bin/bash
# Run full 80-epoch training with best configuration from sweep
# Implements two-stage training: Stage 1 (variance bootstrap) → Stage 2 (Sharpe optimization)

set -e

SWEEP_DIR="${1:-output/sweep_results}"
TOP_CONFIG_FILE="$SWEEP_DIR/top_config_ids.txt"
OUTPUT_DIR="output/production_training"
LOG_DIR="$OUTPUT_DIR/logs"
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"

# Training parameters
STAGE1_EPOCHS=5   # Variance bootstrap
STAGE2_EPOCHS=75  # Sharpe optimization (total 80 epochs)
DATA_PATH="output/ml_dataset_latest_full.parquet"
CONFIG="configs/atft/config_sharpe_optimized.yaml"

mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"

# Check if top config file exists
if [ ! -f "$TOP_CONFIG_FILE" ]; then
    echo "❌ Error: Top config file not found: $TOP_CONFIG_FILE"
    echo "Run evaluation first:"
    echo "  python scripts/evaluate_sweep_results.py --sweep-dir $SWEEP_DIR"
    exit 1
fi

# Read best config ID
best_config=$(head -n 1 "$TOP_CONFIG_FILE")

if [ -z "$best_config" ]; then
    echo "❌ Error: No top configuration found"
    exit 1
fi

echo "================================================================================"
echo "PRODUCTION TRAINING WITH BEST CONFIGURATION"
echo "================================================================================"
echo "Best config: $best_config"
echo "Stage 1: $STAGE1_EPOCHS epochs (variance bootstrap)"
echo "Stage 2: $STAGE2_EPOCHS epochs (Sharpe optimization with SWA/ENS)"
echo "Total: $((STAGE1_EPOCHS + STAGE2_EPOCHS)) epochs"
echo "================================================================================"

# Load configuration from meta file
meta_file="$SWEEP_DIR/logs/${best_config}.meta"
if [ ! -f "$meta_file" ]; then
    echo "❌ Error: Meta file not found: $meta_file"
    exit 1
fi

echo "Loading configuration from $meta_file..."
source "$meta_file"

# Display loaded config
echo "Configuration:"
echo "  TURNOVER_WEIGHT=$TURNOVER_WEIGHT"
echo "  PRED_VAR_WEIGHT=$PRED_VAR_WEIGHT"
echo "  OUTPUT_NOISE_STD=$OUTPUT_NOISE_STD"
echo "  RANKIC_WEIGHT=$RANKIC_WEIGHT"
echo "  CS_IC_WEIGHT=$CS_IC_WEIGHT"
echo "================================================================================"

# ============================================================================
# STAGE 1: Variance Bootstrap (High variance preservation)
# ============================================================================
echo ""
echo "STAGE 1: Variance Bootstrap (${STAGE1_EPOCHS} epochs)"
echo "================================================================================"
echo "Goal: Establish stable prediction variance > 0.01"
echo "Strategy: High PRED_VAR_WEIGHT, zero TURNOVER_WEIGHT, noise injection"
echo "================================================================================"

export FEATURE_CLIP_VALUE=10
export DEGENERACY_GUARD=1
export HEAD_NOISE_STD=0.02
export PRED_VAR_MIN=0.012
export USE_RANKIC=1
export USE_CS_IC=1
export USE_TURNOVER_PENALTY=1
export ALLOW_UNSAFE_DATALOADER=1
export NUM_WORKERS=4
export PERSISTENT_WORKERS=1

# Stage 1 specific: Maximize variance preservation
export TURNOVER_WEIGHT=0.0          # No turnover penalty yet
export PRED_VAR_WEIGHT=1.0          # Max variance preservation
# Other settings from best config
export OUTPUT_NOISE_STD=$OUTPUT_NOISE_STD
export RANKIC_WEIGHT=$RANKIC_WEIGHT
export CS_IC_WEIGHT=$CS_IC_WEIGHT

stage1_log="$LOG_DIR/stage1_bootstrap.log"
stage1_checkpoint="$CHECKPOINT_DIR/stage1_final.pth"

echo "Starting Stage 1 training..."
echo "Log: $stage1_log"

python scripts/integrated_ml_training_pipeline.py \
    --config "$CONFIG" \
    --max-epochs $STAGE1_EPOCHS \
    --batch-size 2048 \
    --data-path "$DATA_PATH" \
    > "$stage1_log" 2>&1

# Check if Stage 1 completed successfully
if [ $? -ne 0 ]; then
    echo "❌ Stage 1 failed! Check log: $stage1_log"
    exit 1
fi

# Find latest checkpoint from Stage 1
latest_checkpoint=$(ls -t output/checkpoints/*.pth 2>/dev/null | head -n 1)
if [ -z "$latest_checkpoint" ]; then
    echo "⚠️  Warning: No checkpoint found from Stage 1"
    echo "Proceeding to Stage 2 from scratch..."
else
    echo "✅ Stage 1 completed. Checkpoint: $latest_checkpoint"
    cp "$latest_checkpoint" "$stage1_checkpoint"
fi

# ============================================================================
# STAGE 2: Sharpe Optimization (with SWA + Snapshot Ensemble)
# ============================================================================
echo ""
echo "STAGE 2: Sharpe Optimization (${STAGE2_EPOCHS} epochs)"
echo "================================================================================"
echo "Goal: Maximize Sharpe ratio while maintaining pred_std > 0.01"
echo "Strategy: Moderate turnover penalty, reduced variance weight, SWA/ENS"
echo "================================================================================"

# Stage 2 specific: Balance Sharpe and variance
export TURNOVER_WEIGHT=$(echo "$TURNOVER_WEIGHT * 2" | bc)  # Increase from sweep value
if [ $(echo "$TURNOVER_WEIGHT < 0.05" | bc) -eq 1 ]; then
    export TURNOVER_WEIGHT=0.05  # Minimum 0.05
fi
export PRED_VAR_WEIGHT=0.4      # Reduced from 1.0

# Enable SWA and Snapshot Ensemble
export USE_SWA=1
export SWA_START_EPOCH=60       # Start SWA after 60 epochs
export SNAPSHOT_ENS=1
export SNAPSHOT_NUM=4

stage2_log="$LOG_DIR/stage2_sharpe_optimization.log"

echo "Stage 2 configuration:"
echo "  TURNOVER_WEIGHT=$TURNOVER_WEIGHT (increased)"
echo "  PRED_VAR_WEIGHT=$PRED_VAR_WEIGHT (reduced)"
echo "  SWA enabled from epoch $SWA_START_EPOCH"
echo "  Snapshot ensemble: $SNAPSHOT_NUM snapshots"
echo ""
echo "Starting Stage 2 training..."
echo "Log: $stage2_log"

# Construct checkpoint load argument if Stage 1 checkpoint exists
checkpoint_arg=""
if [ -f "$stage1_checkpoint" ]; then
    checkpoint_arg="--checkpoint $stage1_checkpoint"
    echo "Resuming from Stage 1 checkpoint: $stage1_checkpoint"
fi

python scripts/integrated_ml_training_pipeline.py \
    --config "$CONFIG" \
    --max-epochs $((STAGE1_EPOCHS + STAGE2_EPOCHS)) \
    --batch-size 2048 \
    --data-path "$DATA_PATH" \
    $checkpoint_arg \
    > "$stage2_log" 2>&1

# Check if Stage 2 completed successfully
if [ $? -ne 0 ]; then
    echo "❌ Stage 2 failed! Check log: $stage2_log"
    exit 1
fi

echo ""
echo "================================================================================"
echo "PRODUCTION TRAINING COMPLETED"
echo "================================================================================"
echo "Stage 1 log: $stage1_log"
echo "Stage 2 log: $stage2_log"
echo "Checkpoints: $CHECKPOINT_DIR"
echo ""
echo "Next steps:"
echo "  1. Evaluate training results:"
echo "     python scripts/evaluate_trained_model.py --log-file $stage2_log"
echo ""
echo "  2. Run backtest with transaction costs:"
echo "     python scripts/backtest_sharpe_model.py \\"
echo "       --checkpoint output/checkpoints/best_model.pth \\"
echo "       --data-path $DATA_PATH \\"
echo "       --output-dir output/backtest_production"
echo ""
echo "================================================================================"
