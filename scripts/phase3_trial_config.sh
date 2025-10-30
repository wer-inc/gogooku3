#!/bin/bash
# Phase 3 Trial Configuration
# Objective: Test degeneracy fixes with short 3-epoch run

# =============================================================================
# DEGENERACY FIXES (Phase 3)
# =============================================================================

# 1. Lower Learning Rate (5e-4 → 1e-4)
export LEARNING_RATE=1e-4

# 2. Increase RankIC Weight (0.3 → 0.5)
export USE_RANKIC=1
export RANKIC_WEIGHT=0.5

# 3. Increase Cross-Sectional IC Weight (improves prediction diversity)
export CS_IC_WEIGHT=0.25  # Increased from 0.15

# 4. Add Learning Rate Warmup (5 epochs)
export WARMUP_EPOCHS=5

# 5. Stronger Gradient Clipping
export GRAD_CLIP_NORM=0.5  # Reduced from 1.0

# 6. Add Prediction Variance Penalty (experimental)
# This will be implemented in loss function
export VARIANCE_PENALTY_WEIGHT=0.1

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

export FORCE_PHASE2=1
export MAX_EPOCHS=3  # Short trial first
export BATCH_SIZE=2048
export DATA_PATH=output/ml_dataset_latest_full.parquet

# =============================================================================
# EXPERIMENT TRACKING
# =============================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export LOG_FILE="_logs/training/phase3_trial_${TIMESTAMP}.log"

echo "=================================="
echo "Phase 3 Trial Configuration"
echo "=================================="
echo "Learning Rate: $LEARNING_RATE (reduced 5x)"
echo "RankIC Weight: $RANKIC_WEIGHT (increased from 0.3)"
echo "CS IC Weight: $CS_IC_WEIGHT (increased from 0.15)"
echo "Warmup Epochs: $WARMUP_EPOCHS"
echo "Grad Clip: $GRAD_CLIP_NORM"
echo "Variance Penalty: $VARIANCE_PENALTY_WEIGHT"
echo "Epochs: $MAX_EPOCHS (trial run)"
echo "Log File: $LOG_FILE"
echo "=================================="
