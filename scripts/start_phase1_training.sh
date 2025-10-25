#!/bin/bash
# Phase 1 Training Launcher
# Purpose: Start 60-epoch training in Safe Mode with Option B.1 dataset

set -e

LOGFILE="_logs/training/phase1_full_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================"
echo "Phase 1 Training: 60 Epochs (Safe Mode)"
echo "============================================================"
echo "Started: $(date)"
echo "Dataset: output/ml_dataset_latest_full.parquet (Option B.1)"
echo "Features: 381 (with ret_prev_* lagged features)"
echo "Log file: $LOGFILE"
echo ""
echo "Expected duration: ~20 hours (60 epochs × 20 min/epoch)"
echo ""

# Run training
make train-safe EPOCHS=60 2>&1 | tee "$LOGFILE"

echo ""
echo "============================================================"
echo "Phase 1 Training Complete"
echo "============================================================"
echo "Finished: $(date)"
echo "Log file: $LOGFILE"
