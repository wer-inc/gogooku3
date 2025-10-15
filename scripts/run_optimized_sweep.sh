#!/bin/bash
# Optimized execution script for parallel sweep
# Automatically stops existing processes, optimizes parallelism, and runs full pipeline

set -e

echo "================================================================================"
echo "OPTIMIZED PARALLEL SWEEP + AUTO-TRAINING PIPELINE"
echo "================================================================================"
echo "This script will:"
echo "  1. Stop existing training processes (free GPU)"
echo "  2. Optimize parallelism (MAX_PARALLEL_JOBS=8)"
echo "  3. Run full automated pipeline (6-8 hours)"
echo ""
echo "System resources:"
echo "  CPU: 256 cores"
echo "  GPU: A100 80GB"
echo "  Memory: 430GB free"
echo "  Disk: 157GB free"
echo "================================================================================"

read -p "Continue? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Step 1: Stop existing processes
echo ""
echo "Step 1: Stopping existing training processes..."
pkill -f "train_atft|integrated_ml_training" && echo "✅ Stopped" || echo "⚠️  No processes to stop"
sleep 2

# Step 2: Check GPU is free
echo ""
echo "Step 2: Checking GPU status..."
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{if($1>10000) print "⚠️ GPU still in use: "$1"MB"; else print "✅ GPU available: "$1"MB used"}'

# Step 3: Optimize parallelism
echo ""
echo "Step 3: Optimizing parallelism..."
if grep -q "MAX_PARALLEL_JOBS=4" scripts/parallel_sweep.sh; then
    sed -i 's/MAX_PARALLEL_JOBS=4/MAX_PARALLEL_JOBS=8/' scripts/parallel_sweep.sh
    echo "✅ Changed MAX_PARALLEL_JOBS: 4 → 8"
else
    echo "⚠️  Already optimized or value differs"
fi

# Step 4: Run full pipeline
echo ""
echo "Step 4: Starting automated pipeline..."
echo "Estimated time: 6-8 hours"
echo "Log: output/production_training/pipeline_*.log"
echo ""

bash scripts/auto_sharpe_optimization.sh --yes

echo ""
echo "================================================================================"
echo "PIPELINE COMPLETED!"
echo "================================================================================"
