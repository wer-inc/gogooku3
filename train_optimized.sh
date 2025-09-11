#!/bin/bash

echo "============================================================"
echo "Optimized ATFT-GAT-FAN Training for Sharpe 0.849"
echo "With improved metrics, loss function, and phase training"
echo "============================================================"

# Enhanced graph parameters for better performance
export USE_ADV_GRAPH_TRAIN=1
export GRAPH_CORR_METHOD=ewm_demean
export EWM_HALFLIFE=20          # Reduced from 30 for recent focus
export SHRINKAGE_GAMMA=0.1
export GRAPH_K=20               # Increased from 15 for more connections
export GRAPH_EDGE_THR=0.2       # Reduced from 0.25 for more edges
export GRAPH_SYMMETRIC=1

# Optimized training parameters
export TRAIN_RATIO=0.7
export VAL_RATIO=0.2
export GAP_DAYS=5
export NUM_WORKERS=8
export PREFETCH_FACTOR=4
export PIN_MEMORY=1
export PERSISTENT_WORKERS=1

# Enhanced loss function parameters
export USE_HUBER=1
export HUBER_DELTA=0.01
export SHARPE_EPS=1e-8
export USE_RANKIC=1             # Enable RankIC loss
export RANKIC_WEIGHT=0.15
export SHARPE_WEIGHT=0.1

# Disable Student-t for initial training (can enable later)
export ENABLE_STUDENT_T=0
export USE_T_NLL=0

echo "Enhanced Configuration:"
echo "  - Graph K (neighbors): 20"
echo "  - Edge threshold: 0.2"
echo "  - EWM halflife: 20 days"
echo "  - Huber + RankIC + Sharpe loss"
echo "  - Optimized phase training"

# Run optimized training
python scripts/integrated_ml_training_pipeline.py \
    --batch-size 512 \
    --max-epochs 75 \
    --lr 0.0005 \
    --adv-graph-train \
    2>&1 | tee training_optimized.log

echo "============================================================"
echo "Optimized training completed! Check training_optimized.log"
echo "============================================================"