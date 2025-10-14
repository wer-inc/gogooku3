#!/bin/bash

echo "============================================================"
echo "ATFT-GAT-FAN Training with Advanced Graph Features"
echo "Target Sharpe Ratio: 0.849"
echo "============================================================"

# Set optimal graph parameters
export USE_ADV_GRAPH_TRAIN=1
export GRAPH_CORR_METHOD=ewm_demean
export EWM_HALFLIFE=30
export SHRINKAGE_GAMMA=0.1
export GRAPH_K=15
export GRAPH_EDGE_THR=0.25
export GRAPH_SYMMETRIC=1

# Training parameters
export TRAIN_RATIO=0.7
export VAL_RATIO=0.2
export GAP_DAYS=5
export NUM_WORKERS=8
export PREFETCH_FACTOR=4
export PIN_MEMORY=1
export PERSISTENT_WORKERS=1

# Loss function parameters
export USE_HUBER=1
export HUBER_DELTA=0.01
export SHARPE_EPS=1e-8

# Disable Student-t for now (can enable later)
export ENABLE_STUDENT_T=0
export USE_T_NLL=0

echo "Graph Parameters:"
echo "  - K (neighbors): 15"
echo "  - Edge threshold: 0.25"
echo "  - EWM halflife: 30 days"
echo "  - Shrinkage gamma: 0.1"
echo "  - Correlation method: ewm_demean"

# Run training with advanced graph
python scripts/integrated_ml_training_pipeline.py \
    --batch-size 512 \
    --max-epochs 75 \
    --adv-graph-train \
    2>&1 | tee training_with_graph.log

echo "============================================================"
echo "Training completed! Check training_with_graph.log for details"
echo "============================================================"