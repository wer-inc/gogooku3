#!/bin/bash

# Export environment variables for Student-t distribution
export USE_DAY_BATCH=1
export MIN_NODES_PER_DAY=1000
export GRAPH_CORR_METHOD=ewm_demean
export EWM_HALFLIFE=30
export SHRINKAGE_GAMMA=0.1
export GRAPH_K=15
export GRAPH_EDGE_THR=0.25
export GRAPH_SYMMETRIC=1
export ENABLE_STUDENT_T=1
export USE_T_NLL=1
export NLL_WEIGHT=0.02
export ENABLE_QUANTILES=0
export TRAIN_RATIO=0.7
export VAL_RATIO=0.2
export GAP_DAYS=5
export NUM_WORKERS=8
export PREFETCH_FACTOR=4
export PIN_MEMORY=1
export PERSISTENT_WORKERS=1
export SHARPE_EPS=1e-8

# Run training pipeline with 1 epoch for testing
python scripts/integrated_ml_training_pipeline.py --batch-size 512 --max-epochs 1 --adv-graph-train