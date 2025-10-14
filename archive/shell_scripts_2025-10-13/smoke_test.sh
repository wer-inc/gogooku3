#!/bin/bash

echo "============================================================"
echo "Smoke Test: ATFT-GAT-FAN Training (1 epoch)"
echo "============================================================"

# Export environment variables for testing
export ENABLE_STUDENT_T=0  # Disable Student-t for initial test
export USE_T_NLL=0
export TRAIN_RATIO=0.7
export VAL_RATIO=0.2
export GAP_DAYS=5
export NUM_WORKERS=4
export PREFETCH_FACTOR=2
export PIN_MEMORY=1

# Run 1 epoch smoke test
python scripts/integrated_ml_training_pipeline.py \
    --batch-size 256 \
    --max-epochs 1 \
    --sample-size 10000

echo "============================================================"
echo "Smoke test completed!"
echo "============================================================"