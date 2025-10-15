#!/bin/bash
# Sharpe Optimization Test Run (Step C)
# Enables turnover penalty and proper metric logging

# Enable turnover penalty (already implemented in MultiHorizonLoss)
export USE_TURNOVER_PENALTY=1
export TURNOVER_WEIGHT=0.2
export TURNOVER_ALPHA=0.9

# Enable existing loss components
export USE_RANKIC=1
export RANKIC_WEIGHT=0.1
export USE_CS_IC=1
export CS_IC_WEIGHT=0.15

# DataLoader settings - FORCE SINGLE WORKER to avoid crashes
export FORCE_SINGLE_PROCESS=1
export ALLOW_UNSAFE_DATALOADER=0

# Run the test
python scripts/integrated_ml_training_pipeline.py \
  --config configs/atft/config_sharpe_optimized.yaml \
  --max-epochs 10 \
  --batch-size 1024 \
  --data-path output/ml_dataset_latest_full.parquet \
  > /tmp/sharpe_test_run.log 2>&1 &

echo "Test run started with PID: $!"
echo "Monitor with: tail -f /tmp/sharpe_test_run.log"
echo "Validate with: python scripts/validate_test_run.py --log-file /tmp/sharpe_test_run.log"
