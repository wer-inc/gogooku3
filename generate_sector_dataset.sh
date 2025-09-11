#!/bin/bash

echo "============================================================"
echo "Generating Sector33-enriched Dataset"
echo "Date range: 2024-01-01 to 2025-01-01"
echo "============================================================"

# Set environment variables
export PYTHONPATH=/home/ubuntu/gogooku3-standalone:$PYTHONPATH

# Run the full dataset pipeline with sector information
python scripts/pipelines/run_full_dataset.py \
    --jquants \
    --start-date 2024-01-01 \
    --end-date 2025-01-01 \
    --sector-series-mcap auto \
    --sector-te-targets target_5d \
    --sector-te-levels 33 \
    2>&1 | tee sector_dataset_generation.log

echo "============================================================"
echo "Dataset generation completed!"
echo "Check output/ml_dataset_*_full.parquet for the result"
echo "============================================================"