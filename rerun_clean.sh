#!/bin/bash
PYTHONPATH=data/src TMPDIR=/workspace/tmp POLARS_TEMP_DIR=/workspace/tmp \
python gogooku5/data/tools/drop_high_null_columns.py \
  --input output_g5/ml_dataset_20230101_20241231_20251120_081840_full.parquet \
  --output output_g5/ml_dataset_2023_2024_clean.parquet \
  --threshold 90.0
