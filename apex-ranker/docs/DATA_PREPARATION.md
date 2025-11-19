# Apex Ranker Dataset Preparation (SecId + Targets)

This document describes how to prepare the gogooku5 dataset so that Apex Ranker can train against the latest SecId-aware data. The steps assume the dataset rebuild pipeline writes to `/workspace/gogooku3/output_g5`.

## 1. Rebuild gogooku5 dataset (SecId v1.3.0)

Use the Dagster job or the CLI to produce the merged dataset:

```
/workspace/gogooku3/output_g5/ml_dataset_full.parquet
```

This file does **not** contain target columns yet.

## 2. Add target columns (forward returns)

Run the `add_future_returns.py` helper in the repo root:

```bash
cd /workspace/gogooku3
./venv/bin/python add_future_returns.py \
  --input output_g5/ml_dataset_full.parquet \
  --output output_g5/ml_dataset_full_with_targets.parquet \
  --symlink output_g5/ml_dataset_latest_with_targets.parquet \
  --horizons 1 5 10 20
```

This writes `target_1d`, `target_5d`, etc. into the dataset while streaming (no full in-memory load).

## 3. Point Apex Ranker to the new dataset

All scripts in `apex-ranker/scripts/path_constants.py` now default to:

```
DATASET_RAW  = /workspace/gogooku3/output_g5/ml_dataset_full_with_targets.parquet
DATASET_CLEAN = /workspace/gogooku3/output_g5/ml_dataset_full_with_targets_clean.parquet
```

You can override them via environment variables if needed:

```bash
export DATASET_RAW=/workspace/gogooku3/output_g5/ml_dataset_full_with_targets.parquet
export DATASET_CLEAN=/workspace/gogooku3/output_g5/ml_dataset_full_with_targets_clean.parquet
```

## 4. Optional: quality filtering

If you need to run the quality gate:

```bash
cd /workspace/gogooku3/apex-ranker
python scripts/filter_dataset_quality.py \
  --input "$DATASET_RAW" \
  --output "$DATASET_CLEAN"
```

## 5. Training / inference

All Apex Ranker configs (`configs/*.yaml`) can now point to `$DATASET_RAW` or `$DATASET_CLEAN`. For example:

```yaml
parquet_path: /workspace/gogooku3/output_g5/ml_dataset_full_with_targets.parquet
```

By ensuring the dataset contains SecId (Int32) and target columns, Apex Ranker can train immediately after each gogooku5 rebuild.
