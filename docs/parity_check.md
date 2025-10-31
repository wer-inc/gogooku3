# Dataset Parity Workflow

This document describes how to compare a gogooku5 dataset against the legacy gogooku3 output to ensure feature parity.

## CLI Usage

```bash
python gogooku5/data/scripts/compare_parity.py \
    /path/to/gogooku3/output/ml_dataset_latest_full.parquet \
    gogooku5/data/output/ml_dataset_latest.parquet \
    --output-json parity_report.json
```

The script prints a human-readable summary (row deltas, schema differences, numeric deviations) and, when `--output-json` is supplied, writes a detailed report for automated tooling.

### Sample JSON Summary

```json
{
  "schema_mismatch": [],
  "rows_only_reference": 0,
  "rows_only_candidate": 0,
  "column_diffs": [
    {
      "name": "foreign_sentiment",
      "max_abs_diff": 7.4e-07,
      "mean_abs_diff": 2.1e-07
    }
  ]
}
```

Interpretation:

- `schema_mismatch` lists columns where types diverge; empty when aligned.
- `rows_only_*` report key mismatches (using `Code, Date` composite key).
- Each `column_diffs` entry provides numeric tolerance metrics and flags reference-only or candidate-only columns when present.

## Health Check Integration

`tools/project-health-check.sh` consumes the optional environment variables `PARITY_BASELINE_PATH` (legacy gogooku3 parquet) and `PARITY_CANDIDATE_PATH` (explicit gogooku5 parquet, if auto-detection is insufficient). Example:

```bash
PARITY_BASELINE_PATH=/path/to/gogooku3/ml_dataset_latest_full.parquet \
PARITY_CANDIDATE_PATH=gogooku5/data/output/ml_dataset_latest_full.parquet \
python tools/project-health-check.sh
```

The parity results appear under the new **Dataset parity check** section. If `PARITY_CANDIDATE_PATH` is supplied, the script logs the override being used; otherwise it falls back to the auto-detected dataset from the data pipeline status stage. Differences trigger warnings; exact matches are logged as successes.

## Interpreting Output

- **schema mismatch / column diffs** – columns that are missing or have differing types.
- **row delta** – counts of rows that appear in one dataset but not the other (based on `Code,Date` keys by default).
- **numeric diffs** – max and mean absolute differences for numeric columns that exceed a tolerance (`>1e-6` flagged in the health check summary).

The JSON report includes per-column details so downstream automation can enforce stricter tolerance thresholds if required.
