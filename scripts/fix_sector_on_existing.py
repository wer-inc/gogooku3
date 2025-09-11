#!/usr/bin/env python3
"""
One-shot fixer: attach sector features (base, series, encodings, relative, TE)
to an existing dataset parquet using a listed_info parquet snapshot.

Usage:
  python scripts/fix_sector_on_existing.py \
    --input output/ml_dataset_20240101_20250101_XXXXXXXX_full.parquet \
    --listed-info output/listed_info_history_20250101.parquet \
    --output output/ml_dataset_20240101_20250101_XXXXXXXX_full_sectorfix.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
import polars as pl

# Ensure project root on path for imports
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data.ml_dataset_builder import MLDatasetBuilder


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fix existing dataset by attaching sector features")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--listed-info", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--te-targets", default="target_5d", type=str, help="Comma-separated targets for TE")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    assert args.input.exists(), f"Input parquet not found: {args.input}"
    assert args.listed_info.exists(), f"Listed info parquet not found: {args.listed_info}"

    df = pl.read_parquet(args.input)
    info = pl.read_parquet(args.listed_info)

    b = MLDatasetBuilder()
    # 1) Sector base (handles snapshot valid_from fallback and LocalCode join)
    df = b.add_sector_features(df, info)
    # 2) Sector series (33-level eq + mcap if available)
    df = b.add_sector_series(df, level="33", windows=(1, 5, 20), series_mcap="auto")
    # 3) Encodings
    df = b.add_sector_encodings(df, onehot_17=True, onehot_33=False, freq_daily=True)
    # 4) Relative-to-sector
    df = b.add_relative_to_sector(df, level="33", x_cols=("returns_5d", "ma_gap_5_20"))
    # 5) Target encoding for requested targets
    targets = [s.strip() for s in (args.te_targets or "").split(",") if s.strip()]
    if not targets:
        targets = ["target_5d"]
    for t in targets:
        df = b.add_sector_target_encoding(df, target_col=t, level="33", k_folds=5, lag_days=1, m=100.0)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.output)
    print(f"Saved fixed dataset: {args.output}")

    # Basic coverage printout
    def cov(col: str) -> float:
        return float(df[col].is_not_null().sum()) / len(df) if col in df.columns else 0.0
    for c in ["sector33_code", "sec_ret_1d_eq", "rel_to_sec_5d", "te33_sec_target_5d"]:
        if c in df.columns:
            print(f"coverage {c}: {cov(c):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

