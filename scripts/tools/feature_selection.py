#!/usr/bin/env python3
from __future__ import annotations

"""
Offline feature selection runner.

Usage:
  python scripts/tools/feature_selection.py \
    --input output/ml_dataset_latest_full.parquet \
    --method mutual_info --top-k 120 --target target_1d \
    --output output/selected_features.json
"""

import argparse
from pathlib import Path
import polars as pl
from src.gogooku3.features.feature_selector import SelectionConfig, select_features, save_selected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Parquet dataset path")
    ap.add_argument("--method", default="mutual_info", choices=["mutual_info", "lasso", "random_forest"])
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--min-importance", type=float, default=0.0)
    ap.add_argument("--target", default="target_1d")
    ap.add_argument("--output", default="output/selected_features.json")
    args = ap.parse_args()

    df = pl.read_parquet(args.input)
    cfg = SelectionConfig(method=args.method, top_k=args.top_k, min_importance=args.min_importance, target_column=args.target)
    selected = select_features(df, cfg)
    save_selected(selected, Path(args.output))
    print(f"✅ selected {len(selected)} features → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

