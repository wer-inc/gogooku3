#!/usr/bin/env python3
"""Add graph-based peer features to a merged ML dataset.

This tool is intended to be run *after* chunk merging. It reads a
merged Parquet dataset, applies :class:`GraphFeatureEngineer` over the
full date range, and writes out a new Parquet file with ``graph_*``
columns attached.

Usage (example):

    python gogooku5/data/tools/add_graph_features_full.py \\
        --input /workspace/gogooku3/output_g5/datasets/ml_dataset_2020_2025Q3_full.parquet \\
        --output /workspace/gogooku3/output_g5/datasets/ml_dataset_2020_2025Q3_with_graph.parquet \\
        --date-col Date --code-col Code
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from builder.features.core.graph.features import GraphFeatureConfig, GraphFeatureEngineer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add graph_* features to a merged ML dataset.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the merged ML dataset parquet file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the parquet file with graph_* features.",
    )
    parser.add_argument(
        "--date-col",
        default="Date",
        help="Name of the date column in the dataset (default: Date).",
    )
    parser.add_argument(
        "--code-col",
        default="Code",
        help="Name of the security code column in the dataset (default: Code).",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=60,
        help="Rolling window length (in days) used for graph construction (default: 60).",
    )
    parser.add_argument(
        "--min-observations",
        type=int,
        default=20,
        help="Minimum number of observations required for a code to be eligible (default: 20).",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.3,
        help="Absolute correlation threshold to treat an edge as 'strong' (default: 0.3).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"[ERROR] Input dataset not found: {input_path}", file=sys.stderr)
        return 1

    print(f"[INFO] Loading dataset from: {input_path}")
    lf = pl.scan_parquet(str(input_path))
    df = lf.collect()
    if df.is_empty():
        print("[WARN] Input dataset is empty; nothing to do.", file=sys.stderr)
        return 1

    cfg = GraphFeatureConfig(
        code_column=args.code_col,
        date_column=args.date_col,
        window_days=args.window_days,
        min_observations=args.min_observations,
        correlation_threshold=args.correlation_threshold,
        # Keep default return_column resolution logic.
    )
    engineer = GraphFeatureEngineer(config=cfg)

    print(
        f"[INFO] Applying GraphFeatureEngineer with window_days={cfg.window_days}, "
        f"min_observations={cfg.min_observations}, corr_threshold={cfg.correlation_threshold}"
    )
    out = engineer.add_features(df)

    # Ensure output directory exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Writing dataset with graph features to: {output_path}")
    out.write_parquet(str(output_path))
    print("[INFO] Completed graph feature augmentation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

