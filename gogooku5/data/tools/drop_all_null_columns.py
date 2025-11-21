#!/usr/bin/env python3
"""Drop columns that are entirely NULL from a merged ML dataset.

This utility is intended as a lightweight post-processing step before training.
It scans a parquet file, identifies columns where every row is NULL, and drops
them while preserving key identifier/target columns.

Example:

.. code-block:: bash

    PYTHONPATH=gogooku5/data/src \\
      python gogooku5/data/tools/drop_all_null_columns.py \\
        --input  data/output/datasets/ml_dataset_2025_with_graph33_basis.parquet \\
        --output data/output/datasets/ml_dataset_2025_pruned.parquet \\
        --keep-col Date --keep-col Code --keep-col target_1d --keep-col target_5d
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import polars as pl


def _find_all_null_columns(df: pl.DataFrame, keep: Iterable[str]) -> List[Tuple[str, int]]:
    """Return list of (column_name, null_count) for columns that are 100% NULL.

    Columns listed in ``keep`` are never dropped, even if they are all NULL.
    """

    height = df.height
    if height == 0:
        return []

    keep_set = set(keep)
    null_counts_df = df.null_count()
    null_counts_dict = null_counts_df.to_dict(as_series=False)
    to_drop: List[Tuple[str, int]] = []
    for name, null_list in null_counts_dict.items():
        if name in keep_set:
            continue
        null_count = null_list[0]  # Get first (and only) row value
        if null_count == height:
            to_drop.append((name, null_count))
    return to_drop


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Drop columns that are entirely NULL from a parquet dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input parquet file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output parquet file with all-NULL columns removed.",
    )
    parser.add_argument(
        "--keep-col",
        action="append",
        default=[],
        help="Column name to always keep (can be specified multiple times).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional JSON report file to write list of dropped columns.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""

    args = parse_args()
    input_path: Path = args.input
    output_path: Path = args.output

    if not input_path.exists():
        print(f"❌ Input dataset not found: {input_path}")
        return 1

    print(f"📂 Loading dataset from {input_path}")
    df = pl.read_parquet(str(input_path))
    print(f"   rows={df.height:,}, cols={df.width}")

    # Always keep common identifier columns if present.
    keep_cols = set(args.keep_col or [])
    for col in ("Date", "date", "Code", "code"):
        if col in df.columns:
            keep_cols.add(col)

    to_drop = _find_all_null_columns(df, keep_cols)
    if not to_drop:
        print("ℹ️  No all-NULL columns found. Writing dataset unchanged.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(str(output_path), compression="zstd")
        return 0

    drop_names = [name for name, _ in to_drop]
    print(f"🧹 Dropping {len(drop_names)} all-NULL columns:")
    for name, null_count in to_drop:
        print(f"   - {name} (NULLs={null_count})")

    df_clean = df.drop(drop_names)
    print(f"   → rows={df_clean.height:,}, cols={df_clean.width}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"💾 Writing pruned dataset to {output_path}")
    df_clean.write_parquet(str(output_path), compression="zstd")

    if args.report:
        report = {
            "input": str(input_path),
            "output": str(output_path),
            "dropped_columns": [name for name, _ in to_drop],
            "num_dropped": len(to_drop),
            "rows": df_clean.height,
            "cols": df_clean.width,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w") as f:
            json.dump(report, f, indent=2)
        print(f"📝 Report written to {args.report}")

    print("✅ Completed all-NULL column pruning.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
