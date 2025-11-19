#!/usr/bin/env python3
"""Drop high-NULL columns from a parquet dataset.

This utility is similar to drop_all_null_columns.py, but uses a configurable
NULL率しきい値で列を削除します。学習前に「ほぼ情報が無い」特徴量を
まとめて落とすための軽量なポストプロセスです。

Example:

.. code-block:: bash

    PYTHONPATH=gogooku5/data/src \\
      python gogooku5/data/tools/drop_high_null_columns.py \\
        --input  output_g5/datasets/ml_dataset_2023_2024_final.parquet \\
        --output output_g5/datasets/ml_dataset_2023_2024_clean.parquet \\
        --threshold 90.0 \\
        --keep-col Date --keep-col Code
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import polars as pl


def _find_high_null_columns(
    df: pl.DataFrame,
    keep: Iterable[str],
    *,
    threshold: float,
) -> List[Tuple[str, int, float]]:
    """Return list of (column_name, null_count, null_rate) for high-NULL columns.

    Columns listed in ``keep`` are never dropped, even if閾値を超えていても保持します。
    """

    height = df.height
    if height == 0:
        return []

    keep_set = set(keep)
    null_counts_df = df.null_count()
    null_counts_dict = null_counts_df.to_dict(as_series=False)
    to_drop: List[Tuple[str, int, float]] = []
    for name, null_list in null_counts_dict.items():
        if name in keep_set:
            continue
        null_count = null_list[0]
        rate = (null_count / height) * 100.0
        if rate >= threshold:
            to_drop.append((name, null_count, rate))
    return to_drop


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Drop columns whose NULL rate is greater than or equal to a threshold.",
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
        help="Output parquet file with high-NULL columns removed.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=90.0,
        help="Drop columns with NULL率 >= threshold (percentage).",
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
        help="Optional JSON report file to write list of dropped columns and null stats.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""

    args = parse_args()
    input_path: Path = args.input
    output_path: Path = args.output
    threshold: float = args.threshold

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

    to_drop = _find_high_null_columns(df, keep_cols, threshold=threshold)
    if not to_drop:
        print(f"ℹ️  No columns with NULL率 >= {threshold:.2f}% found. Writing dataset unchanged.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(str(output_path), compression="zstd")
        return 0

    drop_names = [name for name, _, _ in to_drop]
    print(f"🧹 Dropping {len(drop_names)} high-NULL columns (threshold={threshold:.2f}%):")
    for name, null_count, rate in sorted(to_drop, key=lambda t: t[2], reverse=True):
        print(f"   - {name}: NULLs={null_count:,} ({rate:.2f}%)")

    df_clean = df.drop(drop_names)
    print(f"   → rows={df_clean.height:,}, cols={df_clean.width}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"💾 Writing cleaned dataset to {output_path}")
    df_clean.write_parquet(str(output_path), compression="zstd")

    if args.report:
        report = {
            "input": str(input_path),
            "output": str(output_path),
            "threshold": threshold,
            "dropped_columns": [
                {"name": name, "null_count": null_count, "null_rate": rate}
                for name, null_count, rate in to_drop
            ],
            "num_dropped": len(to_drop),
            "rows": df_clean.height,
            "cols": df_clean.width,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w") as f:
            json.dump(report, f, indent=2)
        print(f"📝 Report written to {args.report}")

    print("✅ Completed high-NULL column pruning.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

