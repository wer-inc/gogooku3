#!/usr/bin/env python
"""Compare gogooku3 and gogooku5 dataset outputs for parity."""
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare gogooku3 vs gogooku5 datasets"
    )
    parser.add_argument("--g3", required=True, help="Path to gogooku3 parquet")
    parser.add_argument("--g5", required=True, help="Path to gogooku5 parquet")
    parser.add_argument(
        "--output", required=True, help="Where to write JSON diff summary"
    )
    parser.add_argument(
        "--columns",
        nargs="*",
        default=[],
        help="Optional list of columns to compare (defaults to intersection)",
    )
    parser.add_argument("--rel-threshold", type=float, default=1e-4)
    parser.add_argument("--abs-threshold", type=float, default=1e-6)
    return parser.parse_args()


def normalise(df: pl.DataFrame) -> pl.DataFrame:
    rename = {col: col.lower() for col in df.columns}
    df = df.rename(rename)
    if "date" in df.columns:
        df = df.with_columns(pl.col("date").cast(pl.Date))
    if "code" in df.columns:
        df = df.with_columns(pl.col("code").cast(pl.Utf8))
    return df


def compute_diffs(
    df_g3: pl.DataFrame, df_g5: pl.DataFrame, cols: Iterable[str]
) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for col in cols:
        s3 = df_g3[col]
        s5 = df_g5[col]
        diff = (s5 - s3).abs()
        rel = diff / (s3.abs() + 1e-9)
        stats[col] = {
            "count": float(len(diff)),
            "abs_max": float(diff.max()),
            "abs_mean": float(diff.mean()),
            "rel_max": float(rel.max()),
            "rel_mean": float(rel.mean()),
        }
    return stats


def main() -> None:
    args = parse_args()
    df_g3 = normalise(pl.read_parquet(args.g3))
    df_g5 = normalise(pl.read_parquet(args.g5))

    join_keys = [
        c for c in ["code", "date"] if c in df_g3.columns and c in df_g5.columns
    ]
    if not join_keys:
        raise SystemExit("No common join keys (code/date) found")

    df_joined = df_g3.join(df_g5, on=join_keys, how="inner", suffix="_g5")

    if args.columns:
        columns = args.columns
    else:
        columns = [
            col
            for col in df_g3.columns
            if col in df_g5.columns and col not in join_keys
        ]

    diffs = {}
    for col in columns:
        c_g3 = df_joined[col]
        c_g5 = (
            df_joined[f"{col}_g5"]
            if f"{col}_g5" in df_joined.columns
            else df_joined[col]
        )
        diff = (c_g5 - c_g3).abs()
        rel = diff / (c_g3.abs() + 1e-9)
        diffs[col] = {
            "count": float(len(diff)),
            "abs_max": float(diff.max()),
            "abs_mean": float(diff.mean()),
            "rel_max": float(rel.max()),
            "rel_mean": float(rel.mean()),
            "abs_exceeds": float((diff > args.abs_threshold).sum()),
            "rel_exceeds": float((rel > args.rel_threshold).sum()),
        }

    Path(args.output).write_text(json.dumps(diffs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
