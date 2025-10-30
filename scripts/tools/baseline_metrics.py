#!/usr/bin/env python3
from __future__ import annotations

"""
Baseline cross-sectional RankIC and hit rate using a simple momentum factor.

Usage:
  python scripts/tools/baseline_metrics.py output/ml_dataset_latest_full.parquet \
      --factor returns_5d --horizons 1,5,10,20

Notes:
  - RankIC is computed per Date as the Pearson correlation of within-date ranks
    between the factor at t and forward return label feat_ret_{h}d at t.
  - Requires columns: Date, Code, factor, feat_ret_{h}d.
"""

import argparse
from collections.abc import Iterable
from pathlib import Path

import polars as pl


def ranks(expr: pl.Expr, by: str) -> pl.Expr:
    return expr.rank(method="average").over(by)


def compute_rankic(df: pl.DataFrame, factor_col: str, label_col: str) -> float | None:
    if not {"Date", factor_col, label_col}.issubset(df.columns):
        return None
    # Prepare per-date ranks and compute Pearson on ranks
    d = df.select(["Date", factor_col, label_col]).drop_nulls([factor_col, label_col])
    if d.is_empty():
        return None
    d = d.with_columns(
        [
            ranks(pl.col(factor_col), by="Date").alias("_r_f"),
            ranks(pl.col(label_col), by="Date").alias("_r_y"),
        ]
    )
    # Compute daily correlations and average
    daily = d.group_by("Date").agg(pl.pearson_corr(pl.col("_r_f"), pl.col("_r_y")).alias("rho")).drop_nulls()
    if daily.is_empty():
        return None
    return float(daily["rho"].mean())


def compute_hit_rate(df: pl.DataFrame, factor_col: str, label_col: str) -> float | None:
    d = df.select([factor_col, label_col]).drop_nulls([factor_col, label_col])
    if d.is_empty():
        return None
    hr = (
        (pl.sign(pl.col(factor_col)) == pl.sign(pl.col(label_col)))
        .cast(pl.Int8)
        .mean()
        .alias("hit")
    )
    return float(d.select(hr)["hit"][0])


def main() -> int:
    ap = argparse.ArgumentParser(description="Baseline RankIC metrics")
    ap.add_argument("dataset", type=Path, help="Parquet with factor and labels")
    ap.add_argument("--factor", type=str, default="returns_5d", help="Factor column to use (default: returns_5d)")
    ap.add_argument("--horizons", type=str, default="1,5,10,20")
    args = ap.parse_args()

    if not args.dataset.exists():
        print(f"ERROR: dataset not found: {args.dataset}")
        return 1

    df = pl.read_parquet(args.dataset, columns=["Date", "Code", args.factor, "feat_ret_1d", "feat_ret_5d", "feat_ret_10d", "feat_ret_20d"])  # type: ignore[list-item]
    horizons: Iterable[int] = (int(x.strip()) for x in args.horizons.split(",") if x.strip())

    print("=== Baseline metrics ===")
    for h in horizons:
        label = f"feat_ret_{h}d"
        if label not in df.columns:
            print(f"h={h}d: label {label} missing; skip")
            continue
        rankic = compute_rankic(df, args.factor, label)
        hit = compute_hit_rate(df, args.factor, label)
        print(
            f"h={h}d: RankIC={None if rankic is None else f'{rankic:.4f}'}  "
            f"HitRate={None if hit is None else f'{hit*100:.1f}%'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

