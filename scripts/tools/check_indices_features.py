#!/usr/bin/env python3
"""
Self-check tool for indices features in an equity dataset parquet.

Usage:
  python scripts/tools/check_indices_features.py --dataset output/ml_dataset_latest_full.parquet

Checks:
  - Presence and non-null coverage of spread_* columns (constant per Date)
  - Presence and non-null coverage of breadth_* columns
  - Presence and distribution of is_halt_20201001 flag
  - Presence and non-null coverage of sect_* columns
  - Optional: verify spreads are equal across codes per Date (sampled dates)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def coverage(series: pl.Series) -> float:
    n = len(series)
    if n == 0:
        return 0.0
    return float((n - int(series.null_count())) / n)


def main() -> int:
    ap = argparse.ArgumentParser(description="Check indices features in dataset parquet")
    ap.add_argument("--dataset", type=Path, required=True, help="Path to equity dataset parquet")
    ap.add_argument("--sample-dates", type=int, default=5, help="Number of sample dates to check spread constancy")
    args = ap.parse_args()

    assert args.dataset.exists(), f"Dataset not found: {args.dataset}"
    df = pl.read_parquet(args.dataset)
    print(f"Loaded: {args.dataset} shape={df.shape}")

    # Spreads
    spread_cols = [c for c in df.columns if c.startswith("spread_")]
    print(f"Spread columns ({len(spread_cols)}): {sorted(spread_cols)[:10]}{'...' if len(spread_cols)>10 else ''}")
    for c in spread_cols:
        cov = coverage(df[c])
        print(f"  {c:>20}: coverage={cov:.1%}")

    # Breadth
    breadth_cols = [c for c in df.columns if c.startswith("breadth_")]
    print(f"Breadth columns ({len(breadth_cols)}): {breadth_cols}")
    for c in breadth_cols:
        cov = coverage(df[c])
        print(f"  {c:>20}: coverage={cov:.1%}")

    # Halt flag
    if "is_halt_20201001" in df.columns:
        halt_days = df.filter(pl.col("is_halt_20201001") == 1)["Date"].unique().to_list()
        print(f"Halt flag present. Days flagged: {halt_days}")
    else:
        print("Halt flag not found: is_halt_20201001")

    # Sector index features
    sect_cols = [c for c in df.columns if c.startswith("sect_")]
    print(f"Sector index columns ({len(sect_cols)}): {sorted(sect_cols)[:10]}{'...' if len(sect_cols)>10 else ''}")
    for c in sect_cols[:20]:
        cov = coverage(df[c])
        print(f"  {c:>20}: coverage={cov:.1%}")

    # Constancy check: spreads should be same across stocks on same date
    if spread_cols:
        dates = (
            df.select("Date").unique().sort("Date")["Date"].head(args.sample_dates).to_list()  # type: ignore
        )
        for d in dates:
            sub = df.filter(pl.col("Date") == d)
            for c in spread_cols[:5]:
                vals = sub[c].drop_nulls().unique().to_list()
                if len(vals) > 1:
                    print(f"[WARN] Spread {c} has {len(vals)} unique values on {d}")
        print("Spread constancy check (sample) completed.")

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

