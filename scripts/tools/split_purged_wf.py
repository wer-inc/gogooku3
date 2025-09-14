#!/usr/bin/env python3
from __future__ import annotations

"""
Purged Walk-Forward splitter with embargo for time series evaluation.

Usage:
  python scripts/tools/split_purged_wf.py --dataset output/ml_dataset_latest_full.parquet \
    --n-splits 5 --embargo-days 20

Outputs fold boundaries and sample counts; can optionally save to JSON.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl


def unique_sorted_dates(df: pl.DataFrame) -> list:
    return df.select("Date").unique().sort("Date")["Date"].to_list()


def build_purged_wf_splits(dates: list, n_splits: int, embargo_days: int) -> list[dict[str, Any]]:
    if not dates:
        return []
    n = len(dates)
    fold_len = max(n // n_splits, 1)
    splits: list[dict[str, Any]] = []
    for i in range(n_splits):
        test_start_idx = i * fold_len
        test_end_idx = n if i == n_splits - 1 else min((i + 1) * fold_len, n)
        test_dates = dates[test_start_idx:test_end_idx]
        if not test_dates:
            continue
        test_start_date = test_dates[0]
        # Embargo: drop last embargo_days from train if they overlap test start
        emb_idx = max(0, test_start_idx - embargo_days)
        train_dates = dates[:emb_idx]
        splits.append(
            {
                "fold": i,
                "train_start": dates[0],
                "train_end": dates[emb_idx - 1] if emb_idx > 0 else None,
                "test_start": test_start_date,
                "test_end": test_dates[-1],
                "n_train_days": len(train_dates),
                "n_test_days": len(test_dates),
            }
        )
    return splits


def main() -> int:
    ap = argparse.ArgumentParser(description="Purged Walk-Forward splitter")
    ap.add_argument("--dataset", type=Path, required=True, help="Parquet with Date column")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--embargo-days", type=int, default=20)
    ap.add_argument("--save-json", type=Path, default=None, help="Optional path to save splits JSON")
    args = ap.parse_args()

    if not args.dataset.exists():
        print(f"ERROR: dataset not found: {args.dataset}")
        return 1

    df = pl.read_parquet(args.dataset, columns=["Date"]).with_columns(pl.col("Date").cast(pl.Date))
    dates = unique_sorted_dates(df)
    splits = build_purged_wf_splits(dates, args.n_splits, args.embargo_days)
    if not splits:
        print("No splits generated (empty or invalid dataset)")
        return 1

    print("Purged Walk-Forward (with embargo) splits:")
    for s in splits:
        print(
            f" fold {s['fold']}: train {s['train_start']} → {s['train_end']}  |  test {s['test_start']} → {s['test_end']}"
            f"  (train days={s['n_train_days']}, test days={s['n_test_days']})"
        )

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(splits, f, indent=2, default=str)
        print(f"Saved splits JSON: {args.save_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

