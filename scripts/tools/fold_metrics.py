#!/usr/bin/env python3
from __future__ import annotations

"""
Compute per-fold RankIC and HitRate using a splits JSON (purged WF + embargo).

Usage:
  python scripts/tools/fold_metrics.py \
    --dataset output/ml_dataset_latest_full.parquet \
    --splits-json output/eval_splits_5fold_20d.json \
    --factors returns_5d,ret_1d_vs_sec,rank_ret_1d,graph_degree \
    --horizons 1,5,10,20 \
    --out reports/fold_metrics.csv
"""

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import polars as pl


def ranks(expr: pl.Expr, by: str) -> pl.Expr:
    return expr.rank(method="average").over(by)


def rankic(df: pl.DataFrame, factor: str, label: str) -> float | None:
    if not {"Date", factor, label}.issubset(df.columns):
        return None
    d = df.select(["Date", factor, label]).drop_nulls([factor, label])
    if d.is_empty():
        return None
    d = d.with_columns([
        ranks(pl.col(factor), by="Date").alias("_rf"),
        ranks(pl.col(label), by="Date").alias("_ry"),
    ])
    daily = d.group_by("Date").agg(pl.pearson_corr(pl.col("_rf"), pl.col("_ry")).alias("rho")).drop_nulls()
    if daily.is_empty():
        return None
    return float(daily["rho"].mean())


def hitrate(df: pl.DataFrame, factor: str, label: str) -> float | None:
    if not {factor, label}.issubset(df.columns):
        return None
    d = df.select([factor, label]).drop_nulls([factor, label])
    if d.is_empty():
        return None
    return float(d.select(((pl.sign(pl.col(factor)) == pl.sign(pl.col(label))).cast(pl.Int8).mean()).alias("hit"))["hit"][0])


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-fold metrics using splits JSON")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--splits-json", type=Path, required=True)
    ap.add_argument("--factors", type=str, default="returns_5d,ret_1d_vs_sec,rank_ret_1d,graph_degree")
    ap.add_argument("--horizons", type=str, default="1,5,10,20")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not args.dataset.exists() or not args.splits_json.exists():
        print("ERROR: dataset or splits JSON not found")
        return 1

    with open(args.splits_json, encoding="utf-8") as f:
        splits = json.load(f)
    if not isinstance(splits, list) or not splits:
        print("ERROR: invalid or empty splits JSON")
        return 1

    factors = [s.strip() for s in args.factors.split(",") if s.strip()]
    horizons: Iterable[int] = (int(x.strip()) for x in args.horizons.split(",") if x.strip())
    label_cols = [f"feat_ret_{h}d" for h in horizons]

    cols = ["Code", "Date"] + factors + label_cols
    df = pl.read_parquet(args.dataset, columns=[c for c in cols if c])

    rows = []
    for s in splits:
        fold = s.get("fold")
        test_start = s.get("test_start")
        test_end = s.get("test_end")
        if not test_start or not test_end:
            continue
        dsub = df.filter((pl.col("Date") >= pl.lit(pl.Date(test_start))) & (pl.col("Date") <= pl.lit(pl.Date(test_end))))
        for fcol in factors:
            for h in label_cols:
                if fcol not in dsub.columns or h not in dsub.columns:
                    continue
                ric = rankic(dsub, fcol, h)
                hr = hitrate(dsub, fcol, h)
                rows.append({
                    "fold": fold,
                    "factor": fcol,
                    "horizon": int(h.split("_")[2][:-1]),
                    "rankic": ric,
                    "hitrate": hr,
                })

    if not rows:
        print("No fold metrics computed")
        return 1

    out = args.out or Path("reports") / "fold_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(out)
    print(f"Saved fold metrics: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

