#!/usr/bin/env python3
from __future__ import annotations

"""
Baseline snapshot: quick dataset summary and key coverage metrics.

Usage:
  python scripts/tools/baseline_snapshot.py output/ml_dataset_latest_full.parquet
"""

import sys
from pathlib import Path

import polars as pl


def snapshot(path: Path) -> None:
    df = pl.read_parquet(path)
    n_rows = df.height
    n_cols = len(df.columns)
    codes = df.select("Code").unique().height if "Code" in df.columns else None
    dmin = df.select(pl.col("Date").min()).item() if "Date" in df.columns else None
    dmax = df.select(pl.col("Date").max()).item() if "Date" in df.columns else None

    print("=== DATASET SNAPSHOT ===")
    print(f"path: {path}")
    print(f"rows: {n_rows:,}, cols: {n_cols:,}")
    print(f"codes: {codes}")
    print(f"date range: {dmin} → {dmax}")

    # Coverage for common columns
    focus_cols = [
        "returns_1d",
        "returns_5d",
        "ss_sec33_short_share",
        "opt_iv_cmat_30d",
        "opt_term_slope_30_60",
        # Phase 1 features
        "rsi_vol_interact",
        "macd_hist_slope",
        "rank_ret_1d",
        # Phase 2 features
        "ret_1d_vs_sec",
        "ret_1d_rank_in_sec",
        # Phase 3 features
        "graph_degree",
        "graph_degree_z",
        "graph_clustering",
        "graph_core",
        "graph_pagerank",
    ]
    present = [c for c in focus_cols if c in df.columns]
    if present:
        cov = (
            df.select([(pl.col(c).is_not_null().mean() * 100).alias(c) for c in present])
            .to_pandas()
            .to_dict(orient="records")[0]
        )
        print("coverage (% non-null):", {k: f"{v:.1f}%" for k, v in cov.items()})

    # Returns distribution (quick sanity)
    if "returns_1d" in df.columns:
        q = df.select(
            [
                pl.col("returns_1d").quantile(0.01).alias("q01"),
                pl.col("returns_1d").median().alias("q50"),
                pl.col("returns_1d").quantile(0.99).alias("q99"),
            ]
        )
        print("returns_1d quantiles:", q.to_dict(as_series=False))


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/tools/baseline_snapshot.py <dataset.parquet>")
        return 1
    p = Path(sys.argv[1])
    if not p.exists():
        print(f"ERROR: file not found: {p}")
        return 1
    snapshot(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
