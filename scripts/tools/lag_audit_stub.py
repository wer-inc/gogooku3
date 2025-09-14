#!/usr/bin/env python3
from __future__ import annotations

"""
Lag audit stub: summarize disclosure→effective and effective→use lags where available.

Usage examples:
  # Sector short features parquet (has Date, effective_date)
  python scripts/tools/lag_audit_stub.py output/sector_short_selling_*.parquet

  # Any parquet with PublishedDate/AnnouncementDate
  python scripts/tools/lag_audit_stub.py output/event_raw_statements_*.parquet
"""

import glob
import sys
from pathlib import Path

import polars as pl


def audit_one(path: Path) -> None:
    df = pl.read_parquet(path)
    print(f"=== {path.name} ===")
    cols = set(df.columns)
    # Case A: Date + effective_date
    if {"Date", "effective_date"}.issubset(cols):
        d = df.select([
            pl.col("Date").cast(pl.Date).alias("Date"),
            pl.col("effective_date").cast(pl.Date).alias("effective_date"),
        ]).drop_nulls()
        if d.height:
            lag = (d["effective_date"] - d["Date"]).dt.days()
            print(
                "Date→effective lags (days):",
                {
                    "min": int(lag.min()),
                    "p50": float(lag.quantile(0.5)),
                    "p95": float(lag.quantile(0.95)),
                    "max": int(lag.max()),
                    "n": d.height,
                },
            )
    # Case B: PublishedDate/AnnouncementDate to effective_date
    pub_cols = [c for c in ("PublishedDate", "AnnouncementDate") if c in cols]
    if pub_cols and "effective_date" in cols:
        pub = pub_cols[0]
        d = df.select([
            pl.col(pub).cast(pl.Date).alias("pub"),
            pl.col("effective_date").cast(pl.Date).alias("effective_date"),
        ]).drop_nulls()
        if d.height:
            lag = (d["effective_date"] - d["pub"]).dt.days()
            print(
                f"{pub}→effective lags (days):",
                {
                    "min": int(lag.min()),
                    "p50": float(lag.quantile(0.5)),
                    "p95": float(lag.quantile(0.95)),
                    "max": int(lag.max()),
                    "n": d.height,
                },
            )


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/tools/lag_audit_stub.py <parquet_or_glob>")
        return 1
    pattern = sys.argv[1]
    files = sorted([Path(p) for p in glob.glob(pattern)]) if any(ch in pattern for ch in "*?[]") else [Path(pattern)]
    if not files:
        print(f"No files matched: {pattern}")
        return 1
    for f in files:
        if f.exists():
            try:
                audit_one(f)
            except Exception as e:
                print(f"ERROR auditing {f}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

