#!/usr/bin/env python3
from __future__ import annotations

"""
Lightweight dataset quality checks for leakage, nulls, and schema issues.

Usage:
  python scripts/tools/data_checks.py output/ml_dataset_latest_full.parquet

Exits with code 0 on success, 1 if any check fails.
"""

import sys
from pathlib import Path

import polars as pl


def validate_dataset(parquet_path: str, *, null_threshold: float = 0.3) -> bool:
    p = Path(parquet_path)
    if not p.exists():
        print(f"ERROR: file not found: {p}")
        return False

    df = pl.read_parquet(p)
    errors: list[str] = []

    # 1) Primary key uniqueness (Code, Date)
    if {"Code", "Date"}.issubset(df.columns):
        dupes = df.group_by(["Code", "Date"]).len().filter(pl.col("len") > 1)
        if dupes.height > 0:
            errors.append(f"Duplicate Code-Date pairs: {dupes.height}")
    else:
        errors.append("Missing columns for primary key: Code/Date")

    # 2) Missing value ratios
    if df.height > 0:
        null_counts = df.null_count().to_dict(as_series=False)
        high_nulls: dict[str, float] = {}
        for col in df.columns:
            # Handle case where null_counts[col] might be a list with single value
            null_val = null_counts.get(col, [0])
            if isinstance(null_val, list):
                null_val = null_val[0] if null_val else 0
            rate = float(null_val) / float(df.height)
            if rate > null_threshold:
                high_nulls[col] = rate
        if high_nulls:
            errors.append(f"High null rate columns (> {null_threshold:.0%}): {sorted(high_nulls.items())}")

    # 3) Return outliers (sanity)
    if "returns_1d" in df.columns:
        extreme = df.filter((pl.col("returns_1d") > 2.0) | (pl.col("returns_1d") < -0.9)).height
        if extreme > 0:
            errors.append(f"Extreme returns_1d outliers detected: {extreme} rows")

    # 4) Future-return label alignment (optional)
    if {"Code", "Date", "feat_ret_1d", "returns_1d"}.issubset(df.columns):
        # Sort for deterministic window behavior, then compute previous-date alignment
        _df = df.sort(["Code", "Date"])  # ensure shift(1) over Code reflects t-1 by date
        merged = (
            _df.select(["Code", "Date", "feat_ret_1d"]).join(
                _df.select(
                    [pl.col("Code"), pl.col("Date").shift(1).over("Code").alias("PrevDate"), pl.col("returns_1d")]
                ),
                left_on=["Code", "Date"],
                right_on=["Code", "PrevDate"],
                how="inner",
            )
        )
        mismatches = merged.filter((pl.col("feat_ret_1d") - pl.col("returns_1d")).abs() > 1e-8).height
        if mismatches > 0:
            errors.append(f"feat_ret_1d != next-day returns_1d mismatches: {mismatches}")

    # 5) Effective-date ordering checks (generic)
    if {"Date", "effective_date"}.issubset(df.columns):
        leaks = df.filter(pl.col("Date") >= pl.col("effective_date")).height
        if leaks > 0:
            errors.append(f"Potential leak: {leaks} rows where Date >= effective_date")

    if errors:
        print("Dataset quality checks: FAILED")
        for e in errors:
            print("ERROR:", e)
        return False
    else:
        print("Dataset quality checks: PASSED")
        return True


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/tools/data_checks.py <dataset.parquet>")
        return 1
    ok = validate_dataset(sys.argv[1])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
