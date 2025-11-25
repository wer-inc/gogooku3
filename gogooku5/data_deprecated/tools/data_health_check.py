#!/usr/bin/env python3
"""
Lightweight data health checker for timeseries ML datasets.

Features:
- Same-day leakage hints: correlation between targets and all numeric features.
- Distribution drift: PSI (baseline vs current) for numeric columns.
- Homogeneity: KS two-sample test (if SciPy available).

Usage:
  PYTHONPATH=data/src python data/tools/data_health_check.py \
    --dataset data/output/chunks/2020Q1/ml_dataset.parquet \
    --baseline data/output/chunks/2020Q2/ml_dataset.parquet \
    --sample-rows 100000
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import polars as pl

try:
    from scipy.stats import ks_2samp
except Exception:  # pragma: no cover - optional dependency
    ks_2samp = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset health checks (correlation, PSI, KS)")
    parser.add_argument("--dataset", required=True, help="Path to current dataset parquet")
    parser.add_argument("--baseline", help="Optional baseline dataset parquet for PSI/KS")
    parser.add_argument(
        "--target-cols",
        default="target_1d,target_5d,target_10d,target_20d",
        help="Comma-separated target columns for correlation leak check",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.30,
        help="Absolute correlation threshold to flag potential leakage",
    )
    parser.add_argument(
        "--psi-threshold",
        type=float,
        default=0.25,
        help="PSI threshold to flag distribution drift",
    )
    parser.add_argument(
        "--ks-threshold",
        type=float,
        default=0.01,
        help="p-value threshold for KS test to flag distribution shift",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=100_000,
        help="Max rows to sample from each dataset (to keep memory bounded)",
    )
    parser.add_argument(
        "--date-col",
        default="Date",
        help="Date column name (excluded from numeric feature list)",
    )
    parser.add_argument(
        "--code-col",
        default="Code",
        help="Code column name (excluded from numeric feature list)",
    )
    return parser.parse_args()


def load_numeric(path: Path, *, sample_rows: int, exclude: Iterable[str]) -> pl.DataFrame:
    """Load numeric columns with an optional row cap to stay memory-friendly."""

    scan = pl.scan_parquet(path)
    # Drop non-numeric; bool is kept as Int8 for correlation/PSI.
    df = (
        scan.select(pl.exclude([pl.Utf8, pl.Categorical, pl.Binary]))
        .drop(list(exclude))
        .limit(sample_rows)
        .collect()
        .with_columns([pl.col(pl.Boolean).cast(pl.Int8)])
    )
    return df


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index with quantile binning on expected."""

    # Guard against empty arrays
    if expected.size == 0 or actual.size == 0:
        return math.nan

    qs = np.quantile(expected, np.linspace(0, 1, bins + 1))
    e_hist, _ = np.histogram(expected, qs)
    a_hist, _ = np.histogram(actual, qs)
    e = np.clip(e_hist / len(expected), 1e-6, 1)
    a = np.clip(a_hist / len(actual), 1e-6, 1)
    return float(np.sum((a - e) * np.log(a / e)))


def correlation_flags(df: pl.DataFrame, targets: List[str], threshold: float) -> Dict[str, float]:
    """Return {feature: |corr|} for correlations over threshold."""

    if df.is_empty():
        return {}

    # Convert to pandas for quick corrwith
    pdf = df.to_pandas()
    flagged: Dict[str, float] = {}
    for tgt in targets:
        if tgt not in pdf.columns:
            continue
        corrs = pdf.drop(columns=[c for c in targets if c != tgt], errors="ignore").corrwith(pdf[tgt])
        for feat, corr_val in corrs.items():
            if feat == tgt or np.isnan(corr_val):
                continue
            if abs(corr_val) >= threshold:
                flagged[f"{feat} (vs {tgt})"] = float(abs(corr_val))
    return dict(sorted(flagged.items(), key=lambda kv: kv[1], reverse=True))


def psi_flags(
    baseline: pl.DataFrame,
    current: pl.DataFrame,
    threshold: float,
) -> Dict[str, float]:
    """Return {feature: psi} where PSI exceeds threshold."""

    flags: Dict[str, float] = {}
    common = [c for c in baseline.columns if c in current.columns]
    for col in common:
        if baseline[col].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8):
            base_vals = baseline[col].drop_nans().drop_nulls().to_numpy()
            curr_vals = current[col].drop_nans().drop_nulls().to_numpy()
            if len(base_vals) == 0 or len(curr_vals) == 0:
                continue
            val = psi(base_vals, curr_vals)
            if not math.isnan(val) and val >= threshold:
                flags[col] = val
    return dict(sorted(flags.items(), key=lambda kv: kv[1], reverse=True))


def ks_flags(
    baseline: pl.DataFrame,
    current: pl.DataFrame,
    p_threshold: float,
) -> Dict[str, float]:
    """Return {feature: p_value} where KS p-value is below threshold."""

    if ks_2samp is None:
        return {}

    flags: Dict[str, float] = {}
    common = [c for c in baseline.columns if c in current.columns]
    for col in common:
        if baseline[col].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8):
            base_vals = baseline[col].drop_nans().drop_nulls().to_numpy()
            curr_vals = current[col].drop_nans().drop_nulls().to_numpy()
            if len(base_vals) == 0 or len(curr_vals) == 0:
                continue
            _, p_value = ks_2samp(base_vals, curr_vals)
            if p_value < p_threshold:
                flags[col] = float(p_value)
    return dict(sorted(flags.items(), key=lambda kv: kv[1]))


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset)
    baseline_path = Path(args.baseline) if args.baseline else None

    exclude_cols = {args.date_col, args.code_col}
    target_cols = [c.strip() for c in args.target_cols.split(",") if c.strip()]
    exclude_cols.update(target_cols)

    # Load current dataset
    current_df = load_numeric(dataset_path, sample_rows=args.sample_rows, exclude=exclude_cols)
    print(f"[LOAD] current: {dataset_path} rows={len(current_df)} cols={len(current_df.columns)}")

    # Correlation-based leakage hints
    corr_hits = correlation_flags(current_df, target_cols, args.corr_threshold)
    if corr_hits:
        print(f"\n[LEAK?] Correlations |rho| >= {args.corr_threshold}")
        for feat, corr_val in corr_hits.items():
            print(f"  {feat}: {corr_val:.3f}")
    else:
        print(f"\n[LEAK?] No correlations above {args.corr_threshold}")

    if baseline_path:
        baseline_df = load_numeric(baseline_path, sample_rows=args.sample_rows, exclude=exclude_cols)
        print(f"\n[LOAD] baseline: {baseline_path} rows={len(baseline_df)} cols={len(baseline_df.columns)}")

        psi_hits = psi_flags(baseline_df, current_df, args.psi_threshold)
        if psi_hits:
            print(f"\n[PSI] Drift flags (>= {args.psi_threshold})")
            for col, val in psi_hits.items():
                print(f"  {col}: {val:.3f}")
        else:
            print(f"\n[PSI] No columns over {args.psi_threshold}")

        ks_hits = ks_flags(baseline_df, current_df, args.ks_threshold)
        if ks_hits:
            print(f"\n[KS] Distribution shift flags (p < {args.ks_threshold})")
            for col, pval in ks_hits.items():
                print(f"  {col}: p={pval:.3e}")
        else:
            msg = "SciPy not installed; KS skipped" if ks_2samp is None else f"No columns with p < {args.ks_threshold}"
            print(f"\n[KS] {msg}")
    else:
        print("\n[PSI/KS] Skipped (no baseline provided)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
