#!/usr/bin/env python3
"""
Validate an ML dataset against docs/ml/dataset.md (canonical spec).

- Verifies presence of canonical columns
- Verifies cross features match exactly the spec's 8 columns
- Verifies group counts for mkt_*, stmt_*, flow_* meet minimums
  (exact counts can vary by window coverage; mkt_ is expected >= 26, stmt_ >= 17, flow_ >= 17)

Usage:
  python scripts/tools/validate_dataset_against_spec.py \
    --dataset output/ml_dataset_YYYYMMDD_HHMMSS.parquet \
    --docs docs/ml/dataset.md
Exit code is non-zero if validation fails.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import polars as pl

CANONICAL_CROSS = [
    "beta_60d",
    "alpha_1d",
    "alpha_5d",
    "rel_strength_5d",
    "trend_align_mkt",
    "alpha_vs_regime",
    "idio_vol_ratio",
    "beta_stability_60d",
]

REQUIRED_CORE = [
    "Code",
    "LocalCode",
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Section",
    "section_norm",
]


def extract_backticked_names(md_text: str) -> set[str]:
    return set(re.findall(r"`([A-Za-z0-9_]+)`", md_text))


def main():
    ap = argparse.ArgumentParser(description="Validate dataset against dataset.md")
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--docs", type=Path, default=Path("docs/ml/dataset.md"))
    args = ap.parse_args()

    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}")
        sys.exit(2)
    if not args.docs.exists():
        print(f"Docs not found: {args.docs}")
        sys.exit(2)

    df = pl.read_parquet(args.dataset)
    cols = df.columns

    spec_text = args.docs.read_text(encoding="utf-8", errors="ignore")
    md_names = extract_backticked_names(spec_text)

    # Checks
    errors: list[str] = []

    # 1) Required core columns
    for c in REQUIRED_CORE:
        if c not in cols:
            errors.append(f"Missing required core column: {c}")

    # 2) Cross features exactly 8
    missing_cross = [c for c in CANONICAL_CROSS if c not in cols]
    if missing_cross:
        errors.append(f"Missing cross features: {missing_cross}")
    # Ensure no helper betas shipped
    helpers = [c for c in ["beta_60d_raw", "beta_20d_raw", "beta_rolling"] if c in cols]
    if helpers:
        errors.append(f"Helper beta columns should not be present: {helpers}")

    # 3) Groups by prefix
    mkt_count = len([c for c in cols if c.startswith("mkt_")])
    if mkt_count < 26:
        errors.append(f"mkt_* features too few: {mkt_count} < 26")

    stmt_count = len([c for c in cols if c.startswith("stmt_")])
    if stmt_count < 17:
        errors.append(f"stmt_* features too few: {stmt_count} < 17")

    flow_count = len([c for c in cols if c.startswith("flow_")])
    # dataset.md lists 17 flow features target; allow >= 12 to pass if partial, else enforce 17
    if flow_count < 17:
        errors.append(f"flow_* features too few: {flow_count} < 17")

    # 4) Validity flag naming must be normalized
    if "is_ema_5_valid" in cols:
        errors.append(
            "Found legacy flag name: is_ema_5_valid (should be is_ema5_valid)"
        )

    # 5) idio_vol_ratio presence
    if "idio_vol_ratio" not in cols:
        errors.append("Missing idio_vol_ratio")

    # 6) Extra names mentioned in dataset.md but absent (best effort)
    # Filter to those likely to be columns
    md_col_like = [
        n
        for n in md_names
        if any(
            n.startswith(p)
            for p in (
                "mkt_",
                "stmt_",
                "flow_",
                "returns_",
                "log_returns_",
                "volatility_",
                "ema_",
                "sma_",
                "ma_gap_",
                "price_to_",
                "high_low_ratio",
                "close_to_high",
                "close_to_low",
                "rsi_",
                "macd",
                "bb_",
                "atr_",
                "adx_",
                "stoch_",
                "target_",
            )
        )
        or n
        in REQUIRED_CORE
        + CANONICAL_CROSS
        + ["TurnoverValue", "row_idx", "shares_outstanding"]
    ]
    missing_from_md = [n for n in md_col_like if n not in cols]
    # Only warn for optional ones; not all docs tokens are mandatory in every run
    # We keep it informative.

    # Report
    ok = not errors
    print(f"Dataset: {args.dataset}")
    print(f"Columns: {len(cols)} | mkt={mkt_count} stmt={stmt_count} flow={flow_count}")
    if missing_from_md:
        print(
            f"Note: {len(missing_from_md)} doc-listed names missing (informative): {missing_from_md[:10]} ..."
        )
    if ok:
        print("✅ Validation PASSED against dataset.md")
        sys.exit(0)
    else:
        print("❌ Validation FAILED:")
        for e in errors:
            print(" -", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
