#!/usr/bin/env python3
"""
Check that a generated dataset conforms to docs/ml/dataset.md feature contract.

Usage:
  python scripts/quality/check_dataset.py --input output/ml_dataset_latest_full.parquet

Exits non-zero if required columns are missing. Prints a short report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def required_columns() -> dict[str, set[str]]:
    mkt = {
        "mkt_ret_1d",
        "mkt_ret_5d",
        "mkt_ret_10d",
        "mkt_ret_20d",
        "mkt_ema_5",
        "mkt_ema_20",
        "mkt_ema_60",
        "mkt_ema_200",
        "mkt_dev_20",
        "mkt_gap_5_20",
        "mkt_ema20_slope_3",
        "mkt_vol_20d",
        "mkt_atr_14",
        "mkt_natr_14",
        "mkt_bb_pct_b",
        "mkt_bb_bw",
        "mkt_dd_from_peak",
        "mkt_big_move_flag",
        "mkt_ret_1d_z",
        "mkt_vol_20d_z",
        "mkt_bb_bw_z",
        "mkt_dd_from_peak_z",
        "mkt_bull_200",
        "mkt_trend_up",
        "mkt_high_vol",
        "mkt_squeeze",
    }
    cross = {
        "beta_60d",
        "alpha_1d",
        "alpha_5d",
        "rel_strength_5d",
        "trend_align_mkt",
        "alpha_vs_regime",
        "idio_vol_ratio",
    }
    flow = {
        "flow_foreign_net_ratio",
        "flow_individual_net_ratio",
        "flow_activity_ratio",
        "foreign_share_activity",
        "flow_foreign_net_z",
        "flow_individual_net_z",
        "flow_activity_z",
        "flow_smart_idx",
        "flow_smart_mom4",
        "flow_shock_flag",
        "flow_impulse",
        "flow_days_since",
    }
    stmt = {
        "stmt_yoy_sales",
        "stmt_yoy_op",
        "stmt_yoy_np",
        "stmt_opm",
        "stmt_npm",
        "stmt_progress_op",
        "stmt_progress_np",
        "stmt_rev_fore_op",
        "stmt_rev_fore_np",
        "stmt_rev_fore_eps",
        "stmt_rev_div_fore",
        "stmt_roe",
        "stmt_roa",
        "stmt_change_in_est",
        "stmt_nc_flag",
        "stmt_imp_statement",
        "stmt_days_since_statement",
    }
    targets = {
        "target_1d",
        "target_5d",
        "target_10d",
        "target_20d",
        "target_1d_binary",
        "target_5d_binary",
        "target_10d_binary",
    }
    core = {"Code", "Date", "Open", "High", "Low", "Close", "Volume"}
    return {
        "core": core,
        "mkt": mkt,
        "cross": cross,
        "flow": flow,
        "stmt": stmt,
        "targets": targets,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="dataset.md compliance checker")
    ap.add_argument(
        "--input", type=Path, default=Path("output/ml_dataset_latest_full.parquet")
    )
    args = ap.parse_args()

    if not args.input.exists():
        print(f"Input parquet not found: {args.input}")
        return 2

    df = pl.read_parquet(args.input)
    cols = set(df.columns)
    req = required_columns()

    missing: dict[str, list[str]] = {}
    for k, vs in req.items():
        miss = sorted(vs - cols)
        if miss:
            missing[k] = miss

    if not missing:
        print("✅ dataset.md compliance: OK")
        print(f"  Rows: {len(df):,}, Cols: {len(df.columns)}")
        return 0
    else:
        print("❌ dataset.md compliance: MISSING COLUMNS")
        for k, miss in missing.items():
            print(f"  - {k}: {len(miss)} missing -> {miss}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
