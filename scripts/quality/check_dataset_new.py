#!/usr/bin/env python3
"""
Check that a generated dataset conforms to docs/ml/dataset_new.md (v1.1).

Usage:
  python scripts/quality/check_dataset_new.py --input output/ml_dataset_latest_full.parquet

Exits non-zero if required columns are missing. Prints brief coverage stats.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import polars as pl


def required_columns() -> dict[str, set[str]]:
    # Meta and identifiers
    meta = {
        "Code",
        "Date",
        "Section",
        "MarketCode",
        "row_idx",
        "shares_outstanding",
        # sector meta
        "sector17_code",
        "sector17_name",
        "sector17_id",
        "sector33_code",
        "sector33_name",
        "sector33_id",
    }

    # OHLCV core
    core = {"Open", "High", "Low", "Close", "Volume", "TurnoverValue"}

    # Returns + volatility
    rets = {
        "returns_1d",
        "returns_5d",
        "returns_10d",
        "returns_20d",
        "returns_60d",
        "returns_120d",
        "log_returns_1d",
        "log_returns_5d",
        "log_returns_10d",
        "log_returns_20d",
    }
    vola = {
        "volatility_5d",
        "volatility_10d",
        "volatility_20d",
        "volatility_60d",
        "realized_volatility",
    }

    # MA/TA
    ta = {
        "sma_5",
        "sma_10",
        "sma_20",
        "sma_60",
        "sma_120",
        "ema_5",
        "ema_10",
        "ema_20",
        "ema_60",
        "ema_200",
        "price_to_sma5",
        "price_to_sma20",
        "price_to_sma60",
        "ma_gap_5_20",
        "ma_gap_20_60",
        "high_low_ratio",
        "close_to_high",
        "close_to_low",
        "rsi_2",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_histogram",
        "atr_14",
        "adx_14",
        "stoch_k",
        "bb_width",
        "bb_position",
        "turnover_rate",
        "dollar_volume",
    }

    # TOPIX market block
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

    # Cross features
    cross = {
        "beta_60d",
        "alpha_1d",
        "alpha_5d",
        "rel_strength_5d",
        "trend_align_mkt",
        "alpha_vs_regime",
        "idio_vol_ratio",
        "beta_stability_60d",
    }

    # Flow (weekly tradesspec)
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

    # Statements (PEAD)
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

    # Targets
    targets = {
        "target_1d",
        "target_5d",
        "target_10d",
        "target_20d",
        "target_1d_binary",
        "target_5d_binary",
        "target_10d_binary",
        "feat_ret_1d",
        "feat_ret_5d",
        "feat_ret_10d",
        "feat_ret_20d",
    }

    return {
        "meta": meta,
        "core": core,
        "rets": rets,
        "vola": vola,
        "ta": ta,
        "mkt": mkt,
        "cross": cross,
        "flow": flow,
        "stmt": stmt,
        "targets": targets,
    }


def print_coverage(df: pl.DataFrame, cols: list[str], title: str) -> None:
    present = [c for c in cols if c in df.columns]
    if not present:
        print(f"  {title}: 0 present")
        return
    sample = df.sample(min(200_000, len(df)), with_replacement=False) if len(df) > 200_000 else df
    nn = sample.select([pl.col(c).is_not_null().mean().alias(c) for c in present])
    ratios = [float(nn[c][0]) for c in present]
    mean_ratio = sum(ratios) / len(ratios)
    print(f"  {title}: {len(present)} present, avg non-null {mean_ratio:.1%}")


def main() -> int:
    ap = argparse.ArgumentParser(description="dataset_new.md compliance checker")
    ap.add_argument("--input", type=Path, default=Path("output/ml_dataset_latest_full.parquet"))
    args = ap.parse_args()

    if not args.input.exists():
        print(f"Input parquet not found: {args.input}")
        return 2

    df = pl.read_parquet(args.input)
    cols = set(df.columns)
    spec = required_columns()

    missing: dict[str, list[str]] = {}
    for k, vs in spec.items():
        miss = sorted(list(vs - cols))
        if miss:
            missing[k] = miss

    if missing:
        print("❌ dataset_new.md compliance: MISSING COLUMNS")
        for k, miss in missing.items():
            print(f"  - {k}: {len(miss)} missing → {miss}")
        # Still show quick coverage on present groups
        for k, vs in spec.items():
            print_coverage(df, list(vs & cols), title=k)
        return 1

    print("✅ dataset_new.md compliance: OK")
    print(f"  Rows: {len(df):,}, Cols: {len(df.columns)}")
    # Coverage snapshot
    for k, vs in spec.items():
        print_coverage(df, list(vs), title=k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

