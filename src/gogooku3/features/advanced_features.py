from __future__ import annotations

"""
Advanced feature engineering (Phase 1): high-ROI, leak-safe, T+0 features.

Includes:
- Interaction features: RSI×Volatility, Momentum×Volume confirmation
- Change/acceleration: MACD histogram slope, Volume acceleration
- Cross-sectional: daily return percentile, daily volume z-score
- Calendar flags: weekday one-hot, month-start/end flags (5 trading days window)

All computations are per-Code for time-series ops, and per-Date for
cross-sectional ops. No forward-filling; NULLs are preserved.
"""

import math
import os

import polars as pl

from src.gogooku3.features.tech_indicators import compute_rsi_polars, compute_macd_polars

EPS = 1e-12


def _ensure_ma(df: pl.DataFrame, col: str, window: int, out: str) -> pl.DataFrame:
    if out in df.columns:
        return df
    if col not in df.columns:
        return df
    return df.with_columns(
        pl.col(col).rolling_mean(window).over("Code").alias(out)
    )


def _compute_rsi14(df: pl.DataFrame) -> pl.DataFrame:
    """Compute RSI(14) using optimized Polars implementation."""
    if "rsi_14" in df.columns:
        return df
    # Try Close first, then adjustmentclose
    price_col = "Close" if "Close" in df.columns else "adjustmentclose"
    if price_col not in df.columns:
        return df
    # Use optimized Polars implementation (30-40% faster)
    return compute_rsi_polars(df, column=price_col, period=14, group_col="Code", output_name="rsi_14")


def _compute_realized_vol_20(df: pl.DataFrame) -> pl.DataFrame:
    if "realized_vol_20" in df.columns:
        return df
    # Use returns_1d if available; else approximate from Close
    if "returns_1d" in df.columns:
        vol = pl.col("returns_1d").rolling_std(20).over("Code") * math.sqrt(252.0)
        return df.with_columns([vol.alias("realized_vol_20")])
    if "Close" in df.columns:
        ret = (pl.col("Close") / pl.col("Close").shift(1).over("Code") - 1.0)
        vol = ret.rolling_std(20).over("Code") * math.sqrt(252.0)
        return df.with_columns([vol.alias("realized_vol_20")])
    return df


def _compute_amihud_20(df: pl.DataFrame) -> pl.DataFrame:
    if "amihud_20" in df.columns:
        return df
    required = {"returns_1d", "dollar_volume"}
    if not required.issubset(df.columns):
        return df
    illiq = (
        (pl.col("returns_1d").abs() / (pl.col("dollar_volume") + EPS))
        .rolling_mean(20, min_periods=5)
        .over("Code")
    ).alias("amihud_20")
    return df.with_columns([illiq])


def _compute_macd_hist_slope(df: pl.DataFrame) -> pl.DataFrame:
    """Compute MACD histogram slope using optimized Polars implementation."""
    if "macd_hist_slope" in df.columns:
        return df
    # Try Close first, then adjustmentclose
    price_col = "Close" if "Close" in df.columns else "adjustmentclose"
    if price_col not in df.columns:
        return df
    # Use optimized Polars implementation (30-40% faster)
    df = compute_macd_polars(df, column=price_col, fast=12, slow=26, signal=9, group_col="Code")
    # Compute histogram slope (difference of histogram)
    if "macd_histogram" in df.columns:
        df = df.with_columns([
            pl.col("macd_histogram").diff().over("Code").alias("macd_hist_slope")
        ])
    return df


def add_advanced_features(df: pl.DataFrame) -> pl.DataFrame:
    """Attach advanced features to the equity panel.

    Returns a new DataFrame with additional columns. Never forward-fills.
    """
    if df.is_empty():
        return df

    out = df

    # Precompute helpful MAs
    if "Volume" in out.columns:
        out = _ensure_ma(out, "Volume", 5, "volume_ma_5")
        out = _ensure_ma(out, "Volume", 20, "volume_ma_20")

    # RSI and realized vol
    out = _compute_rsi14(out)
    out = _compute_realized_vol_20(out)
    out = _compute_amihud_20(out)

    # Interaction features
    if {"rsi_14", "realized_vol_20"}.issubset(out.columns):
        out = out.with_columns([
            (pl.col("rsi_14") * pl.col("realized_vol_20")).alias("rsi_vol_interact")
        ])
    if {"returns_5d", "Volume", "volume_ma_20"}.issubset(out.columns):
        out = out.with_columns([
            (pl.col("returns_5d") * (pl.col("Volume") / (pl.col("volume_ma_20") + EPS))).alias(
                "vol_confirmed_mom"
            )
        ])

    # Change/acceleration features
    out = _compute_macd_hist_slope(out)
    if {"volume_ma_5", "volume_ma_20"}.issubset(out.columns):
        out = out.with_columns([
            (pl.col("volume_ma_5") / (pl.col("volume_ma_20") + EPS) - 1.0).alias("volume_accel")
        ])

    # Cross-sectional features (per Date)
    # CRITICAL FIX (2025-10-24): Create T-1 lagged returns to prevent data leakage
    # See: reports/critical_issue_20251024.md, patches/fix_leakage_lag_injection.md
    if "returns_1d" in out.columns:
        out = out.with_columns([
            pl.col("returns_1d").shift(1).over("Code").alias("lag_returns_1d")
        ])

    _use_gpu_etl = os.getenv("USE_GPU_ETL", "0") == "1"
    if _use_gpu_etl:
        try:
            from src.utils.gpu_etl import cs_rank_and_z  # type: ignore

            out = cs_rank_and_z(
                out,
                rank_col="lag_returns_1d",  # FIXED: Use T-1 lagged returns
                z_col="Volume",
                group_keys=("Date",),
                out_rank_name="rank_ret_prev_1d",  # FIXED: Renamed to indicate T-1
                out_z_name="volume_cs_z",
            )
        except Exception:
            _use_gpu_etl = False  # fall back to CPU path below

    if not _use_gpu_etl:
        if "lag_returns_1d" in out.columns:  # FIXED: Check for lagged column
            cnt = pl.count().over("Date")
            rk = pl.col("lag_returns_1d").rank(method="average").over("Date")  # FIXED: Use T-1
            out = out.with_columns([
                pl.when(cnt > 1).then((rk - 1.0) / (cnt - 1.0)).otherwise(0.5).alias("rank_ret_prev_1d")  # FIXED: Renamed
            ])
            # Streaks and momentum persistence
            # up_streak_1d/down_streak_1d: consecutive days of positive/negative returns
            sign = (pl.col("returns_1d") > 0).cast(pl.Int8).alias("_pos")
            out = out.with_columns([sign.over("Code")])
            # Compute streaks via cumulative groups of equal sign
            grp = (pl.col("_pos") != pl.col("_pos").shift(1).over("Code")).cast(pl.Int8)
            out = out.with_columns([
                grp.over("Code").alias("_chg")
            ])
            out = out.with_columns([
                pl.col("_chg").cumsum().over("Code").alias("_grp_id")
            ])
            # Within each group, assign sequential count; positive groups → up_streak, negative → down_streak
            out = out.with_columns([
                pl.arange(1, pl.len() + 1).over(["Code", "_grp_id"]).alias("_seq")
            ])
            out = out.with_columns([
                pl.when(pl.col("_pos") == 1).then(pl.col("_seq")).otherwise(0).alias("up_streak_1d"),
                pl.when(pl.col("_pos") == 0).then(pl.col("_seq")).otherwise(0).alias("down_streak_1d"),
            ])
            # Momentum persistence: share of positive days last 5
            out = out.with_columns([
                pl.col("_pos").rolling_mean(5).over("Code").alias("mom_persist_5d")
            ])
            # Cleanup temps
            out = out.drop([c for c in ("_pos", "_chg", "_grp_id", "_seq") if c in out.columns])
        if "Volume" in out.columns:
            out = out.with_columns([
                ((pl.col("Volume") - pl.col("Volume").mean().over("Date")) / (pl.col("Volume").std().over("Date") + EPS)).alias(
                    "volume_cs_z"
                )
            ])

    # Calendar features (shared across codes per Date)
    if "Date" in out.columns:
        cal = out.select("Date").unique().sort("Date")
        cal = cal.with_columns([
            pl.col("Date").dt.weekday().alias("dow"),
            (pl.col("Date").dt.year()).alias("yy"),
            (pl.col("Date").dt.month()).alias("mm"),
        ])
        # Position within month
        cal = cal.with_columns([
            pl.col("Date").cum_count().over(["yy", "mm"]).alias("pos"),
            pl.count().over(["yy", "mm"]).alias("n_in_month"),
        ])
        cal = cal.with_columns([
            (pl.col("pos") <= 5).cast(pl.Int8).alias("month_start_flag"),
            ((pl.col("n_in_month") - pl.col("pos") + 1) <= 5).cast(pl.Int8).alias("month_end_flag"),
            (pl.col("dow") == 0).cast(pl.Int8).alias("is_mon"),
            (pl.col("dow") == 1).cast(pl.Int8).alias("is_tue"),
            (pl.col("dow") == 2).cast(pl.Int8).alias("is_wed"),
            (pl.col("dow") == 3).cast(pl.Int8).alias("is_thu"),
            (pl.col("dow") == 4).cast(pl.Int8).alias("is_fri"),
        ]).drop(["yy", "mm", "pos", "n_in_month", "dow"])
        out = out.join(cal, on="Date", how="left")

    # Validity flag (1 when inputs present for core interactions)
    conds = []
    if {"rsi_14", "realized_vol_20"}.issubset(out.columns):
        conds.append(pl.col("rsi_14").is_not_null() & pl.col("realized_vol_20").is_not_null())
    if {"returns_5d", "Volume", "volume_ma_20"}.issubset(out.columns):
        conds.append(pl.col("returns_5d").is_not_null() & pl.col("Volume").is_not_null() & pl.col("volume_ma_20").is_not_null())
    if conds:
        valid = conds[0]
        for c in conds[1:]:
            valid = valid | c
        out = out.with_columns([valid.cast(pl.Int8).alias("is_adv_valid")])
    else:
        out = out.with_columns([pl.lit(0).cast(pl.Int8).alias("is_adv_valid")])

    return out
