from __future__ import annotations

"""
Sector aggregation features (セクター集約特徴).

Implements dataset_new.md v1.1 Section 6.2-6.3:
- Sector equal-weighted returns (median-based, robust)
- Sector momentum, EMA, gaps
- Sector volatility and Z-scores
- Individual relative to sector
- Beta to sector

Requirements:
- sector33_id column (or sector33_code)
- Date column for time-series operations
"""

import polars as pl
from typing import Optional

EPS = 1e-12


def _find_sector_col(df: pl.DataFrame) -> Optional[str]:
    """Find the sector column to use for grouping."""
    candidates = ["sector33_id", "sector33_code", "sec33", "Sector33Code"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def add_sector_aggregation_features(
    df: pl.DataFrame,
    *,
    sector_col: Optional[str] = None,
    min_members: int = 3,
) -> pl.DataFrame:
    """
    Add sector aggregation features (~30 columns).

    Features added:
    - sec_ret_1d_eq, sec_ret_5d_eq, sec_ret_20d_eq: Sector equal-weighted returns (median)
    - sec_mom_20: Sector 20-day momentum
    - sec_ema_5, sec_ema_20: Sector exponential moving averages
    - sec_gap_5_20: Sector EMA gap
    - sec_vol_20, sec_vol_20_z: Sector volatility and Z-score
    - sec_member_cnt, sec_small_flag: Meta information
    - rel_to_sec_5d: Individual return relative to sector
    - beta_to_sec_60: Beta to sector (60-day window)
    - alpha_vs_sec_1d: Alpha vs sector
    - z_in_sec_returns_5d, z_in_sec_ma_gap_5_20: Z-scores within sector

    Args:
        df: Input DataFrame with price/return data
        sector_col: Sector column name (auto-detected if None)
        min_members: Minimum sector members to be valid (default: 3)

    Returns:
        DataFrame with added sector aggregation features
    """
    if df.is_empty():
        return df

    sec = sector_col or _find_sector_col(df)
    if not sec or sec not in df.columns:
        # No sector column found
        return df

    # Ensure sorted by sector and date for time-series operations
    df = df.sort([sec, "Date"])

    # =========================================================================
    # 1. Sector equal-weighted returns (median-based for robustness)
    # =========================================================================
    cross_keys = ["Date", sec]

    sector_returns = []
    for ret_col in ["returns_1d", "returns_5d", "returns_20d"]:
        if ret_col in df.columns:
            alias = ret_col.replace("returns_", "sec_ret_") + "_eq"
            sector_returns.append(
                pl.col(ret_col).median().over(cross_keys).alias(alias)
            )

    if sector_returns:
        df = df.with_columns(sector_returns)

    # =========================================================================
    # 2. Sector time-series features (momentum, EMA, gaps)
    # =========================================================================

    # Sector 20-day momentum (cumulative returns)
    if "sec_ret_1d_eq" in df.columns:
        df = df.with_columns([
            pl.col("sec_ret_1d_eq")
              .rolling_sum(window_size=20, min_periods=20)
              .over(sec)
              .alias("sec_mom_20"),
        ])

    # Sector EMAs (5-day and 20-day)
    if "sec_ret_1d_eq" in df.columns:
        df = df.with_columns([
            pl.col("sec_ret_1d_eq")
              .ewm_mean(span=5, min_periods=5)
              .over(sec)
              .alias("sec_ema_5"),
            pl.col("sec_ret_1d_eq")
              .ewm_mean(span=20, min_periods=20)
              .over(sec)
              .alias("sec_ema_20"),
        ])

    # Sector EMA gap (5-20)
    if "sec_ema_5" in df.columns and "sec_ema_20" in df.columns:
        df = df.with_columns([
            ((pl.col("sec_ema_5") - pl.col("sec_ema_20")) / (pl.col("sec_ema_20").abs() + EPS))
              .alias("sec_gap_5_20"),
        ])

    # =========================================================================
    # 3. Sector volatility
    # =========================================================================
    if "sec_ret_1d_eq" in df.columns:
        df = df.with_columns([
            pl.col("sec_ret_1d_eq")
              .rolling_std(window_size=20, min_periods=20)
              .over(sec)
              .mul(252 ** 0.5)  # Annualize
              .alias("sec_vol_20"),
        ])

    # Sector volatility Z-score (time-series Z-score over 252 days)
    if "sec_vol_20" in df.columns:
        df = df.with_columns([
            pl.col("sec_vol_20")
              .rolling_mean(window_size=252, min_periods=252)
              .over(sec)
              .alias("_sec_vol_20_mean_252"),
            pl.col("sec_vol_20")
              .rolling_std(window_size=252, min_periods=252)
              .over(sec)
              .alias("_sec_vol_20_std_252"),
        ])
        df = df.with_columns([
            ((pl.col("sec_vol_20") - pl.col("_sec_vol_20_mean_252")) /
             (pl.col("_sec_vol_20_std_252") + EPS))
              .alias("sec_vol_20_z"),
        ])

    # =========================================================================
    # 4. Sector meta information
    # =========================================================================
    df = df.with_columns([
        pl.count().over(cross_keys).alias("sec_member_cnt"),
    ])

    df = df.with_columns([
        (pl.col("sec_member_cnt") < min_members).cast(pl.Int8).alias("sec_small_flag"),
    ])

    # =========================================================================
    # 5. Individual relative to sector
    # =========================================================================

    # Simple deviation from sector
    if "returns_5d" in df.columns and "sec_ret_5d_eq" in df.columns:
        df = df.with_columns([
            (pl.col("returns_5d") - pl.col("sec_ret_5d_eq")).alias("rel_to_sec_5d"),
        ])

    # Beta to sector (60-day rolling covariance)
    # beta_to_sec_60 = Cov(returns_1d, sec_ret_1d_eq; 60) / Var(sec_ret_1d_eq; 60)
    if "returns_1d" in df.columns and "sec_ret_1d_eq" in df.columns:
        # Calculate rolling covariance and variance
        df = df.with_columns([
            # Mean of individual returns over 60 days
            pl.col("returns_1d")
              .rolling_mean(window_size=60, min_periods=60)
              .over("Code")
              .alias("_ret_1d_mean_60"),
            # Mean of sector returns over 60 days
            pl.col("sec_ret_1d_eq")
              .rolling_mean(window_size=60, min_periods=60)
              .over("Code")
              .alias("_sec_ret_1d_mean_60"),
        ])

        # Covariance: E[(X - E[X])(Y - E[Y])]
        df = df.with_columns([
            ((pl.col("returns_1d") - pl.col("_ret_1d_mean_60")) *
             (pl.col("sec_ret_1d_eq") - pl.col("_sec_ret_1d_mean_60")))
              .rolling_mean(window_size=60, min_periods=60)
              .over("Code")
              .alias("_cov_ret_sec_60"),
            # Variance of sector returns
            (pl.col("sec_ret_1d_eq") - pl.col("_sec_ret_1d_mean_60")).pow(2)
              .rolling_mean(window_size=60, min_periods=60)
              .over("Code")
              .alias("_var_sec_60"),
        ])

        # Beta = Cov / Var
        df = df.with_columns([
            (pl.col("_cov_ret_sec_60") / (pl.col("_var_sec_60") + EPS))
              .alias("beta_to_sec_60"),
        ])

    # Alpha vs sector
    if "returns_1d" in df.columns and "beta_to_sec_60" in df.columns and "sec_ret_1d_eq" in df.columns:
        df = df.with_columns([
            (pl.col("returns_1d") - pl.col("beta_to_sec_60") * pl.col("sec_ret_1d_eq"))
              .alias("alpha_vs_sec_1d"),
        ])

    # =========================================================================
    # 6. Z-scores within sector (cross-sectional on each date)
    # =========================================================================

    # Z-score of returns_5d within sector
    if "returns_5d" in df.columns:
        df = df.with_columns([
            pl.col("returns_5d").mean().over(cross_keys).alias("_sec_mean_ret5d_cs"),
            pl.col("returns_5d").std().over(cross_keys).alias("_sec_std_ret5d_cs"),
        ])
        df = df.with_columns([
            ((pl.col("returns_5d") - pl.col("_sec_mean_ret5d_cs")) /
             (pl.col("_sec_std_ret5d_cs") + EPS))
              .alias("z_in_sec_returns_5d"),
        ])

    # Z-score of ma_gap_5_20 within sector
    if "ma_gap_5_20" in df.columns:
        df = df.with_columns([
            pl.col("ma_gap_5_20").mean().over(cross_keys).alias("_sec_mean_ma_gap_cs"),
            pl.col("ma_gap_5_20").std().over(cross_keys).alias("_sec_std_ma_gap_cs"),
        ])
        df = df.with_columns([
            ((pl.col("ma_gap_5_20") - pl.col("_sec_mean_ma_gap_cs")) /
             (pl.col("_sec_std_ma_gap_cs") + EPS))
              .alias("z_in_sec_ma_gap_5_20"),
        ])

    # Clean up temporary columns
    temp_cols = [c for c in df.columns if c.startswith("_sec_") or c.startswith("_ret_") or
                 c.startswith("_cov_") or c.startswith("_var_")]
    if temp_cols:
        df = df.drop(temp_cols)

    return df
