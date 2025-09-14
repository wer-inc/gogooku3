from __future__ import annotations

"""
Short selling features → daily panel integration (leak-safe, Polars).

This module implements short selling data integration with:
- T+1 effective date computation from PublishedDate
- Short selling ratio, volume, and position features
- Extreme value detection and percentile features
- Liquidity-normalized features using ADV20
- As-of backward join to prevent future data leakage
- Statistical features: moving averages, Z-scores, momentum

Key features generated:
- ss_ratio: Short selling ratio (%)
- ss_volume_ratio: Short selling volume / total volume
- ss_balance_ratio: Short selling balance / shares outstanding
- ss_lending_ratio: Securities lending ratio
- ss_momentum_5d, ss_momentum_10d: Short selling momentum
- ss_extreme_flag: Extreme short selling activity flag
- ss_*_z52: 52-week Z-scores for mean reversion signals

Public API:
- build_short_effective(df_ss, next_business_day_fn)
- add_short_core_features(ss)
- add_short_statistical_features(ss)
- attach_adv20_and_scale(ss_feat, adv20)
- asof_attach_to_daily(quotes, ss_k)
- add_short_selling_block(quotes, short_df, positions_df, adv20_df, enable_z_scores=True)
"""

import datetime as _dt
from collections.abc import Callable

import polars as pl

EPS = 1e-12


def _next_business_day_jp(date_col: pl.Expr) -> pl.Expr:
    """Next business day for JP market (Mon-Fri); skips weekends.

    Weekday mapping: Monday=0 ... Sunday=6
    - Mon-Thu -> +1 day
    - Fri -> +3 days (next Monday)
    - Sat -> +2 days (next Monday)
    - Sun -> +1 day (next Monday)
    """
    wd = date_col.dt.weekday()
    return (
        pl.when(wd <= 3)
        .then(date_col + pl.duration(days=1))
        .when(wd == 4)
        .then(date_col + pl.duration(days=3))
        .when(wd == 5)
        .then(date_col + pl.duration(days=2))
        .otherwise(date_col + pl.duration(days=1))
    )


def build_short_effective(
    df_ss: pl.DataFrame,
    next_business_day_fn: Callable[[pl.Expr], pl.Expr] | None = None,
) -> pl.DataFrame:
    """Build short selling data with effective_start (T+1 rule from PublishedDate).

    Args:
        df_ss: Raw short selling data with PublishedDate, Date columns
        next_business_day_fn: Function to compute next business day

    Returns:
        DataFrame with effective_start column (T+1 availability rule)
    """
    if df_ss.is_empty():
        return df_ss

    nbd_fn = next_business_day_fn or _next_business_day_jp

    # Compute effective start (T+1 from PublishedDate)
    return df_ss.with_columns([
        nbd_fn(pl.col("PublishedDate")).alias("effective_start")
    ])


def add_short_core_features(ss: pl.DataFrame) -> pl.DataFrame:
    """Add core short selling features.

    Features:
    - ss_ratio: Short selling ratio (normalized %)
    - ss_volume_ratio: Short selling volume / total volume
    - ss_volume_abs: Absolute short selling volume
    - ss_activity: Short selling activity level
    """
    if ss.is_empty():
        return ss

    return ss.with_columns([
        # Short selling ratio (already percentage, normalize to 0-1)
        (pl.col("ShortSellingRatio") / 100.0).alias("ss_ratio"),

        # Volume-based features
        (pl.col("ShortSellingVolume") / (pl.col("TotalVolume") + EPS)).alias("ss_volume_ratio"),
        pl.col("ShortSellingVolume").alias("ss_volume_abs"),

        # Activity proxy (volume * ratio)
        (pl.col("ShortSellingVolume") * pl.col("ShortSellingRatio") / 100.0).alias("ss_activity"),
    ])


def add_short_positions_features(positions: pl.DataFrame) -> pl.DataFrame:
    """Add short selling positions features.

    Features:
    - ss_balance: Short selling balance (shares)
    - ss_balance_ratio: Balance as percentage
    - ss_balance_change: Daily change in balance
    - ss_lending_balance: Securities lending balance
    - ss_lending_ratio: Lending ratio
    """
    if positions.is_empty():
        return positions

    return positions.with_columns([
        # Balance features
        pl.col("ShortSellingBalance").alias("ss_balance"),
        (pl.col("ShortSellingBalanceRatio") / 100.0).alias("ss_balance_ratio"),
        pl.col("ShortSellingBalanceChange").alias("ss_balance_change"),

        # Lending features
        pl.col("LendingBalance").alias("ss_lending_balance"),
        (pl.col("LendingBalanceRatio") / 100.0).alias("ss_lending_ratio"),

        # Net short pressure (balance relative to lending)
        ((pl.col("ShortSellingBalance") / (pl.col("LendingBalance") + EPS)) - 1.0).alias("ss_net_pressure"),
    ])


def add_short_statistical_features(ss: pl.DataFrame) -> pl.DataFrame:
    """Add statistical features: moving averages, momentum, volatility.

    Features:
    - ss_ratio_ma5, ss_ratio_ma20: Moving averages of short ratio
    - ss_momentum_5d, ss_momentum_10d: Short selling momentum
    - ss_vol_5d, ss_vol_20d: Short selling volatility
    - ss_trend_strength: Trend strength indicator
    """
    if ss.is_empty():
        return ss

    return ss.with_columns([
        # Moving averages
        pl.col("ss_ratio").rolling_mean(5).over("Code").alias("ss_ratio_ma5"),
        pl.col("ss_ratio").rolling_mean(20).over("Code").alias("ss_ratio_ma20"),

        # Momentum (change from N days ago)
        (pl.col("ss_ratio") - pl.col("ss_ratio").shift(5).over("Code")).alias("ss_momentum_5d"),
        (pl.col("ss_ratio") - pl.col("ss_ratio").shift(10).over("Code")).alias("ss_momentum_10d"),

        # Volatility (rolling std)
        pl.col("ss_ratio").rolling_std(5).over("Code").alias("ss_vol_5d"),
        pl.col("ss_ratio").rolling_std(20).over("Code").alias("ss_vol_20d"),

        # Volume momentum
        (pl.col("ss_volume_ratio") - pl.col("ss_volume_ratio").shift(5).over("Code")).alias("ss_volume_momentum_5d"),

        # Trend strength (current vs MA20)
        ((pl.col("ss_ratio") - pl.col("ss_ratio_ma20")) / (pl.col("ss_ratio_ma20") + EPS)).alias("ss_trend_strength"),
    ])


def add_short_extreme_detection(ss: pl.DataFrame, percentile_window: int = 252) -> pl.DataFrame:
    """Add extreme short selling detection features.

    Features:
    - ss_percentile_252d: Percentile rank over 252 days
    - ss_extreme_high: High short selling activity flag (>95th percentile)
    - ss_extreme_low: Low short selling activity flag (<5th percentile)
    - ss_regime_shift: Regime shift detection
    """
    if ss.is_empty():
        return ss

    return ss.with_columns([
        # Percentile rank (0-1 scale)
        pl.col("ss_ratio").rank(method="average").over("Code") / pl.col("ss_ratio").count().over("Code"),

        # Rolling percentile (252-day window)
        pl.col("ss_ratio").rolling_quantile(0.95, window_size=percentile_window).over("Code").alias("ss_p95_252d"),
        pl.col("ss_ratio").rolling_quantile(0.05, window_size=percentile_window).over("Code").alias("ss_p05_252d"),

        # Extreme flags
        (pl.col("ss_ratio") > pl.col("ss_p95_252d")).cast(pl.Int8).alias("ss_extreme_high"),
        (pl.col("ss_ratio") < pl.col("ss_p05_252d")).cast(pl.Int8).alias("ss_extreme_low"),

        # Combined extreme flag
        ((pl.col("ss_ratio") > pl.col("ss_p95_252d")) | (pl.col("ss_ratio") < pl.col("ss_p05_252d"))).cast(pl.Int8).alias("ss_extreme_flag"),
    ])


def add_short_zscore_features(ss: pl.DataFrame, z_window: int = 252) -> pl.DataFrame:
    """Add Z-score features for mean reversion signals.

    Features:
    - ss_ratio_z52: Short ratio Z-score (52-week)
    - ss_volume_ratio_z52: Volume ratio Z-score
    - ss_activity_z52: Activity Z-score
    - ss_mean_reversion_signal: Mean reversion signal strength
    """
    if ss.is_empty():
        return ss

    # Compute rolling statistics
    ss = ss.with_columns([
        # Rolling means
        pl.col("ss_ratio").rolling_mean(z_window).over("Code").alias("_ss_ratio_mean"),
        pl.col("ss_volume_ratio").rolling_mean(z_window).over("Code").alias("_ss_volume_ratio_mean"),
        pl.col("ss_activity").rolling_mean(z_window).over("Code").alias("_ss_activity_mean"),

        # Rolling stds
        pl.col("ss_ratio").rolling_std(z_window).over("Code").alias("_ss_ratio_std"),
        pl.col("ss_volume_ratio").rolling_std(z_window).over("Code").alias("_ss_volume_ratio_std"),
        pl.col("ss_activity").rolling_std(z_window).over("Code").alias("_ss_activity_std"),
    ])

    # Compute Z-scores
    ss = ss.with_columns([
        ((pl.col("ss_ratio") - pl.col("_ss_ratio_mean")) / (pl.col("_ss_ratio_std") + EPS)).alias("ss_ratio_z52"),
        ((pl.col("ss_volume_ratio") - pl.col("_ss_volume_ratio_mean")) / (pl.col("_ss_volume_ratio_std") + EPS)).alias("ss_volume_ratio_z52"),
        ((pl.col("ss_activity") - pl.col("_ss_activity_mean")) / (pl.col("_ss_activity_std") + EPS)).alias("ss_activity_z52"),
    ])

    # Mean reversion signal (absolute Z-score)
    ss = ss.with_columns([
        pl.col("ss_ratio_z52").abs().alias("ss_mean_reversion_signal")
    ])

    # Drop temporary columns
    temp_cols = [c for c in ss.columns if c.startswith("_ss_")]
    if temp_cols:
        ss = ss.drop(temp_cols)

    return ss


def attach_adv20_and_scale(ss_feat: pl.DataFrame, adv20_df: pl.DataFrame | None) -> pl.DataFrame:
    """Attach ADV20 and create liquidity-scaled features.

    Features:
    - ss_volume_to_adv20: Short volume scaled by ADV20
    - ss_balance_to_adv20: Balance scaled by ADV20
    - ss_turnover_adj: Liquidity-adjusted turnover
    """
    if ss_feat.is_empty() or adv20_df is None or adv20_df.is_empty():
        return ss_feat

    # Join with ADV20 data
    ss_with_adv = ss_feat.join(
        adv20_df.select(["Code", "Date", "ADV20_shares"]),
        on=["Code", "Date"],
        how="left"
    )

    # Create liquidity-scaled features
    return ss_with_adv.with_columns([
        # Volume scaled features
        (pl.col("ss_volume_abs") / (pl.col("ADV20_shares") + EPS)).alias("ss_volume_to_adv20"),

        # Balance scaled features (if available)
        (pl.col("ss_balance").fill_null(0) / (pl.col("ADV20_shares") + EPS)).alias("ss_balance_to_adv20"),

        # Turnover adjustment
        (pl.col("ss_activity") / (pl.col("ADV20_shares") + EPS)).alias("ss_turnover_adj"),

        # Liquidity regime (high/low ADV20)
        (pl.col("ADV20_shares") > pl.col("ADV20_shares").quantile(0.7).over("Date")).cast(pl.Int8).alias("ss_high_liquidity"),
    ])


def asof_attach_to_daily(quotes: pl.DataFrame, ss_k: pl.DataFrame) -> pl.DataFrame:
    """As-of join short selling data to daily quotes (T+1 leak-safe).

    Args:
        quotes: Daily quotes (Code, Date, ...)
        ss_k: Short selling data with effective_start

    Returns:
        Daily quotes with short selling features attached
    """
    if ss_k.is_empty():
        return quotes

    # Create join keys for as-of join
    quotes_keys = quotes.select(["Code", "Date"]).unique()

    # Filter short data to valid effective dates only
    ss_valid = ss_k.filter(
        pl.col("effective_start").is_not_null() &
        pl.col("effective_start") <= pl.col("Date")  # Only use data effective by this date
    )

    if ss_valid.is_empty():
        # Add null columns for consistency
        ss_cols = [c for c in ss_k.columns if c.startswith("ss_")]
        null_exprs = [pl.lit(None).alias(c) for c in ss_cols]
        null_exprs.append(pl.lit(0).cast(pl.Int8).alias("is_ss_valid"))
        return quotes.with_columns(null_exprs)

    # As-of join: for each (Code, Date), get the latest short data where effective_start <= Date
    joined = quotes_keys.join_asof(
        ss_valid.sort(["Code", "effective_start"]),
        left_on="Date",
        right_on="effective_start",
        by="Code",
        strategy="backward"
    )

    # Add validity flag
    joined = joined.with_columns([
        (pl.col("effective_start").is_not_null()).cast(pl.Int8).alias("is_ss_valid")
    ])

    # Join back to original quotes
    return quotes.join(joined, on=["Code", "Date"], how="left")


def combine_short_and_positions(
    short_df: pl.DataFrame,
    positions_df: pl.DataFrame | None
) -> pl.DataFrame:
    """Combine short selling ratio data with positions data.

    Args:
        short_df: Short selling ratio data
        positions_df: Short selling positions data (optional)

    Returns:
        Combined DataFrame with both ratio and positions features
    """
    if positions_df is None or positions_df.is_empty():
        return short_df

    # Join on (Code, Date)
    combined = short_df.join(
        positions_df,
        on=["Code", "Date"],
        how="outer_coalesce"  # Keep all records from both sides
    )

    return combined


def add_short_selling_block(
    quotes: pl.DataFrame,
    short_df: pl.DataFrame | None,
    positions_df: pl.DataFrame | None = None,
    adv20_df: pl.DataFrame | None = None,
    *,
    enable_z_scores: bool = True,
    z_window: int = 252,
    extreme_window: int = 252,
) -> pl.DataFrame:
    """Complete short selling features integration pipeline.

    Args:
        quotes: Base daily quotes DataFrame
        short_df: Short selling ratio data
        positions_df: Short selling positions data (optional)
        adv20_df: ADV20 data for liquidity scaling (optional)
        enable_z_scores: Whether to compute Z-score features
        z_window: Window for Z-score calculation
        extreme_window: Window for extreme detection

    Returns:
        DataFrame with short selling features attached
    """
    if short_df is None or short_df.is_empty():
        # Add null features for consistency
        null_cols = [
            "ss_ratio", "ss_volume_ratio", "ss_volume_abs", "ss_activity",
            "ss_ratio_ma5", "ss_ratio_ma20", "ss_momentum_5d", "ss_momentum_10d",
            "ss_vol_5d", "ss_vol_20d", "ss_trend_strength", "ss_extreme_flag",
            "is_ss_valid"
        ]
        if enable_z_scores:
            null_cols.extend(["ss_ratio_z52", "ss_volume_ratio_z52", "ss_activity_z52"])

        null_exprs = [pl.lit(None).alias(c) for c in null_cols[:-1]]  # All except is_ss_valid
        null_exprs.append(pl.lit(0).cast(pl.Int8).alias("is_ss_valid"))

        return quotes.with_columns(null_exprs)

    # Step 1: Build effective dates (T+1 rule)
    ss_eff = build_short_effective(short_df)

    # Step 2: Combine with positions data if available
    if positions_df is not None and not positions_df.is_empty():
        positions_eff = build_short_effective(positions_df)
        positions_feat = add_short_positions_features(positions_eff)
        ss_combined = combine_short_and_positions(ss_eff, positions_feat)
    else:
        ss_combined = ss_eff

    # Step 3: Core features
    ss_feat = add_short_core_features(ss_combined)

    # Step 4: Statistical features
    ss_feat = add_short_statistical_features(ss_feat)

    # Step 5: Extreme detection
    ss_feat = add_short_extreme_detection(ss_feat, percentile_window=extreme_window)

    # Step 6: Z-scores (optional)
    if enable_z_scores:
        ss_feat = add_short_zscore_features(ss_feat, z_window=z_window)

    # Step 7: ADV20 scaling (optional)
    if adv20_df is not None:
        ss_feat = attach_adv20_and_scale(ss_feat, adv20_df)

    # Step 8: As-of join to daily panel (T+1 leak-safe)
    result = asof_attach_to_daily(quotes, ss_feat)

    return result