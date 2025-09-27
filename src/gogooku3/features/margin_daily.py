from __future__ import annotations

"""
Daily margin interest → daily panel integration (leak-safe, Polars).

This module implements the daily-to-daily pipeline with:
 - effective_start computation (T+1 rule from PublishedDate)
 - daily core features (stocks, diffs, z-scores, ratios)
 - ADV20-based scaling for liquidity normalization
 - PublishReason flags expansion (6 regulatory flags)
 - TSE regulation classification handling
 - as-of backward join to attach to a daily (Code, Date) grid

Minimal public API:
 - build_daily_effective(df_d, next_business_day)
 - add_daily_core_features(d)
 - add_publish_reason_flags(d)
 - attach_adv20_and_scale(d_feat, adv20)
 - asof_attach_to_daily(quotes, dk)
 - add_daily_margin_block(quotes, daily_df, adv20_df, *, enable_z_scores=True)

Notes:
 - ADV20_shares must be derived from adjusted volume; if only Volume is
   available, a reasonable approximation can be used upstream.
 - No time leakage: values become valid starting from effective_start only.
 - Daily margin data is sparser than weekly, typically 100-300 stocks per day.
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


def _add_business_days_jp(d: _dt.date, n: int) -> _dt.date:
    """Add n Japanese business days to a date (Mon-Fri).

    Handles weekend rollovers; ignores holidays.
    """
    cur = d
    count = 0
    while count < n:
        cur = cur + _dt.timedelta(days=1)
        if cur.weekday() < 5:  # Mon-Fri
            count += 1
    return cur


def build_daily_effective(
    df_d: pl.DataFrame,
    next_business_day: Callable[[pl.Expr], pl.Expr] | None = None,
) -> pl.DataFrame:
    """
    Compute effective_start for daily margin series.

    Rules:
      - effective_start = next_business_day(PublishedDate) [T+1 rule]

    Returns a DataFrame sorted by (Code, effective_start).
    """
    # Prefer injected holiday-aware next-business-day expr; fall back to weekday-only
    nbd = next_business_day or _next_business_day_jp

    d = df_d.with_columns(
        nbd(pl.col("PublishedDate")).alias("effective_start")
    )

    d = d.sort(["Code", "effective_start"])
    return d


def add_daily_core_features(d: pl.DataFrame) -> pl.DataFrame:
    """Derive daily core features (stock, diffs, z-scores, composition)."""
    g = (
        d.with_columns(
            [
                pl.col("LongMarginOutstanding").cast(pl.Float64).alias("dmi_long"),
                pl.col("ShortMarginOutstanding").cast(pl.Float64).alias("dmi_short"),
            ]
        )
        .with_columns(
            [
                (pl.col("dmi_long") - pl.col("dmi_short")).alias("dmi_net"),
                (pl.col("dmi_long") + pl.col("dmi_short")).alias("dmi_total"),
                (pl.col("dmi_long") / (pl.col("dmi_short") + EPS)).alias("dmi_credit_ratio"),
                (
                    (pl.col("dmi_long") - pl.col("dmi_short"))
                    / (pl.col("dmi_long") + pl.col("dmi_short") + EPS)
                ).alias("dmi_imbalance"),
                # Use API-provided ratio if available, otherwise calculate
                pl.when(pl.col("ShortLongRatio").is_not_null())
                .then(pl.col("ShortLongRatio") / 100.0)
                .otherwise(pl.col("dmi_short") / (pl.col("dmi_long") + EPS))
                .alias("dmi_short_long_ratio"),
                # Composition shares
                (
                    pl.col("LongStandardizedMarginOutstanding")
                    / (pl.col("dmi_long") + EPS)
                ).alias("dmi_std_share_long"),
                (
                    pl.col("ShortStandardizedMarginOutstanding")
                    / (pl.col("dmi_short") + EPS)
                ).alias("dmi_std_share_short"),
                (
                    pl.col("LongNegotiableMarginOutstanding")
                    / (pl.col("dmi_long") + EPS)
                ).alias("dmi_neg_share_long"),
                (
                    pl.col("ShortNegotiableMarginOutstanding")
                    / (pl.col("dmi_short") + EPS)
                ).alias("dmi_neg_share_short"),
            ]
        )
        .with_columns(
            [
                # Daily changes (1d lag)
                pl.col("dmi_long").diff().over("Code").alias("dmi_d_long_1d"),
                pl.col("dmi_short").diff().over("Code").alias("dmi_d_short_1d"),
                pl.col("dmi_net").diff().over("Code").alias("dmi_d_net_1d"),
                pl.col("dmi_credit_ratio").diff().over("Code").alias("dmi_d_ratio_1d"),
            ]
        )
        .with_columns([
            # Z-scores (26-day window, ~1 month)
            (
                (pl.col("dmi_short") - pl.col("dmi_short").rolling_mean(26).over("Code"))
                / (pl.col("dmi_short").rolling_std(26).over("Code") + EPS)
            ).alias("dmi_z26_short"),
            (
                (pl.col("dmi_long") - pl.col("dmi_long").rolling_mean(26).over("Code"))
                / (pl.col("dmi_long").rolling_std(26).over("Code") + EPS)
            ).alias("dmi_z26_long"),
            (
                (pl.col("dmi_total") - pl.col("dmi_total").rolling_mean(26).over("Code"))
                / (pl.col("dmi_total").rolling_std(26).over("Code") + EPS)
            ).alias("dmi_z26_total"),
            (
                (pl.col("dmi_d_short_1d") - pl.col("dmi_d_short_1d").rolling_mean(26).over("Code"))
                / (pl.col("dmi_d_short_1d").rolling_std(26).over("Code") + EPS)
            ).alias("dmi_z26_d_short_1d"),
        ])
    )
    return g


def add_publish_reason_flags(d: pl.DataFrame) -> pl.DataFrame:
    """Expand PublishReason struct into individual binary flags."""

    # Check if PublishReason column exists and has struct data
    publish_reason_dtype = d["PublishReason"].dtype

    reason_flags = [
        "Restricted",
        "DailyPublication",
        "Monitoring",
        "RestrictedByJSF",
        "PrecautionByJSF",
        "UnclearOrSecOnAlert"
    ]

    if publish_reason_dtype == pl.Null or d["PublishReason"].null_count() == len(d):
        d = d.with_columns([
            pl.lit(0, dtype=pl.Int8).alias(f"dmi_reason_{flag.lower()}")
            for flag in reason_flags
        ])
        d = d.with_columns(pl.lit(0, dtype=pl.Int8).alias("dmi_reason_count"))

    elif publish_reason_dtype == pl.Struct:
        def get_reason_flag(key: str) -> pl.Expr:
            return (
                pl.when(pl.col("PublishReason").is_not_null())
                .then(
                    pl.col("PublishReason").struct.field(key).cast(pl.Utf8).fill_null("0")
                )
                .otherwise("0")
                .eq("1")
                .cast(pl.Int8)
                .alias(f"dmi_reason_{key.lower()}")
            )

        d = d.with_columns([get_reason_flag(flag) for flag in reason_flags])
        d = d.with_columns(
            sum(pl.col(f"dmi_reason_{flag.lower()}") for flag in reason_flags).alias("dmi_reason_count")
        )

    elif publish_reason_dtype == pl.Utf8:
        def contains_flag(flag: str) -> pl.Expr:
            pattern = f"'{flag}': '1'"
            return (
                pl.when(pl.col("PublishReason").is_null() | (pl.col("PublishReason").str.len_chars() == 0))
                .then(0)
                .otherwise(pl.col("PublishReason").str.contains(pattern).cast(pl.Int8))
                .alias(f"dmi_reason_{flag.lower()}")
            )

        d = d.with_columns([contains_flag(flag) for flag in reason_flags])
        d = d.with_columns(
            sum(pl.col(f"dmi_reason_{flag.lower()}") for flag in reason_flags).alias("dmi_reason_count")
        )

    else:
        d = d.with_columns([
            pl.lit(0, dtype=pl.Int8).alias(f"dmi_reason_{flag.lower()}")
            for flag in reason_flags
        ])
        d = d.with_columns(pl.lit(0, dtype=pl.Int8).alias("dmi_reason_count"))

    # Add regulation classification ordinal mapping
    reg_map = {
        "001": 1,  # JSF caution/restriction
        "002": 2,  # TSE daily publication
        "003": 3,  # TSE regulation level 1
        "004": 4,  # TSE regulation level 2
        "005": 5,  # TSE regulation level 3
        "006": 6,  # TSE regulation level 4
        "101": 7,  # TSE regulation release
        "102": 8,  # TSE monitoring
    }

    # Create mapping expression with null handling for empty strings
    reg_mapping = pl.lit(None, dtype=pl.Int8)  # Start with null
    for code, level in reg_map.items():
        reg_mapping = pl.when(pl.col("TSEMarginBorrowingAndLendingRegulationClassification") == code).then(level).otherwise(reg_mapping)

    d = d.with_columns([
        reg_mapping.alias("dmi_tse_reg_level")
    ])

    return d


def attach_adv20_and_scale(d_feat: pl.DataFrame, adv20: pl.DataFrame) -> pl.DataFrame:
    """Join ADV20 data and compute liquidity-scaled features."""
    # Join ADV20 data as-of backward (effective_start should match or be before Date in adv20)
    d = d_feat.join_asof(
        adv20.select(["Code", "Date", "ADV20_shares"]).sort(["Code", "Date"]),
        left_on="effective_start",
        right_on="Date",
        by="Code",
        strategy="backward"
    )

    # Compute liquidity-scaled features
    d = d.with_columns([
        # Stock levels scaled by liquidity (days-to-cover approximation)
        (pl.col("dmi_long") / (pl.col("ADV20_shares") + EPS)).alias("dmi_long_to_adv20"),
        (pl.col("dmi_short") / (pl.col("ADV20_shares") + EPS)).alias("dmi_short_to_adv20"),
        (pl.col("dmi_total") / (pl.col("ADV20_shares") + EPS)).alias("dmi_total_to_adv20"),
        # Daily changes scaled by liquidity
        (pl.col("dmi_d_long_1d") / (pl.col("ADV20_shares") + EPS)).alias("dmi_d_long_to_adv1d"),
        (pl.col("dmi_d_short_1d") / (pl.col("ADV20_shares") + EPS)).alias("dmi_d_short_to_adv1d"),
        (pl.col("dmi_d_net_1d") / (pl.col("ADV20_shares") + EPS)).alias("dmi_d_net_to_adv1d"),
    ])

    return d


def asof_attach_to_daily(quotes: pl.DataFrame, dk: pl.DataFrame) -> pl.DataFrame:
    """Attach daily margin features to daily quotes using as-of join."""
    dk = dk.sort(["Code", "effective_start"])
    quotes = quotes.sort(["Code", "Date"])

    # Join as-of backward: for each (Code, Date), get the most recent effective_start <= Date
    result = quotes.join_asof(
        dk,
        left_on="Date",
        right_on="effective_start",
        by="Code",
        strategy="backward",
        suffix="_dmi"
    )

    # Add timing features
    result = result.with_columns([
        # Impulse flag: 1 on the effective_start date, 0 otherwise
        (pl.col("Date") == pl.col("effective_start")).cast(pl.Int8).alias("dmi_impulse"),
        # Days since publication
        (pl.col("Date") - pl.col("PublishedDate")).dt.total_days().alias("dmi_days_since_pub"),
        # Days since application date
        (pl.col("Date") - pl.col("ApplicationDate")).dt.total_days().alias("dmi_days_since_app"),
        # Valid data flag
        pl.when(pl.col("effective_start").is_not_null()).then(1).otherwise(0).cast(pl.Int8).alias("is_dmi_valid"),
    ])

    return result


def add_daily_margin_block(
    quotes: pl.DataFrame,
    daily_df: pl.DataFrame,
    adv20_df: pl.DataFrame | None = None,
    *,
    enable_z_scores: bool = True,
    next_business_day: Callable[[pl.Expr], pl.Expr] | None = None,
) -> pl.DataFrame:
    """
    Complete pipeline: daily margin → features → attach to quotes.

    Args:
        quotes: Daily stock quotes (Code, Date, ...)
        daily_df: Raw daily margin data from API
        adv20_df: ADV20 data for scaling (Code, Date, ADV20_shares)
        enable_z_scores: Whether to compute z-score features

    Returns:
        Enhanced quotes with dmi_* features attached
    """
    if daily_df.is_empty():
        # Return quotes with null dmi features
        return quotes.with_columns([
            pl.lit(None, dtype=pl.Float64).alias("dmi_long"),
            pl.lit(None, dtype=pl.Float64).alias("dmi_short"),
            pl.lit(None, dtype=pl.Int8).alias("is_dmi_valid"),
        ])

    # Step 1: Build effective dates (holiday-aware if provided)
    d = build_daily_effective(daily_df, next_business_day=next_business_day)

    # Step 2: Add core features
    d = add_daily_core_features(d)

    # Step 3: Add PublishReason flags
    d = add_publish_reason_flags(d)

    # Step 4: Attach ADV20 and scale if provided
    if adv20_df is not None and not adv20_df.is_empty():
        d = attach_adv20_and_scale(d, adv20_df)

    # Step 5: Attach to daily quotes
    result = asof_attach_to_daily(quotes, d)

    return result


def create_interaction_features(df: pl.DataFrame) -> pl.DataFrame:
    """Create interaction features for enhanced predictive power."""
    interactions = []

    # Only create interactions if we have the required columns
    required_cols = ["dmi_short_to_adv20", "dmi_long_to_adv20", "dmi_impulse"]
    if all(col in df.columns for col in required_cols):
        interactions.extend([
            # Squeeze setup: high short position + positive return signal
            (
                (pl.col("dmi_short_to_adv20") > pl.col("dmi_short_to_adv20").quantile(0.8))
                & (pl.col("returns_1d").fill_null(0) > 0)
            ).cast(pl.Int8).alias("dmi_short_squeeze_setup_d"),

            # Long unwind risk: high long position + negative return
            (
                (pl.col("dmi_z26_long").fill_null(0) > 1.5)
                & (pl.col("returns_1d").fill_null(0) < 0)
            ).cast(pl.Int8).alias("dmi_long_unwind_risk_d"),

            # Trend alignment: net flow direction matches price trend
            (
                pl.col("dmi_d_net_1d").fill_null(0).sign()
                == pl.col("returns_3d").fill_null(0).sign()
            ).cast(pl.Int8).alias("dmi_with_trend_d"),
        ])

    if interactions:
        df = df.with_columns(interactions)

    return df
    elif str(publish_reason_dtype) == str(pl.Object):  # dict-like objects
        def get_from_object(key: str) -> pl.Expr:
            return (
                pl.when(pl.col("PublishReason").is_not_null())
                .then(
                    pl.col("PublishReason").map_elements(lambda x: (x or {}).get(key, "0"), return_dtype=pl.Utf8)
                )
                .otherwise("0")
                .eq("1")
                .cast(pl.Int8)
                .alias(f"dmi_reason_{key.lower()}")
            )

        d = d.with_columns([get_from_object(flag) for flag in reason_flags])
        d = d.with_columns(
            sum(pl.col(f"dmi_reason_{flag.lower()}") for flag in reason_flags).alias("dmi_reason_count")
        )
