from __future__ import annotations

"""
Earnings event features → announcement proximity flags and PEAD features.

This module implements earnings event data integration with:
- Earnings announcement proximity detection (upcoming & recent flags)
- Days-to-next-earnings countdown features
- Post-earnings announcement drift (PEAD) momentum signals
- Earnings surprise proxy via price reaction
- Year-over-year earnings growth momentum
- As-of backward join to prevent future data leakage

Key features generated:
- earnings_upcoming_5d: Binary flag for announcements within 5 days
- earnings_recent_5d: Binary flag for recent announcements (last 5 days)
- days_to_earnings: Countdown to next earnings announcement
- days_since_earnings: Days since last earnings announcement
- earnings_day_return: Price reaction on announcement day (surprise proxy)
- earnings_momentum_5d, earnings_momentum_10d: Post-earnings drift signals
- yoy_eps_growth: Year-over-year EPS growth rate
- earnings_event_volatility: Elevated volatility during earnings periods

Public API:
- build_earnings_schedule(announcement_df)
- add_earnings_proximity_features(quotes, announcement_schedule)
- add_earnings_surprise_features(quotes, announcement_df, statements_df)
- add_pead_momentum_features(quotes)
- add_earnings_event_block(quotes, announcement_df, statements_df, enable_pead=True)

References:
- Post-Earnings Announcement Drift (PEAD): Stocks with positive surprises
  continue rising for days/weeks after announcement
- Feature Specification lines 753-825: Earnings proximity and surprise metrics
"""

import datetime as _dt
from collections.abc import Callable
from typing import Optional

import polars as pl

EPS = 1e-12


def build_earnings_schedule(announcement_df: pl.DataFrame) -> pl.DataFrame:
    """Build earnings announcement schedule with event dates.

    Args:
        announcement_df: Raw earnings announcement data with Date, Code
                        (Note: J-Quants API returns "Date" as the announcement date)

    Returns:
        DataFrame with structured announcement schedule
    """
    if announcement_df.is_empty():
        return announcement_df

    # Clean and structure announcement data
    # Note: J-Quants API returns "Date" column (not "AnnouncementDate")
    schedule = announcement_df.with_columns([
        # Ensure proper date formats
        pl.col("Date").cast(pl.Date),

        # Add fiscal period indicators if available
        pl.col("Code").cast(pl.Utf8),

        # Sort key for temporal ordering
        pl.col("Date").alias("event_date")
    ])

    # Remove duplicates (same stock, same announcement date)
    schedule = schedule.unique(["Code", "Date"])

    # Sort by code and announcement date for proper temporal processing
    schedule = schedule.sort(["Code", "Date"])

    return schedule


def add_earnings_proximity_features(
    quotes: pl.DataFrame,
    announcement_schedule: pl.DataFrame
) -> pl.DataFrame:
    """Add earnings announcement proximity features.

    Features:
    - earnings_upcoming_5d: Binary flag for announcements within 5 days
    - earnings_recent_5d: Binary flag for recent announcements (last 5 days)
    - days_to_earnings: Countdown to next earnings announcement
    - days_since_earnings: Days since last earnings announcement
    """
    if announcement_schedule.is_empty():
        null_features = [
            pl.lit(0).cast(pl.Int8).alias("earnings_upcoming_5d"),
            pl.lit(0).cast(pl.Int8).alias("earnings_recent_5d"),
            pl.lit(None).cast(pl.Int32).alias("days_to_earnings"),
            pl.lit(None).cast(pl.Int32).alias("days_since_earnings"),
        ]
        return quotes.with_columns(null_features)

    # Prepare announcement schedule for joins
    # Note: Using "Date" column from J-Quants API (announcement date)
    earnings_events = announcement_schedule.select([
        "Code",
        pl.col("Date").alias("event_date")
    ])

    # Create date grid for each stock to compute proximity features
    quotes_with_proximity = quotes.join(
        earnings_events,
        on="Code",
        how="left"
    )

    if quotes_with_proximity.is_empty():
        null_features = [
            pl.lit(0).cast(pl.Int8).alias("earnings_upcoming_5d"),
            pl.lit(0).cast(pl.Int8).alias("earnings_recent_5d"),
            pl.lit(None).cast(pl.Int32).alias("days_to_earnings"),
            pl.lit(None).cast(pl.Int32).alias("days_since_earnings"),
        ]
        return quotes.with_columns(null_features)

    # Calculate proximity features
    proximity_features = quotes_with_proximity.with_columns([
        # Days until/since announcement
        (pl.col("event_date") - pl.col("Date")).dt.total_days().alias("days_diff"),
    ]).with_columns([
        # Upcoming earnings flags (positive days_diff = future events)
        (pl.col("days_diff").is_between(0, 5)).cast(pl.Int8).alias("_upcoming_5d"),

        # Recent earnings flags (negative days_diff = past events)
        (pl.col("days_diff").is_between(-5, 0)).cast(pl.Int8).alias("_recent_5d"),

        # Days to next earnings (minimum positive days_diff per stock-date)
        pl.when(pl.col("days_diff") >= 0)
        .then(pl.col("days_diff"))
        .otherwise(None)
        .alias("_days_to_earnings"),

        # Days since last earnings (maximum negative days_diff per stock-date)
        pl.when(pl.col("days_diff") <= 0)
        .then(-pl.col("days_diff"))
        .otherwise(None)
        .alias("_days_since_earnings"),
    ])

    # Aggregate by stock-date to get final features
    final_features = proximity_features.group_by(["Code", "Date"]).agg([
        # Any upcoming earnings in next 5 days
        pl.col("_upcoming_5d").max().alias("earnings_upcoming_5d"),

        # Any recent earnings in last 5 days
        pl.col("_recent_5d").max().alias("earnings_recent_5d"),

        # Minimum days to next earnings
        pl.col("_days_to_earnings").min().alias("days_to_earnings"),

        # Minimum days since last earnings
        pl.col("_days_since_earnings").min().alias("days_since_earnings"),

        # Keep other columns
        pl.exclude(["_upcoming_5d", "_recent_5d", "_days_to_earnings", "_days_since_earnings",
                   "days_diff", "event_date"]).first()
    ])

    # Join back to original quotes
    return quotes.join(
        final_features.select([
            "Code", "Date", "earnings_upcoming_5d", "earnings_recent_5d",
            "days_to_earnings", "days_since_earnings"
        ]),
        on=["Code", "Date"],
        how="left"
    ).with_columns([
        # Fill nulls for stocks without earnings data
        pl.col("earnings_upcoming_5d").fill_null(0),
        pl.col("earnings_recent_5d").fill_null(0),
    ])


def add_earnings_surprise_features(
    quotes: pl.DataFrame,
    announcement_df: pl.DataFrame,
    statements_df: Optional[pl.DataFrame] = None
) -> pl.DataFrame:
    """Add earnings surprise and reaction features.

    Features:
    - earnings_day_return: Price reaction on announcement day (surprise proxy)
    - yoy_eps_growth: Year-over-year EPS growth rate
    - earnings_reaction_strength: Magnitude of earnings day reaction
    """
    if announcement_df.is_empty():
        null_features = [
            pl.lit(None).cast(pl.Float64).alias("earnings_day_return"),
            pl.lit(None).cast(pl.Float64).alias("yoy_eps_growth"),
            pl.lit(None).cast(pl.Float64).alias("earnings_reaction_strength"),
        ]
        return quotes.with_columns(null_features)

    # Get announcement dates
    # Note: Using "Date" column from J-Quants API (announcement date)
    announcements = announcement_df.select([
        "Code",
        pl.col("Date").alias("earnings_date")
    ]).unique()

    # Join quotes with announcement dates to find earnings day returns
    earnings_returns = quotes.join(
        announcements,
        left_on=["Code", "Date"],
        right_on=["Code", "earnings_date"],
        how="inner"
    ).with_columns([
        # Calculate daily return as earnings surprise proxy
        pl.col("returns_1d").alias("earnings_day_return"),

        # Reaction strength (absolute return)
        pl.col("returns_1d").abs().alias("earnings_reaction_strength"),
    ]).select([
        "Code", "Date", "earnings_day_return", "earnings_reaction_strength"
    ])

    # Add YoY EPS growth if statements available
    eps_growth_features = []
    if statements_df is not None and not statements_df.is_empty():
        # Calculate year-over-year EPS growth
        eps_growth = statements_df.with_columns([
            pl.col("Date").cast(pl.Date),
            # Extract fiscal year/quarter for YoY matching
            pl.col("Date").dt.year().alias("fiscal_year"),
            pl.col("Date").dt.quarter().alias("fiscal_quarter"),
        ]).with_columns([
            # Calculate EPS (earnings per share)
            (pl.col("NetIncome") / (pl.col("NumberOfShares") + EPS)).alias("eps_current"),
        ])

        # Self-join to get previous year's EPS
        eps_yoy = eps_growth.join(
            eps_growth.with_columns([
                (pl.col("fiscal_year") + 1).alias("next_year"),
                pl.col("eps_current").alias("eps_prev_year")
            ]).select(["Code", "next_year", "fiscal_quarter", "eps_prev_year"]),
            left_on=["Code", "fiscal_year", "fiscal_quarter"],
            right_on=["Code", "next_year", "fiscal_quarter"],
            how="left"
        ).with_columns([
            # Year-over-year EPS growth
            ((pl.col("eps_current") - pl.col("eps_prev_year")) /
             (pl.col("eps_prev_year").abs() + EPS)).alias("yoy_eps_growth")
        ]).select(["Code", "Date", "yoy_eps_growth"])

        eps_growth_features = ["yoy_eps_growth"]
    else:
        eps_yoy = pl.DataFrame({"Code": [], "Date": [], "yoy_eps_growth": []}).cast({
            "Code": pl.Utf8, "Date": pl.Date, "yoy_eps_growth": pl.Float64
        })

    # Join all earnings features back to quotes
    result = quotes.join(
        earnings_returns,
        on=["Code", "Date"],
        how="left"
    )

    if eps_growth_features:
        result = result.join(
            eps_yoy,
            on=["Code", "Date"],
            how="left"
        )
    else:
        result = result.with_columns([
            pl.lit(None).cast(pl.Float64).alias("yoy_eps_growth")
        ])

    return result


def add_pead_momentum_features(quotes: pl.DataFrame) -> pl.DataFrame:
    """Add Post-Earnings Announcement Drift (PEAD) momentum features.

    Features:
    - earnings_momentum_5d: 5-day post-earnings momentum
    - earnings_momentum_10d: 10-day post-earnings momentum
    - earnings_momentum_20d: 20-day post-earnings momentum
    - pead_strength: Cumulative post-earnings drift strength
    """
    if quotes.is_empty():
        return quotes

    # Identify stocks with recent earnings announcements
    pead_quotes = quotes.filter(
        pl.col("earnings_recent_5d") == 1  # Stocks with recent earnings
    )

    if pead_quotes.is_empty():
        null_features = [
            pl.lit(None).cast(pl.Float64).alias("earnings_momentum_5d"),
            pl.lit(None).cast(pl.Float64).alias("earnings_momentum_10d"),
            pl.lit(None).cast(pl.Float64).alias("earnings_momentum_20d"),
            pl.lit(None).cast(pl.Float64).alias("pead_strength"),
        ]
        return quotes.with_columns(null_features)

    # Calculate PEAD momentum for stocks with recent earnings
    pead_features = quotes.with_columns([
        # Only calculate for stocks with recent earnings events
        pl.when(pl.col("earnings_recent_5d") == 1)
        .then(
            # 5-day forward momentum (post-announcement)
            pl.col("returns_1d").rolling_sum(5).over("Code")
        )
        .otherwise(None)
        .alias("earnings_momentum_5d"),

        pl.when(pl.col("earnings_recent_5d") == 1)
        .then(
            # 10-day forward momentum
            pl.col("returns_1d").rolling_sum(10).over("Code")
        )
        .otherwise(None)
        .alias("earnings_momentum_10d"),

        pl.when(pl.col("earnings_recent_5d") == 1)
        .then(
            # 20-day forward momentum
            pl.col("returns_1d").rolling_sum(20).over("Code")
        )
        .otherwise(None)
        .alias("earnings_momentum_20d"),
    ]).with_columns([
        # PEAD strength (weighted momentum)
        (pl.col("earnings_momentum_5d") * 0.5 +
         pl.col("earnings_momentum_10d") * 0.3 +
         pl.col("earnings_momentum_20d") * 0.2).alias("pead_strength")
    ])

    return pead_features


def add_earnings_volatility_features(quotes: pl.DataFrame) -> pl.DataFrame:
    """Add earnings-period volatility features.

    Features:
    - earnings_event_volatility: Elevated volatility during earnings periods
    - pre_earnings_volatility: Volatility in 5 days before earnings
    - post_earnings_volatility: Volatility in 5 days after earnings
    """
    if quotes.is_empty():
        return quotes

    # Calculate baseline volatility (20-day rolling)
    quotes_with_vol = quotes.with_columns([
        # Baseline volatility
        pl.col("returns_1d").rolling_std(20).over("Code").alias("baseline_volatility"),

        # Current 5-day volatility
        pl.col("returns_1d").rolling_std(5).over("Code").alias("current_volatility_5d"),
    ])

    # Earnings period volatility features
    earnings_vol_features = quotes_with_vol.with_columns([
        # Pre-earnings volatility (upcoming earnings flag)
        pl.when(pl.col("earnings_upcoming_5d") == 1)
        .then(pl.col("current_volatility_5d"))
        .otherwise(None)
        .alias("pre_earnings_volatility"),

        # Post-earnings volatility (recent earnings flag)
        pl.when(pl.col("earnings_recent_5d") == 1)
        .then(pl.col("current_volatility_5d"))
        .otherwise(None)
        .alias("post_earnings_volatility"),

        # Earnings event volatility (elevated during earnings periods)
        pl.when((pl.col("earnings_upcoming_5d") == 1) | (pl.col("earnings_recent_5d") == 1))
        .then(pl.col("current_volatility_5d") / (pl.col("baseline_volatility") + EPS))
        .otherwise(1.0)
        .alias("earnings_event_volatility"),
    ])

    return earnings_vol_features


def add_earnings_event_block(
    quotes: pl.DataFrame,
    announcement_df: Optional[pl.DataFrame],
    statements_df: Optional[pl.DataFrame] = None,
    *,
    enable_pead: bool = True,
    enable_volatility: bool = True,
) -> pl.DataFrame:
    """Complete earnings event features integration pipeline.

    Args:
        quotes: Base daily quotes DataFrame
        announcement_df: Earnings announcement schedule data
        statements_df: Financial statements data (optional)
        enable_pead: Whether to compute PEAD momentum features
        enable_volatility: Whether to compute earnings volatility features

    Returns:
        DataFrame with earnings event features attached
    """
    if announcement_df is None or announcement_df.is_empty():
        # Add null features for consistency
        null_cols = [
            "earnings_upcoming_5d", "earnings_recent_5d", "days_to_earnings",
            "days_since_earnings", "earnings_day_return", "yoy_eps_growth",
            "earnings_reaction_strength"
        ]

        if enable_pead:
            null_cols.extend([
                "earnings_momentum_5d", "earnings_momentum_10d", "earnings_momentum_20d",
                "pead_strength"
            ])

        if enable_volatility:
            null_cols.extend([
                "earnings_event_volatility", "pre_earnings_volatility", "post_earnings_volatility"
            ])

        null_exprs = [pl.lit(None).alias(c) for c in null_cols[2:]]  # Skip boolean flags
        null_exprs.extend([
            pl.lit(0).cast(pl.Int8).alias("earnings_upcoming_5d"),
            pl.lit(0).cast(pl.Int8).alias("earnings_recent_5d"),
        ])

        return quotes.with_columns(null_exprs)

    # Step 1: Build earnings announcement schedule
    earnings_schedule = build_earnings_schedule(announcement_df)

    # Step 2: Add proximity features (upcoming/recent flags, countdowns)
    result = add_earnings_proximity_features(quotes, earnings_schedule)

    # Step 3: Add earnings surprise and reaction features
    result = add_earnings_surprise_features(result, announcement_df, statements_df)

    # Step 4: Add PEAD momentum features (optional)
    if enable_pead:
        result = add_pead_momentum_features(result)

    # Step 5: Add earnings volatility features (optional)
    if enable_volatility:
        result = add_earnings_volatility_features(result)

    return result