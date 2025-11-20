"""
Phase 2 Patch D: As-of join utilities for T+1 data availability.

Ensures weekly/snapshot data is joined with correct temporal alignment:
- Weekly data (e.g., margin trading): Available at T+1 9:00 JST
- Snapshot data (e.g., statements): Available at T+1 15:00 JST
- Never joins future data into current row
"""
from __future__ import annotations

import logging
from datetime import time
from typing import Literal
from zoneinfo import ZoneInfo

import polars as pl

LOGGER = logging.getLogger(__name__)
JST = ZoneInfo("Asia/Tokyo")


def prepare_snapshot_pl(
    df: pl.DataFrame,
    published_date_col: str = "PublishedDate",
    trading_calendar: pl.DataFrame | None = None,
    availability_hour: int = 9,
    availability_minute: int = 0,
) -> pl.DataFrame:
    """
    Prepare snapshot dataframe for as-of join with T+1 business day availability.

    Data published on day T becomes available at T+1 (next business day) 09:00 JST.

    Args:
        df: DataFrame with snapshot data (e.g., margin, flow, short selling)
        published_date_col: Published date column name (default: "PublishedDate")
        trading_calendar: Trading calendar DataFrame with 'date' column
                         If None, uses simple weekday heuristic (may miss holidays)
        availability_hour: Hour when data becomes available (default: 9 = 9am JST)
        availability_minute: Minute when data becomes available (default: 0)

    Returns:
        DataFrame with added 'available_ts' column (next business day at specified time)

    Example:
        >>> margin_df = prepare_snapshot_pl(margin, published_date_col="PublishedDate")
        >>> # Data published on 2025-01-06 (Mon) becomes available at 2025-01-07 (Tue) 09:00 JST
    """
    if df.is_empty():
        return df.with_columns(pl.lit(None).cast(pl.Datetime("us", "Asia/Tokyo")).alias("available_ts"))

    # 1) Resolve published date with fallbacks
    fallback_candidates = [published_date_col] + [
        candidate
        for candidate in ("application_date", "ApplicationDate", "date", "Date")
        if candidate != published_date_col
    ]
    date_exprs = [
        pl.col(col_name).cast(pl.Date, strict=False) for col_name in fallback_candidates if col_name in df.columns
    ]

    if not date_exprs:
        LOGGER.warning(
            "No candidate columns available for published date resolution (%s, fallbacks=%s); setting available_ts to NULL",
            published_date_col,
            fallback_candidates[1:],
        )
        return df.with_columns(pl.lit(None).cast(pl.Datetime("us", "Asia/Tokyo")).alias("available_ts"))

    work = df.with_columns(pl.coalesce(*date_exprs).alias("_published_effective"))

    # 2) Get unique published dates (exclude NULLs)
    unique_dates = (
        work.select(pl.col("_published_effective").alias("published_date"))
        .unique()
        .filter(pl.col("published_date").is_not_null())  # Filter out None values
        .sort("published_date")
    )

    # If no valid published dates, return with NULL available_ts
    if unique_dates.is_empty():
        LOGGER.warning(
            "No valid published dates resolved from %s (fallbacks=%s); setting available_ts to NULL",
            published_date_col,
            fallback_candidates[1:],
        )
        return work.drop("_published_effective").with_columns(
            pl.lit(None).cast(pl.Datetime("us", "Asia/Tokyo")).alias("available_ts")
        )

    # 3) Calculate next business day
    if trading_calendar is not None:
        # Use trading calendar for accurate business day calculation
        cal_dates = trading_calendar.select(pl.col("date").cast(pl.Date).alias("date")).sort("date")

        # For each published date, find next business day
        date_map = []
        for pub_date in unique_dates["published_date"].to_list():
            # Find next business day after published date
            next_bdays = cal_dates.filter(pl.col("date") > pub_date)
            if not next_bdays.is_empty():
                next_bday = next_bdays["date"].head(1).item()
                date_map.append({"published_date": pub_date, "next_business_date": next_bday})
            else:
                # Fallback: simple +1 day if no calendar match
                LOGGER.warning("No trading calendar match for %s, using +1 day fallback", pub_date)
                from datetime import timedelta

                date_map.append({"published_date": pub_date, "next_business_date": pub_date + timedelta(days=1)})

        date_mapping = pl.DataFrame(date_map)
    else:
        # Simple heuristic: skip weekends (may miss holidays)
        LOGGER.warning("No trading calendar provided, using weekday heuristic (may miss holidays)")

        def next_weekday(d):
            from datetime import timedelta

            # If Friday, +3 days (Mon); if Saturday, +2 days (Mon); else +1 day
            weekday = d.weekday()
            if weekday == 4:  # Friday
                return d + timedelta(days=3)
            elif weekday == 5:  # Saturday
                return d + timedelta(days=2)
            else:
                return d + timedelta(days=1)

        date_mapping = unique_dates.with_columns(
            [pl.col("published_date").map_elements(next_weekday, return_dtype=pl.Date).alias("next_business_date")]
        )

    # 4) Join next business date back to original dataframe
    result = work.join(date_mapping, left_on="_published_effective", right_on="published_date", how="left")

    # 5) Convert next business date to datetime at specified availability time (JST)
    result = result.with_columns(
        [
            pl.datetime(
                pl.col("next_business_date").dt.year(),
                pl.col("next_business_date").dt.month(),
                pl.col("next_business_date").dt.day(),
                pl.lit(availability_hour),
                pl.lit(availability_minute),
                pl.lit(0),
                time_zone="Asia/Tokyo",
            ).alias("available_ts")
        ]
    ).drop(["_published_effective", "next_business_date", "published_date"], strict=False)

    LOGGER.info(
        "Prepared snapshot data: %d rows, T+1 availability at %02d:%02d JST",
        len(result),
        availability_hour,
        availability_minute,
    )

    return result


def interval_join_pl(
    backbone: pl.DataFrame,
    snapshot: pl.DataFrame,
    on_code: str = "code",
    backbone_ts: str = "asof_ts",
    snapshot_ts: str = "available_ts",
    strategy: Literal["backward", "forward"] = "backward",
    suffix: str = "_snap",
) -> pl.DataFrame:
    """
    As-of join with T-leak detection and mandatory sorting.

    Joins snapshot data to backbone using as-of join strategy.
    Ensures no future data leaks into current row.

    CRITICAL: Both DataFrames must be sorted by (code, timestamp) before join_asof.

    Args:
        backbone: Main dataframe with asof_ts column
        snapshot: Snapshot dataframe (prepared with prepare_snapshot_pl)
        on_code: Stock code column (default: "code")
        backbone_ts: Backbone timestamp column (default: "asof_ts")
        snapshot_ts: Snapshot timestamp column (default: "available_ts")
        strategy: Join strategy (default: "backward" = use latest past data)
        suffix: Suffix for duplicate columns (default: "_snap")

    Returns:
        Joined dataframe with snapshot columns

    Raises:
        ValueError: If T-leak detected (future data joined into current row)

    Example:
        >>> backbone_df = add_asof_timestamp(backbone_df, date_col="date")
        >>> snapshot_df = prepare_snapshot_pl(margin_df, published_date_col="PublishedDate")
        >>> result = interval_join_pl(backbone_df, snapshot_df)
    """
    if backbone.is_empty() or snapshot.is_empty():
        LOGGER.warning("Empty dataframe in as-of join, returning backbone")
        return backbone

    # Validate required columns
    if backbone_ts not in backbone.columns:
        raise ValueError(f"Backbone missing '{backbone_ts}' column - call add_asof_timestamp first")
    if snapshot_ts not in snapshot.columns:
        raise ValueError(f"Snapshot missing '{snapshot_ts}' column - call prepare_snapshot_pl first")

    # CRITICAL: Sort both dataframes before join_asof
    backbone_sorted = backbone.sort([on_code, backbone_ts])
    snapshot_sorted = snapshot.sort([on_code, snapshot_ts])

    LOGGER.debug(
        "Sorted backbone (%d rows) and snapshot (%d rows) for as-of join",
        len(backbone_sorted),
        len(snapshot_sorted),
    )

    # Avoid duplicate column names up front (GPUバックエンドでの衝突を防ぐ)
    protected = {on_code, snapshot_ts, backbone_ts}
    overlap = [c for c in snapshot_sorted.columns if c in backbone_sorted.columns and c not in protected]
    if overlap:
        rename_map = {c: f"{c}{suffix}" for c in overlap}
        snapshot_sorted = snapshot_sorted.rename(rename_map)

    # Perform as-of join
    result = backbone_sorted.join_asof(
        snapshot_sorted,
        left_on=backbone_ts,
        right_on=snapshot_ts,
        by=on_code,
        strategy=strategy,
        suffix=suffix,
    )

    # Phase 2 Patch D: T-leak detection
    _detect_temporal_leaks(result, backbone_ts=backbone_ts, snapshot_ts=snapshot_ts + suffix)

    LOGGER.info(
        "As-of join completed: %d rows, strategy=%s, suffix=%s",
        len(result),
        strategy,
        suffix,
    )

    return result


def _detect_temporal_leaks(
    df: pl.DataFrame,
    backbone_ts: str = "asof_ts",
    snapshot_ts: str = "available_ts_snap",
) -> None:
    """
    Detect temporal leaks in as-of join result.

    Checks if any snapshot data is from the future relative to backbone timestamp.

    Args:
        df: Joined dataframe
        backbone_ts: Backbone timestamp column name
        snapshot_ts: Snapshot timestamp column name (with suffix applied)

    Raises:
        ValueError: If future data detected
    """
    if snapshot_ts not in df.columns:
        # No snapshot timestamps to check
        LOGGER.debug("No snapshot timestamp column '%s', skipping T-leak check", snapshot_ts)
        return

    # Check for future leaks: snapshot_ts > backbone_ts
    # CRITICAL: Snapshot data should NEVER be from the future
    leak_check = df.filter(pl.col(snapshot_ts).is_not_null() & (pl.col(snapshot_ts) > pl.col(backbone_ts)))

    if not leak_check.is_empty():
        n_leaks = len(leak_check)
        sample = leak_check.head(5).select([backbone_ts, snapshot_ts])
        raise ValueError(f"T-leak detected: {n_leaks} rows have future snapshot data\n" f"Sample violations:\n{sample}")

    LOGGER.debug("T-leak check passed: No future data detected")


def add_asof_timestamp(
    df: pl.DataFrame,
    date_col: str = "date",
    time_jst: time = time(15, 0),
) -> pl.DataFrame:
    """
    Add 'asof_ts' column to backbone dataframe.

    Converts date to datetime at specified JST time for as-of joins.

    Args:
        df: Backbone dataframe with date column
        date_col: Date column name (default: "date")
        time_jst: Time in JST when data becomes available (default: 15:00)

    Returns:
        DataFrame with added 'asof_ts' column

    Example:
        >>> backbone = add_asof_timestamp(backbone_df, time_jst=time(15, 0))
        >>> # 2025-01-06 becomes 2025-01-06 15:00:00+09:00 (JST)
    """
    if df.is_empty():
        return df.with_columns(pl.lit(None).cast(pl.Datetime("us", "Asia/Tokyo")).alias("asof_ts"))

    result = df.with_columns(
        pl.col(date_col)
        .cast(pl.Datetime)
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone("Asia/Tokyo")
        .dt.combine(pl.lit(time_jst))
        .alias("asof_ts")
    )

    LOGGER.info(
        "Added asof_ts column: %d rows, time=%s JST",
        len(result),
        time_jst.strftime("%H:%M"),
    )

    return result


def forward_fill_after_publication(
    df: pl.DataFrame,
    *,
    group_cols: str | list[str] = "code",
    sort_col: str = "date",
    columns: list[str] | None = None,
    maintain_original_order: bool = True,
) -> pl.DataFrame:
    """
    Forward fill feature columns within groups after publication/as-of alignment.

    Args:
        df: Input dataframe containing the columns to fill.
        group_cols: Column name(s) used to partition the data (default: "code").
        sort_col: Column defining chronological order inside each group.
        columns: Optional subset of columns to forward fill. Defaults to every column except group/sort helpers.
        maintain_original_order: Whether to restore the original row ordering after filling.

    Returns:
        DataFrame with specified columns forward-filled per group.
    """
    if df.is_empty():
        return df

    groups = [group_cols] if isinstance(group_cols, str) else list(group_cols)
    missing_groups = [col for col in groups if col not in df.columns]
    if missing_groups:
        raise ValueError(f"group column(s) missing: {missing_groups}")
    if sort_col not in df.columns:
        raise ValueError(f"sort column '{sort_col}' not found in dataframe")

    working = df
    if maintain_original_order:
        working = working.with_row_index("_ff_order")

    sort_keys = [*groups, sort_col]
    working = working.sort(sort_keys)

    if columns is None:
        excluded = set(groups + ["_ff_order", sort_col])
        target_cols = [col for col in working.columns if col not in excluded]
    else:
        target_cols = [col for col in columns if col in working.columns]

    if target_cols:
        fill_exprs = [pl.col(col).forward_fill().over(groups).alias(col) for col in target_cols]
        result = working.with_columns(fill_exprs)
    else:
        LOGGER.debug("forward_fill_after_publication: no target columns detected, returning input")
        result = working

    if maintain_original_order:
        return result.sort("_ff_order").drop("_ff_order")
    return result


__all__ = [
    "prepare_snapshot_pl",
    "interval_join_pl",
    "add_asof_timestamp",
    "forward_fill_after_publication",
]
