"""Dividend as-of utilities and feature engineering."""
from __future__ import annotations

import datetime as dt

import polars as pl

from ..utils.asof_join import prepare_snapshot_pl

_SPECIAL_CODES = {"1", "2", "3"}


def prepare_dividend_snapshot(
    df: pl.DataFrame,
    *,
    trading_calendar: pl.DataFrame | None = None,
    availability_hour: int = 15,
    availability_minute: int = 0,
) -> pl.DataFrame:
    """Prepare dividend announcements for interval join."""
    if df.is_empty():
        return pl.DataFrame(
            {
                "Code": pl.Series([], dtype=pl.Utf8),
                "AnnouncementDate": pl.Series([], dtype=pl.Date),
                "ExDate": pl.Series([], dtype=pl.Date),
                "available_ts": pl.Series([], dtype=pl.Datetime("us", "Asia/Tokyo")),
            }
        )

    normalized = df.with_columns(
        [
            pl.col("Code").cast(pl.Utf8, strict=False).alias("Code"),
            pl.col("AnnouncementDate").cast(pl.Date, strict=False).alias("AnnouncementDate"),
            pl.col("AnnouncementTime").cast(pl.Utf8, strict=False).fill_null("15:00:00").alias("AnnouncementTime"),
            pl.col("ExDate").cast(pl.Date, strict=False).alias("ExDate"),
            pl.col("GrossDividendRate").cast(pl.Float64, strict=False).alias("div_amt"),
            pl.col("CommemorativeSpecialCode").cast(pl.Utf8, strict=False).alias("div_special_code"),
            pl.col("StatusCode").cast(pl.Utf8, strict=False).alias("status_code"),
            pl.col("ReferenceNumber").cast(pl.Utf8, strict=False).alias("reference_number"),
        ]
    )

    snapshot = prepare_snapshot_pl(
        normalized,
        published_date_col="AnnouncementDate",
        trading_calendar=trading_calendar,
        availability_hour=availability_hour,
        availability_minute=availability_minute,
    )
    snapshot = snapshot.with_columns(pl.col("available_ts").dt.convert_time_zone("Asia/Tokyo"))
    return snapshot


def _filter_latest_records(snapshot: pl.DataFrame) -> pl.DataFrame:
    """Keep the latest correction per (Code, ExDate)."""
    if snapshot.is_empty():
        return snapshot

    filtered = snapshot
    # StatusCode "3" = deletion
    if "status_code" in filtered.columns:
        filtered = filtered.filter(pl.col("status_code") != "3")

    sort_cols = [
        "Code",
        "ExDate",
        "AnnouncementDate",
        "available_ts",
        "AnnouncementTime",
    ]
    existing_sort = [col for col in sort_cols if col in filtered.columns]
    filtered = filtered.sort(existing_sort)
    filtered = filtered.group_by(["Code", "ExDate"], maintain_order=True).tail(1)
    return filtered


def build_dividend_feature_frame(
    snapshot: pl.DataFrame,
    *,
    trading_calendar: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build dividend feature frame keyed by (Code, available_ts) with P0 features.

    P0 features:
    - div_amount_next: Next ex-date dividend amount
    - div_amount_12m: 12-month dividend sum (same as div_sum_12m)
    - div_is_special: Special dividend flag
    - Quality flags: is_div_valid, div_staleness_days
    """
    if snapshot.is_empty():
        return pl.DataFrame(
            {
                "Code": pl.Series([], dtype=pl.Utf8),
                "available_ts": pl.Series([], dtype=pl.Datetime("us", "Asia/Tokyo")),
                "div_ex_date": pl.Series([], dtype=pl.Date),
                "div_amt": pl.Series([], dtype=pl.Float64),
                "div_amount_next": pl.Series([], dtype=pl.Float64),
                "div_sum_12m": pl.Series([], dtype=pl.Float64),
                "div_amount_12m": pl.Series([], dtype=pl.Float64),
                "div_is_special": pl.Series([], dtype=pl.Int8),
                "div_last_announcement_date": pl.Series([], dtype=pl.Date),
                "is_div_valid": pl.Series([], dtype=pl.Int8),
                "div_staleness_days": pl.Series([], dtype=pl.Int32),
            }
        )

    working = _filter_latest_records(snapshot)
    working = working.with_columns(
        [
            pl.col("div_amt").fill_null(0.0),
            pl.col("div_special_code").is_in(_SPECIAL_CODES).cast(pl.Int8).alias("div_is_special"),
            pl.col("AnnouncementDate").alias("div_last_announcement_date"),
        ]
    )

    def _calc_sum(group: pl.DataFrame) -> pl.DataFrame:
        group = group.sort("available_ts")
        # Convert timezone to UTC to ensure rolling_sum compatibility
        utc_ts = group["available_ts"].dt.convert_time_zone("UTC")
        group = group.with_columns(utc_ts.alias("_available_utc"))
        amounts = group["div_amt"].fill_null(0.0).to_list()
        ts_list = group["_available_utc"].to_list()
        sums: list[float] = []
        for idx, ts_item in enumerate(ts_list):
            total = 0.0
            for back in range(idx, -1, -1):
                if ts_item - ts_list[back] <= dt.timedelta(days=365):
                    total += float(amounts[back])
                else:
                    break
            sums.append(total)
        group = group.with_columns(pl.Series("div_sum_12m", sums))
        return group.drop("_available_utc")

    enriched = working.group_by("Code", maintain_order=True).map_groups(_calc_sum)

    # Calculate div_amount_next (next ex-date dividend)
    # For each available_ts, find the next ExDate and its dividend amount
    def _calc_next_ex_date(group: pl.DataFrame) -> pl.DataFrame:
        group = group.sort("available_ts")
        # Add a date column for comparison
        group = group.with_columns(pl.col("available_ts").dt.date().alias("_ts_date"))
        ex_dates = group["ExDate"].to_list()
        div_amounts = group["div_amt"].to_list()
        ts_dates = group["_ts_date"].to_list()

        next_amounts: list[float | None] = []
        for idx, ts_date in enumerate(ts_dates):
            next_amount = None
            # Find the next ExDate after this available_ts
            for jdx in range(idx, len(ex_dates)):
                if ex_dates[jdx] is not None and ts_date <= ex_dates[jdx]:
                    next_amount = div_amounts[jdx] if div_amounts[jdx] is not None else None
                    break
            next_amounts.append(next_amount)

        return group.with_columns(pl.Series("div_amount_next", next_amounts)).drop("_ts_date")

    enriched = enriched.group_by("Code", maintain_order=True).map_groups(_calc_next_ex_date)

    # Rename ExDate now that downstream logic expects div_ex_date
    if "ExDate" in enriched.columns and "div_ex_date" not in enriched.columns:
        enriched = enriched.rename({"ExDate": "div_ex_date"})

    # Add div_amount_12m (alias for div_sum_12m)
    enriched = enriched.with_columns(pl.col("div_sum_12m").alias("div_amount_12m"))

    # Add quality flags
    enriched = enriched.with_columns(
        [
            # is_div_valid: 1 if we have valid dividend data
            pl.when(pl.col("div_ex_date").is_not_null()).then(1).otherwise(0).cast(pl.Int8).alias("is_div_valid"),
            # div_staleness_days: days since last announcement (calendar days)
            pl.when(pl.col("div_last_announcement_date").is_not_null() & pl.col("available_ts").is_not_null())
            .then((pl.col("available_ts").dt.date() - pl.col("div_last_announcement_date")).dt.total_days())
            .otherwise(None)
            .cast(pl.Int32)
            .alias("div_staleness_days"),
        ]
    )

    keep_cols = [
        "Code",
        "available_ts",
        "div_amt",
        "div_amount_next",
        "div_is_special",
        "div_last_announcement_date",
        "div_sum_12m",
        "div_amount_12m",
        "is_div_valid",
        "div_staleness_days",
        "div_ex_date",
    ]
    available_cols = [col for col in keep_cols if col in enriched.columns]
    result = enriched.select(available_cols)
    result = result.with_columns(
        pl.col("available_ts").dt.convert_time_zone("UTC").dt.replace_time_zone(None).alias("available_ts")
    )
    return result.sort(["Code", "available_ts"])
