"""Normalize trading_calendar records."""

from __future__ import annotations

import polars as pl


def normalize_trading_calendar(records: list[dict]) -> pl.DataFrame:
    if not records:
        return pl.DataFrame()
    df = pl.DataFrame(records)
    df = (
        df.select(
            [
                pl.col("Date").cast(pl.Date, strict=False).alias("date"),
                pl.col("HolidayDivision").cast(pl.Utf8, strict=False).alias("holiday_division"),
            ]
        )
        .drop_nulls(subset=["date"])
        .unique(subset=["date"], keep="last")
        .sort("date")
    )
    return df
