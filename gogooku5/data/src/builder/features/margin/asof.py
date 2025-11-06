"""As-of utilities for margin datasets (daily & weekly)."""
from __future__ import annotations

from typing import Iterable

import polars as pl

from ..utils.asof_join import prepare_snapshot_pl


def _rename_for_snapshot(df: pl.DataFrame, mapping: dict[str, str]) -> pl.DataFrame:
    present = {k: v for k, v in mapping.items() if k in df.columns}
    return df.rename(present) if present else df


def _select_with_defaults(
    snap: pl.DataFrame,
    *,
    code_alias: str = "code",
    date_alias: str = "date",
) -> pl.DataFrame:
    cols: Iterable[str] = snap.columns
    selected = [
        pl.col("Code").cast(pl.Utf8).alias(code_alias)
        if "Code" in cols
        else pl.lit(None).cast(pl.Utf8).alias(code_alias),
        pl.col("Date").cast(pl.Date, strict=False).alias(date_alias)
        if "Date" in cols
        else pl.lit(None).cast(pl.Date).alias(date_alias),
        pl.col("available_ts"),
    ]
    for name in snap.columns:
        if name not in {"Code", "Date", "available_ts"}:
            selected.append(pl.col(name))
    return snap.select(selected)


def prepare_margin_daily_asof(
    df: pl.DataFrame,
    *,
    trading_calendar: pl.DataFrame | None = None,
    availability_hour: int = 9,
    availability_minute: int = 0,
) -> pl.DataFrame:
    """Prepare daily margin dataset for T+1 as-of join."""
    if df.is_empty():
        return df

    normalized = _rename_for_snapshot(
        df,
        {
            "code": "Code",
            "Code": "Code",
            "date": "Date",
            "Date": "Date",
            "published_date": "PublishedDate",
            "PublishedDate": "PublishedDate",
            "application_date": "ApplicationDate",
            "ApplicationDate": "ApplicationDate",
        },
    )

    snap = prepare_snapshot_pl(
        normalized,
        published_date_col="PublishedDate",
        trading_calendar=trading_calendar,
        availability_hour=availability_hour,
        availability_minute=availability_minute,
    )

    return _select_with_defaults(snap)


def prepare_margin_weekly_asof(
    df: pl.DataFrame,
    *,
    trading_calendar: pl.DataFrame | None = None,
    availability_hour: int = 9,
    availability_minute: int = 0,
) -> pl.DataFrame:
    """Prepare weekly margin dataset for T+1 as-of join."""
    if df.is_empty():
        return df

    normalized = _rename_for_snapshot(
        df,
        {
            "code": "Code",
            "Code": "Code",
            "date": "Date",
            "Date": "Date",
            "publisheddate": "PublishedDate",
            "PublishedDate": "PublishedDate",
        },
    )

    snap = prepare_snapshot_pl(
        normalized,
        published_date_col="PublishedDate",
        trading_calendar=trading_calendar,
        availability_hour=availability_hour,
        availability_minute=availability_minute,
    )

    return _select_with_defaults(snap)
