"""Normalize /markets/daily_margin_interest response."""

from __future__ import annotations

import json
from typing import Any

import polars as pl

SENTINELS = {"-", "*", "", "null", "NULL", "None"}

RENAME_MAP: dict[str, str] = {
    "Code": "code",
    "PublishedDate": "published_date",
    "ApplicationDate": "application_date",
    "PublishReason": "publish_reason",
    "ShortMarginOutstanding": "short_margin_outstanding",
    "LongMarginOutstanding": "long_margin_outstanding",
    "DailyChangeShortMarginOutstanding": "daily_change_short_margin_outstanding",
    "DailyChangeLongMarginOutstanding": "daily_change_long_margin_outstanding",
    "ShortMarginOutstandingListedShareRatio": "short_margin_outstanding_listed_share_ratio",
    "LongMarginOutstandingListedShareRatio": "long_margin_outstanding_listed_share_ratio",
    "ShortLongRatio": "short_long_ratio",
    "ShortNegotiableMarginOutstanding": "short_negotiable_margin_outstanding",
    "ShortStandardizedMarginOutstanding": "short_standardized_margin_outstanding",
    "LongNegotiableMarginOutstanding": "long_negotiable_margin_outstanding",
    "LongStandardizedMarginOutstanding": "long_standardized_margin_outstanding",
    "DailyChangeShortNegotiableMarginOutstanding": "daily_change_short_negotiable_margin_outstanding",
    "DailyChangeShortStandardizedMarginOutstanding": "daily_change_short_standardized_margin_outstanding",
    "DailyChangeLongNegotiableMarginOutstanding": "daily_change_long_negotiable_margin_outstanding",
    "DailyChangeLongStandardizedMarginOutstanding": "daily_change_long_standardized_margin_outstanding",
    "TSEMarginBorrowingAndLendingRegulationClassification": "tse_margin_regulation_classification",
    "PrecautionByJSF": "precaution_by_jsf",
    "RestrictedByJSF": "restricted_by_jsf",
}


def _cast_float(df: pl.DataFrame, col: str) -> pl.Expr:
    if col not in df.columns:
        return pl.lit(None).cast(pl.Float64).alias(col)
    if df[col].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64):
        return pl.col(col).cast(pl.Float64).alias(col)
    return (
        pl.when(pl.col(col).cast(pl.Utf8).is_in(SENTINELS))
        .then(None)
        .otherwise(pl.col(col).cast(pl.Float64, strict=False))
        .alias(col)
    )


def _cast_str(df: pl.DataFrame, col: str) -> pl.Expr:
    if col not in df.columns:
        return pl.lit(None).cast(pl.Utf8).alias(col)
    return pl.col(col).cast(pl.Utf8, strict=False).alias(col)


def normalize_daily_margin_interest(records: list[dict[str, Any]]) -> pl.DataFrame:
    """Return normalized daily_margin_interest frame."""
    if not records:
        return pl.DataFrame()

    # Normalize sentinel values in-place before DataFrame construction
    for item in records:
        for key, value in list(item.items()):
            if isinstance(value, str) and value in SENTINELS:
                item[key] = None

    df = pl.DataFrame(records)
    if df.is_empty():
        return pl.DataFrame()

    # Rename to snake_case
    rename_pairs = {src: dst for src, dst in RENAME_MAP.items() if src in df.columns}
    if rename_pairs:
        df = df.rename(rename_pairs)

    cols = df.columns

    # PublishReason may be dict; store as JSON string for DuckDB compatibility
    if "publish_reason" in cols:
        df = df.with_columns(
            pl.col("publish_reason")
            .map_elements(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else x)
            .cast(pl.Utf8, strict=False)
            .alias("publish_reason")
        )

    cast_exprs: list[pl.Expr] = [
        _cast_str(df, "code"),
        _cast_str(df, "tse_margin_regulation_classification"),
        _cast_str(df, "precaution_by_jsf"),
        _cast_str(df, "restricted_by_jsf"),
        pl.col("application_date").str.strptime(pl.Date, strict=False).alias("application_date")
        if "application_date" in cols
        else pl.lit(None).cast(pl.Date).alias("application_date"),
        pl.col("published_date").str.strptime(pl.Date, strict=False).alias("published_date")
        if "published_date" in cols
        else pl.lit(None).cast(pl.Date).alias("published_date"),
    ]

    float_cols = [
        "short_margin_outstanding",
        "long_margin_outstanding",
        "daily_change_short_margin_outstanding",
        "daily_change_long_margin_outstanding",
        "short_margin_outstanding_listed_share_ratio",
        "long_margin_outstanding_listed_share_ratio",
        "short_long_ratio",
        "short_negotiable_margin_outstanding",
        "short_standardized_margin_outstanding",
        "long_negotiable_margin_outstanding",
        "long_standardized_margin_outstanding",
        "daily_change_short_negotiable_margin_outstanding",
        "daily_change_short_standardized_margin_outstanding",
        "daily_change_long_negotiable_margin_outstanding",
        "daily_change_long_standardized_margin_outstanding",
    ]
    cast_exprs.extend(_cast_float(df, c) for c in float_cols)

    df = df.with_columns(cast_exprs)

    # Deduplicate corrections: keep latest published_date per (code, application_date)
    df = (
        df.filter(pl.col("code").is_not_null() & pl.col("application_date").is_not_null())
        .sort(["code", "application_date", "published_date"])
        .group_by(["code", "application_date"], maintain_order=True)
        .agg(pl.all().last())
        .sort(["code", "application_date"])
    )
    return df
