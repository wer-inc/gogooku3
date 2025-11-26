"""Normalize /markets/weekly_margin_interest response."""

from __future__ import annotations

import polars as pl

_FIELD_MAP = {
    "Date": "date",
    "Code": "code",
    "ShortMarginTradeVolume": "short_margin_trade_volume",
    "LongMarginTradeVolume": "long_margin_trade_volume",
    "ShortNegotiableMarginTradeVolume": "short_negotiable_margin_trade_volume",
    "LongNegotiableMarginTradeVolume": "long_negotiable_margin_trade_volume",
    "ShortStandardizedMarginTradeVolume": "short_standardized_margin_trade_volume",
    "LongStandardizedMarginTradeVolume": "long_standardized_margin_trade_volume",
    "IssueType": "issue_type",
}


def normalize_weekly_margin_interest(records: list[dict]) -> pl.DataFrame:
    if not records:
        return pl.DataFrame()
    mapped = [{v: rec.get(k) for k, v in _FIELD_MAP.items()} for rec in records]
    df = pl.DataFrame(mapped)
    for col in df.columns:
        if df[col].dtype == pl.Utf8:
            df = df.with_columns(pl.when(pl.col(col) == "").then(None).otherwise(pl.col(col)).alias(col))
    if "date" in df.columns:
        if df.select(pl.col("date").is_not_null().any()).item():
            df = df.with_columns(pl.col("date").str.strptime(pl.Date, strict=False, exact=False))
    return df
