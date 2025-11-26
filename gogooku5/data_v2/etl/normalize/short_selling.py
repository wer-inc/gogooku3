"""Normalize /markets/short_selling response."""

from __future__ import annotations

import polars as pl

_FIELD_MAP = {
    "Date": "date",
    "Sector33Code": "sector33_code",
    "SellingExcludingShortSellingTurnoverValue": "selling_excluding_short_selling_turnover_value",
    "ShortSellingWithRestrictionsTurnoverValue": "short_selling_with_restrictions_turnover_value",
    "ShortSellingWithoutRestrictionsTurnoverValue": "short_selling_without_restrictions_turnover_value",
}


def normalize_short_selling(records: list[dict]) -> pl.DataFrame:
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
