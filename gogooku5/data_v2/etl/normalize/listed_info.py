"""Normalize listed_info records."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import polars as pl


def _pick(df: pl.DataFrame, candidates: Sequence[str], dtype: pl.PolarsDataType, alias: str) -> pl.Expr:
    for name in candidates:
        if name in df.columns:
            return pl.col(name).cast(dtype, strict=False).alias(alias)
    return pl.lit(None, dtype=dtype).alias(alias)


def normalize_listed_info(records: list[dict[str, Any]]) -> pl.DataFrame:
    if not records:
        return pl.DataFrame()
    df = pl.DataFrame(records)
    out = (
        df.select(
            [
                _pick(df, ["Date", "date"], pl.Date, "date"),
                _pick(df, ["Code", "code"], pl.Utf8, "code"),
                _pick(df, ["CompanyName", "company_name"], pl.Utf8, "company_name"),
                _pick(df, ["CompanyNameEnglish", "company_name_english"], pl.Utf8, "company_name_english"),
                _pick(df, ["Sector17Code", "sector17_code"], pl.Utf8, "sector17_code"),
                _pick(df, ["Sector17CodeName", "sector17_name", "Sector17Name"], pl.Utf8, "sector17_name"),
                _pick(df, ["Sector33Code", "sector33_code"], pl.Utf8, "sector33_code"),
                _pick(df, ["Sector33CodeName", "sector33_name", "Sector33Name"], pl.Utf8, "sector33_name"),
                _pick(df, ["ScaleCategory", "scale_category"], pl.Utf8, "scale_category"),
                _pick(
                    df,
                    ["ScaleCategoryName", "scale_category_name", "ScaleCategoryLabel", "scale_category_label"],
                    pl.Utf8,
                    "scale_category_name",
                ),
                _pick(df, ["MarketCode", "market_code"], pl.Utf8, "market_code"),
                _pick(df, ["MarketCodeName", "market_code_name"], pl.Utf8, "market_code_name"),
                _pick(df, ["MarginCode", "margin_code"], pl.Utf8, "margin_code"),
                _pick(df, ["MarginCodeName", "margin_code_name"], pl.Utf8, "margin_code_name"),
            ]
        )
        .with_columns(
            pl.datetime(
                pl.col("date").dt.year(),
                pl.col("date").dt.month(),
                pl.col("date").dt.day(),
                9,
                0,
                0,
            )
            .dt.replace_time_zone("Asia/Tokyo")
            .alias("available_ts")
        )
        .filter(pl.col("date").is_not_null() & pl.col("code").is_not_null())
        .unique(subset=["date", "code"], keep="last")
        .sort(["date", "code"])
    )
    return out
