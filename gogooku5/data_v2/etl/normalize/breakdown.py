"""Normalize markets/breakdown payload."""

from __future__ import annotations

from typing import Any, Sequence

import polars as pl


def _pick(df: pl.DataFrame, candidates: Sequence[str], dtype: pl.PolarsDataType, alias: str) -> pl.Expr:
    for name in candidates:
        if name in df.columns:
            return pl.col(name).cast(dtype, strict=False).alias(alias)
    return pl.lit(None, dtype=dtype).alias(alias)


def normalize_breakdown(records: list[dict[str, Any]]) -> pl.DataFrame:
    if not records:
        return pl.DataFrame()
    df = pl.DataFrame(records)
    out = (
        df.select(
            [
                _pick(df, ["Date", "date"], pl.Date, "date"),
                _pick(df, ["Code", "code"], pl.Utf8, "code"),
                _pick(df, ["LongSellValue", "long_sell_value"], pl.Float64, "long_sell_value"),
                _pick(
                    df,
                    ["ShortSellWithoutMarginValue", "short_sell_without_margin_value"],
                    pl.Float64,
                    "short_sell_without_margin_value",
                ),
                _pick(df, ["MarginSellNewValue", "margin_sell_new_value"], pl.Float64, "margin_sell_new_value"),
                _pick(df, ["MarginSellCloseValue", "margin_sell_close_value"], pl.Float64, "margin_sell_close_value"),
                _pick(df, ["LongBuyValue", "long_buy_value"], pl.Float64, "long_buy_value"),
                _pick(df, ["MarginBuyNewValue", "margin_buy_new_value"], pl.Float64, "margin_buy_new_value"),
                _pick(df, ["MarginBuyCloseValue", "margin_buy_close_value"], pl.Float64, "margin_buy_close_value"),
                _pick(df, ["LongSellVolume", "long_sell_volume"], pl.Int64, "long_sell_volume"),
                _pick(
                    df,
                    ["ShortSellWithoutMarginVolume", "short_sell_without_margin_volume"],
                    pl.Int64,
                    "short_sell_without_margin_volume",
                ),
                _pick(df, ["MarginSellNewVolume", "margin_sell_new_volume"], pl.Int64, "margin_sell_new_volume"),
                _pick(df, ["MarginSellCloseVolume", "margin_sell_close_volume"], pl.Int64, "margin_sell_close_volume"),
                _pick(df, ["LongBuyVolume", "long_buy_volume"], pl.Int64, "long_buy_volume"),
                _pick(df, ["MarginBuyNewVolume", "margin_buy_new_volume"], pl.Int64, "margin_buy_new_volume"),
                _pick(df, ["MarginBuyCloseVolume", "margin_buy_close_volume"], pl.Int64, "margin_buy_close_volume"),
            ]
        )
        .drop_nulls(subset=["date", "code"])
        .unique(subset=["date", "code"], keep="last")
        .sort(["date", "code"])
    )
    return out
