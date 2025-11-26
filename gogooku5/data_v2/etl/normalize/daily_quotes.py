"""Normalize J-Quants daily_quotes payload."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import polars as pl


def _pick(df: pl.DataFrame, candidates: Sequence[str], dtype: pl.PolarsDataType, alias: str) -> pl.Expr:
    for name in candidates:
        if name in df.columns:
            return pl.col(name).cast(dtype, strict=False).alias(alias)
    return pl.lit(None, dtype=dtype).alias(alias)


def normalize_daily_quotes(records: list[dict[str, Any]]) -> pl.DataFrame:
    if not records:
        return pl.DataFrame()

    df = pl.DataFrame(records)
    out = (
        df.select(
            [
                _pick(df, ["Date", "date"], pl.Date, "date"),
                _pick(df, ["Code", "code"], pl.Utf8, "code"),
                _pick(df, ["Open", "open"], pl.Float64, "open"),
                _pick(df, ["High", "high"], pl.Float64, "high"),
                _pick(df, ["Low", "low"], pl.Float64, "low"),
                _pick(df, ["Close", "close"], pl.Float64, "close"),
                _pick(df, ["UpperLimit", "upper_limit"], pl.Utf8, "upper_limit"),
                _pick(df, ["LowerLimit", "lower_limit"], pl.Utf8, "lower_limit"),
                _pick(df, ["Volume", "volume"], pl.Int64, "volume"),
                _pick(df, ["TurnoverValue", "turnover_value"], pl.Float64, "turnover_value"),
                _pick(df, ["AdjustmentFactor", "adjustment_factor"], pl.Float64, "adjustment_factor"),
                _pick(df, ["AdjustmentOpen", "adjustment_open"], pl.Float64, "adjustment_open"),
                _pick(df, ["AdjustmentHigh", "adjustment_high"], pl.Float64, "adjustment_high"),
                _pick(df, ["AdjustmentLow", "adjustment_low"], pl.Float64, "adjustment_low"),
                _pick(df, ["AdjustmentClose", "adjustment_close"], pl.Float64, "adjustment_close"),
                _pick(df, ["AdjustmentVolume", "adjustment_volume"], pl.Int64, "adjustment_volume"),
                _pick(df, ["MorningOpen", "morning_open"], pl.Float64, "morning_open"),
                _pick(df, ["MorningHigh", "morning_high"], pl.Float64, "morning_high"),
                _pick(df, ["MorningLow", "morning_low"], pl.Float64, "morning_low"),
                _pick(df, ["MorningClose", "morning_close"], pl.Float64, "morning_close"),
                _pick(df, ["MorningUpperLimit", "morning_upper_limit"], pl.Utf8, "morning_upper_limit"),
                _pick(df, ["MorningLowerLimit", "morning_lower_limit"], pl.Utf8, "morning_lower_limit"),
                _pick(df, ["MorningVolume", "morning_volume"], pl.Int64, "morning_volume"),
                _pick(df, ["MorningTurnoverValue", "morning_turnover_value"], pl.Float64, "morning_turnover_value"),
                _pick(df, ["MorningAdjustmentOpen", "morning_adjustment_open"], pl.Float64, "morning_adjustment_open"),
                _pick(df, ["MorningAdjustmentHigh", "morning_adjustment_high"], pl.Float64, "morning_adjustment_high"),
                _pick(df, ["MorningAdjustmentLow", "morning_adjustment_low"], pl.Float64, "morning_adjustment_low"),
                _pick(
                    df, ["MorningAdjustmentClose", "morning_adjustment_close"], pl.Float64, "morning_adjustment_close"
                ),
                _pick(
                    df, ["MorningAdjustmentVolume", "morning_adjustment_volume"], pl.Int64, "morning_adjustment_volume"
                ),
                _pick(df, ["AfternoonOpen", "afternoon_open"], pl.Float64, "afternoon_open"),
                _pick(df, ["AfternoonHigh", "afternoon_high"], pl.Float64, "afternoon_high"),
                _pick(df, ["AfternoonLow", "afternoon_low"], pl.Float64, "afternoon_low"),
                _pick(df, ["AfternoonClose", "afternoon_close"], pl.Float64, "afternoon_close"),
                _pick(df, ["AfternoonUpperLimit", "afternoon_upper_limit"], pl.Utf8, "afternoon_upper_limit"),
                _pick(df, ["AfternoonLowerLimit", "afternoon_lower_limit"], pl.Utf8, "afternoon_lower_limit"),
                _pick(df, ["AfternoonVolume", "afternoon_volume"], pl.Int64, "afternoon_volume"),
                _pick(
                    df, ["AfternoonTurnoverValue", "afternoon_turnover_value"], pl.Float64, "afternoon_turnover_value"
                ),
                _pick(
                    df,
                    ["AfternoonAdjustmentOpen", "afternoon_adjustment_open"],
                    pl.Float64,
                    "afternoon_adjustment_open",
                ),
                _pick(
                    df,
                    ["AfternoonAdjustmentHigh", "afternoon_adjustment_high"],
                    pl.Float64,
                    "afternoon_adjustment_high",
                ),
                _pick(
                    df, ["AfternoonAdjustmentLow", "afternoon_adjustment_low"], pl.Float64, "afternoon_adjustment_low"
                ),
                _pick(
                    df,
                    ["AfternoonAdjustmentClose", "afternoon_adjustment_close"],
                    pl.Float64,
                    "afternoon_adjustment_close",
                ),
                _pick(
                    df,
                    ["AfternoonAdjustmentVolume", "afternoon_adjustment_volume"],
                    pl.Int64,
                    "afternoon_adjustment_volume",
                ),
            ]
        )
        .drop_nulls(subset=["date", "code"])
        .unique(subset=["date", "code"], keep="last")
        .sort(["date", "code"])
    )
    return out
