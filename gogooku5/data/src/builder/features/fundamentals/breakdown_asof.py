"""Investor breakdown as-of normalization and feature engineering."""

from __future__ import annotations

from typing import Iterable, Sequence

import polars as pl

from ..utils.asof_join import prepare_snapshot_pl
from ..utils.rolling import roll_mean_safe, roll_std_safe

EPS = 1e-9

_BUY_COLUMNS: tuple[str, ...] = (
    "LongBuyValue",
    "MarginBuyNewValue",
    "BuyValue",
)

_SELL_COLUMNS: tuple[str, ...] = (
    "LongSellValue",
    "MarginSellNewValue",
    "ShortSellWithoutMarginValue",
    "SellValue",
)

_SHORT_COLUMNS: tuple[str, ...] = (
    "ShortSellWithoutMarginValue",
    "MarginSellNewValue",
)

_LOCAL_WINDOW = 3


def _cast_numeric(df: pl.DataFrame, columns: Iterable[str]) -> pl.DataFrame:
    present = [col for col in columns if col in df.columns]
    if present:
        df = df.with_columns([pl.col(col).cast(pl.Float64, strict=False).alias(col) for col in present])
    return df


def _sum_horizontal(columns: Sequence[str]) -> pl.Expr:
    exprs = [pl.col(col).fill_null(0.0) for col in columns]
    if not exprs:
        return pl.lit(0.0)
    return pl.sum_horizontal(exprs).fill_null(0.0)


def _sign(expr: pl.Expr) -> pl.Expr:
    return pl.when(expr > 0).then(pl.lit(1.0)).when(expr < 0).then(pl.lit(-1.0)).otherwise(pl.lit(0.0))


def _local_extreme_flag(col: str, *, is_max: bool, window: int = _LOCAL_WINDOW) -> pl.Expr:
    """
    Flag whether the current row is a local max/min within +/- window days for the specified column.

    Requires complete window (forward/backward) history; edges without enough history are 0.
    """
    current = pl.col(col)
    conditions: list[pl.Expr] = [current.is_not_null()]
    comparator = pl.col(col)

    for offset in range(1, window + 1):
        forward = comparator.shift(-offset).over("Code")
        backward = comparator.shift(offset).over("Code")
        conditions.extend([forward.is_not_null(), backward.is_not_null()])
        if is_max:
            conditions.extend([current >= forward, current >= backward])
        else:
            conditions.extend([current <= forward, current <= backward])

    return pl.all_horizontal(conditions).cast(pl.Int8)


def prepare_breakdown_snapshot(
    df: pl.DataFrame,
    *,
    trading_calendar: pl.DataFrame | None = None,
    availability_hour: int = 15,
    availability_minute: int = 0,
) -> pl.DataFrame:
    """Normalize raw breakdown payload into snapshot form with availability timestamps."""
    if df.is_empty():
        return pl.DataFrame(
            {
                "code": pl.Series([], dtype=pl.Utf8),
                "date": pl.Series([], dtype=pl.Date),
                "available_ts": pl.Series([], dtype=pl.Datetime("us", "Asia/Tokyo")),
            }
        )

    renamed = df.rename({col: col.strip() for col in df.columns})
    normalized = renamed.with_columns(
        [
            pl.col("Code").cast(pl.Utf8, strict=False).alias("Code"),
            pl.col("Date").cast(pl.Date, strict=False).alias("Date"),
        ]
    )
    normalized = _cast_numeric(normalized, set(_BUY_COLUMNS + _SELL_COLUMNS + _SHORT_COLUMNS))

    snapshot = prepare_snapshot_pl(
        normalized,
        published_date_col="Date",
        trading_calendar=trading_calendar,
        availability_hour=availability_hour,
        availability_minute=availability_minute,
    )
    snapshot = snapshot.with_columns(pl.col("available_ts").dt.convert_time_zone("Asia/Tokyo"))
    return snapshot


def build_breakdown_feature_frame(snapshot: pl.DataFrame) -> pl.DataFrame:
    """Construct P0 breakdown feature set (left-closed, normalized)."""
    if snapshot.is_empty():
        return pl.DataFrame(
            {
                "code": pl.Series([], dtype=pl.Utf8),
                "available_ts": pl.Series([], dtype=pl.Datetime("us", "Asia/Tokyo")),
                "bd_last_publish_date": pl.Series([], dtype=pl.Date),
                "bd_total_value": pl.Series([], dtype=pl.Float64),
                "bd_net_value": pl.Series([], dtype=pl.Float64),
                "bd_net_ratio": pl.Series([], dtype=pl.Float64),
                "bd_short_share": pl.Series([], dtype=pl.Float64),
                "bd_activity_ratio": pl.Series([], dtype=pl.Float64),
                "bd_net_ratio_chg_1d": pl.Series([], dtype=pl.Float64),
                "bd_short_share_chg_1d": pl.Series([], dtype=pl.Float64),
                "bd_net_z20": pl.Series([], dtype=pl.Float64),
                "bd_turn_up": pl.Series([], dtype=pl.Int8),
                "is_bd_valid": pl.Series([], dtype=pl.Int8),
                "bd_net_z260": pl.Series([], dtype=pl.Float64),
                "bd_short_z260": pl.Series([], dtype=pl.Float64),
                "bd_credit_new_net": pl.Series([], dtype=pl.Float64),
                "bd_credit_close_net": pl.Series([], dtype=pl.Float64),
                "bd_net_ratio_local_max": pl.Series([], dtype=pl.Int8),
                "bd_net_ratio_local_min": pl.Series([], dtype=pl.Int8),
            }
        )

    working = snapshot.sort(["Code", "available_ts"]).with_columns(pl.col("Date").alias("bd_last_publish_date"))

    buy_expr = _sum_horizontal([col for col in _BUY_COLUMNS if col in working.columns]).alias("buy_value")
    sell_expr = _sum_horizontal([col for col in _SELL_COLUMNS if col in working.columns]).alias("sell_value")
    short_expr = _sum_horizontal([col for col in _SHORT_COLUMNS if col in working.columns]).alias("short_new_value")

    working = working.with_columns([buy_expr, sell_expr, short_expr])

    working = working.with_columns(
        [
            (pl.col("buy_value") + pl.col("sell_value")).alias("bd_total_value"),
            (pl.col("buy_value") - pl.col("sell_value")).alias("bd_net_value"),
        ]
    )

    working = working.with_columns(
        [
            (pl.col("bd_net_value") / (pl.col("bd_total_value") + EPS)).alias("bd_net_ratio"),
            (pl.col("short_new_value") / (pl.col("bd_total_value") + EPS)).alias("bd_short_share"),
        ]
    )

    working = working.with_columns(
        [
            pl.col("bd_net_ratio").clip(-1.0, 1.0).alias("bd_net_ratio"),
            pl.col("bd_short_share").clip(0.0, 1.0).alias("bd_short_share"),
        ]
    )

    working = working.with_columns(
        [
            (
                pl.col("bd_total_value")
                / (roll_mean_safe(pl.col("bd_total_value"), 20, min_periods=5, by="Code") + EPS)
            ).alias("bd_activity_ratio"),
        ]
    )

    working = working.with_columns(
        [
            (pl.col("bd_net_ratio") - pl.col("bd_net_ratio").shift(1).over("Code")).alias("bd_net_ratio_chg_1d"),
            (pl.col("bd_short_share") - pl.col("bd_short_share").shift(1).over("Code")).alias("bd_short_share_chg_1d"),
        ]
    )

    net_mean_20 = roll_mean_safe(pl.col("bd_net_ratio"), 20, min_periods=5, by="Code")
    net_std_20 = roll_std_safe(pl.col("bd_net_ratio"), 20, min_periods=5, by="Code")

    working = working.with_columns(
        [
            ((pl.col("bd_net_ratio") - net_mean_20) / (net_std_20 + EPS)).alias("bd_net_z20"),
        ]
    )

    net_mean_260 = roll_mean_safe(pl.col("bd_net_ratio"), 260, min_periods=60, by="Code")
    net_std_260 = roll_std_safe(pl.col("bd_net_ratio"), 260, min_periods=60, by="Code")
    short_mean_260 = roll_mean_safe(pl.col("bd_short_share"), 260, min_periods=60, by="Code")
    short_std_260 = roll_std_safe(pl.col("bd_short_share"), 260, min_periods=60, by="Code")

    working = working.with_columns(
        [
            ((pl.col("bd_net_ratio") - net_mean_260) / (net_std_260 + EPS)).alias("bd_net_z260"),
            ((pl.col("bd_short_share") - short_mean_260) / (short_std_260 + EPS)).alias("bd_short_z260"),
        ]
    )

    credit_exprs: list[pl.Expr] = []
    if {"MarginBuyNewValue", "MarginSellNewValue"}.issubset(working.columns):
        credit_exprs.append((pl.col("MarginBuyNewValue") - pl.col("MarginSellNewValue")).alias("bd_credit_new_net"))
    else:
        credit_exprs.append(pl.lit(None).cast(pl.Float64).alias("bd_credit_new_net"))

    if {"MarginSellCloseValue", "MarginBuyCloseValue"}.issubset(working.columns):
        credit_exprs.append(
            (pl.col("MarginSellCloseValue") - pl.col("MarginBuyCloseValue")).alias("bd_credit_close_net")
        )
    else:
        credit_exprs.append(pl.lit(None).cast(pl.Float64).alias("bd_credit_close_net"))

    working = working.with_columns(credit_exprs)

    working = working.with_columns(
        [
            _local_extreme_flag("bd_net_ratio", is_max=True).alias("bd_net_ratio_local_max"),
            _local_extreme_flag("bd_net_ratio", is_max=False).alias("bd_net_ratio_local_min"),
        ]
    )

    working = working.with_columns(
        [
            pl.when(
                (pl.col("bd_net_ratio_chg_1d").abs() > EPS)
                & (pl.col("bd_net_ratio_chg_1d").shift(1).over("Code").abs() > EPS)
            )
            .then(
                (
                    _sign(pl.col("bd_net_ratio_chg_1d")) != _sign(pl.col("bd_net_ratio_chg_1d").shift(1).over("Code"))
                ).cast(pl.Int8)
            )
            .otherwise(pl.lit(0, dtype=pl.Int8))
            .alias("bd_turn_up"),
            (pl.col("bd_total_value") > 0).cast(pl.Int8).alias("is_bd_valid"),
        ]
    )

    result = working.select(
        [
            pl.col("Code").alias("code"),
            "bd_last_publish_date",
            "available_ts",
            "bd_total_value",
            "bd_net_value",
            "bd_net_ratio",
            "bd_short_share",
            "bd_activity_ratio",
            "bd_net_ratio_chg_1d",
            "bd_short_share_chg_1d",
            "bd_net_z20",
            "bd_net_z260",
            "bd_short_z260",
            "bd_credit_new_net",
            "bd_credit_close_net",
            "bd_net_ratio_local_max",
            "bd_net_ratio_local_min",
            "bd_turn_up",
            "is_bd_valid",
        ]
    ).with_columns(pl.col("available_ts").dt.convert_time_zone("UTC").dt.replace_time_zone(None))
    return result
