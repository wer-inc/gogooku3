"""Morning/afternoon session feature engineering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, MutableMapping

import polars as pl

EPS = 1e-12


@dataclass
class SessionFeatureEngineer:
    """Build AM/PM derived features with configurable as-of policy."""

    _FEATURE_COLUMNS: ClassVar[tuple[str, ...]] = (
        "am_gap_prev_close",
        "am_range",
        "am_to_full_range",
        "am_vol_share",
        "am_limit_up_flag",
        "am_limit_down_flag",
        "am_limit_any_flag",
        "pm_gap_am_close",
        "pm_range",
    )

    _BASE_FLOAT_COLUMNS: ClassVar[tuple[str, ...]] = (
        "morning_open",
        "morning_high",
        "morning_low",
        "morning_close",
        "morning_volume",
        "morning_turnover_value",
        "afternoon_open",
        "afternoon_high",
        "afternoon_low",
        "afternoon_close",
        "afternoon_volume",
        "afternoon_turnover_value",
        "adjustmentclose",
        "adjustmenthigh",
        "adjustmentlow",
        "adjustmentvolume",
    )

    _BASE_INT_COLUMNS: ClassVar[tuple[str, ...]] = (
        "morning_upper_limit",
        "morning_lower_limit",
        "afternoon_upper_limit",
        "afternoon_lower_limit",
    )

    def add_features(
        self,
        df: pl.DataFrame,
        *,
        meta: MutableMapping[str, Any] | None = None,
        intraday_mode: bool = False,
    ) -> pl.DataFrame:
        """Add AM/PM derived features."""

        if df.is_empty():
            return self._ensure_columns(df)

        working = df.sort(["code", "date"])
        working = self._ensure_base_columns(working)

        lag_amt = 0 if intraday_mode else 1

        def lag(expr: pl.Expr, periods: int) -> pl.Expr:
            total = periods
            if total <= 0:
                return expr
            return expr.shift(total).over("code")

        am_open = lag(pl.col("morning_open"), lag_amt)
        am_high = lag(pl.col("morning_high"), lag_amt)
        am_low = lag(pl.col("morning_low"), lag_amt)
        # am_close is defined but not used - removed to reduce memory
        am_volume = lag(pl.col("morning_volume"), lag_amt)
        am_upper_limit = lag(pl.col("morning_upper_limit"), lag_amt)
        am_lower_limit = lag(pl.col("morning_lower_limit"), lag_amt)

        prev_close = lag(pl.col("adjustmentclose"), lag_amt + 1)
        adj_high = lag(pl.col("adjustmenthigh"), lag_amt)
        adj_low = lag(pl.col("adjustmentlow"), lag_amt)
        adj_volume = lag(pl.col("adjustmentvolume"), lag_amt)

        am_gap_prev_close = ((am_open / (prev_close + EPS)) - 1.0).alias("am_gap_prev_close")

        am_range = ((am_high - am_low) / (am_open + EPS)).alias("am_range")

        am_to_full_range = ((am_high - am_low) / ((adj_high - adj_low).abs() + EPS)).alias("am_to_full_range")

        am_vol_share = (am_volume / (adj_volume + EPS)).alias("am_vol_share")

        am_limit_up_flag = am_upper_limit.eq(1).fill_null(False).cast(pl.Int8).alias("am_limit_up_flag")
        am_limit_down_flag = am_lower_limit.eq(1).fill_null(False).cast(pl.Int8).alias("am_limit_down_flag")

        working = working.with_columns(
            [
                am_gap_prev_close,
                am_range,
                am_to_full_range,
                am_vol_share,
                am_limit_up_flag,
                am_limit_down_flag,
            ]
        )

        working = working.with_columns(
            (((pl.col("am_limit_up_flag") == 1) | (pl.col("am_limit_down_flag") == 1)).cast(pl.Int8)).alias(
                "am_limit_any_flag"
            )
        )

        pm_shift = 0 if intraday_mode else 1
        pm_gap_base = (pl.col("afternoon_open") / (pl.col("morning_close") + EPS)) - 1.0
        pm_gap_am_close = lag(pm_gap_base, pm_shift).alias("pm_gap_am_close")

        pm_range_base = (pl.col("afternoon_high") - pl.col("afternoon_low")) / (pl.col("afternoon_open") + EPS)
        pm_range = lag(pm_range_base, pm_shift).alias("pm_range")

        working = working.with_columns([pm_gap_am_close, pm_range])

        working = self._ensure_columns(working)

        if meta is not None:
            session_meta = meta.setdefault("session_features", {})
            session_meta.update(
                {
                    "columns": list(self._FEATURE_COLUMNS),
                    "mode": "intraday" if intraday_mode else "eod_shift1",
                    "eps": EPS,
                }
            )

        return working

    def _ensure_base_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        working = df
        for col in self._BASE_FLOAT_COLUMNS:
            if col not in working.columns:
                working = working.with_columns(pl.lit(None).cast(pl.Float64).alias(col))
        for col in self._BASE_INT_COLUMNS:
            if col not in working.columns:
                working = working.with_columns(pl.lit(None).cast(pl.Int8).alias(col))
        return working

    def _ensure_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        working = df
        specs: dict[str, pl.DataType] = {
            "am_gap_prev_close": pl.Float64,
            "am_range": pl.Float64,
            "am_to_full_range": pl.Float64,
            "am_vol_share": pl.Float64,
            "am_limit_up_flag": pl.Int8,
            "am_limit_down_flag": pl.Int8,
            "am_limit_any_flag": pl.Int8,
            "pm_gap_am_close": pl.Float64,
            "pm_range": pl.Float64,
        }
        for col, dtype in specs.items():
            if col not in working.columns:
                working = working.with_columns(pl.lit(None).cast(dtype).alias(col))
        return working
