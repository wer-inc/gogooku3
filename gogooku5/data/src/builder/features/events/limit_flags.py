"""Limit event feature engineering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, MutableMapping

import polars as pl

from ..utils.rolling import roll_sum_safe

EPS = 1e-9


@dataclass
class LimitEventFeatureEngineer:
    """Generate stop-limit derived features."""

    roll_window: int = 5

    _FEATURE_COLUMNS: ClassVar[tuple[str, ...]] = (
        "limit_up_flag",
        "limit_down_flag",
        "limit_any_flag",
        "limit_up_5d",
        "limit_down_5d",
        "days_since_limit",
        "price_locked_flag",
    )

    _BASE_SPECS: ClassVar[dict[str, pl.DataType]] = {
        "upper_limit": pl.Int8,
        "lower_limit": pl.Int8,
        "adjustmentclose": pl.Float64,
        "adjustmenthigh": pl.Float64,
        "adjustmentlow": pl.Float64,
    }

    def add_features(
        self,
        df: pl.DataFrame,
        *,
        meta: MutableMapping[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Augment dataframe with limit-event features."""

        if df.is_empty():
            return self._ensure_columns(df)

        working = df.sort(["code", "date"])
        working = self._ensure_base_columns(working)

        limit_up_flag = pl.col("upper_limit").eq(1).fill_null(False).cast(pl.Int8).alias("limit_up_flag")
        limit_down_flag = pl.col("lower_limit").eq(1).fill_null(False).cast(pl.Int8).alias("limit_down_flag")

        working = working.with_columns([limit_up_flag, limit_down_flag])

        limit_any_flag = (
            ((pl.col("limit_up_flag") == 1) | (pl.col("limit_down_flag") == 1)).cast(pl.Int8).alias("limit_any_flag")
        )
        working = working.with_columns(limit_any_flag)

        limit_up_5d = roll_sum_safe(
            pl.col("limit_up_flag").cast(pl.Float64),
            self.roll_window,
            by="code",
        ).alias("limit_up_5d")
        limit_down_5d = roll_sum_safe(
            pl.col("limit_down_flag").cast(pl.Float64),
            self.roll_window,
            by="code",
        ).alias("limit_down_5d")

        working = working.with_columns([limit_up_5d, limit_down_5d])

        working = working.with_columns(pl.arange(0, pl.len()).over("code").alias("_row_idx"))
        working = working.with_columns(
            pl.when(pl.col("limit_any_flag") == 1).then(pl.col("_row_idx")).otherwise(None).alias("_last_limit_idx")
        )
        working = working.with_columns(pl.col("_last_limit_idx").forward_fill().over("code").alias("_ff_limit_idx"))
        working = working.with_columns((pl.col("_row_idx") - pl.col("_ff_limit_idx")).alias("_days_since_limit"))
        working = working.with_columns(
            pl.when(pl.col("_ff_limit_idx").is_null())
            .then(None)
            .otherwise(pl.col("_days_since_limit"))
            .cast(pl.Int32, strict=False)
            .alias("days_since_limit")
        )

        price_locked_flag = (
            ((pl.col("limit_up_flag") == 1) & (pl.col("adjustmenthigh") - pl.col("adjustmentclose")).abs().lt(EPS))
            | ((pl.col("limit_down_flag") == 1) & (pl.col("adjustmentlow") - pl.col("adjustmentclose")).abs().lt(EPS))
        ).cast(pl.Int8)
        working = working.with_columns(price_locked_flag.alias("price_locked_flag"))

        cleanup_cols = ["_row_idx", "_last_limit_idx", "_ff_limit_idx", "_days_since_limit"]
        working = working.drop([col for col in cleanup_cols if col in working.columns], strict=False)

        working = self._ensure_columns(working)

        if meta is not None:
            limit_meta = meta.setdefault("limit_features", {})
            limit_meta.update(
                {
                    "columns": list(self._FEATURE_COLUMNS),
                    "roll_window": self.roll_window,
                    "policy": "left_closed_shift1",
                }
            )

        return working

    def _ensure_base_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        working = df
        for col, dtype in self._BASE_SPECS.items():
            if col not in working.columns:
                working = working.with_columns(pl.lit(None).cast(dtype).alias(col))
        return working

    def _ensure_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        working = df
        specs: dict[str, pl.DataType] = {
            "limit_up_flag": pl.Int8,
            "limit_down_flag": pl.Int8,
            "limit_any_flag": pl.Int8,
            "limit_up_5d": pl.Float64,
            "limit_down_5d": pl.Float64,
            "days_since_limit": pl.Int32,
            "price_locked_flag": pl.Int8,
        }
        for col, dtype in specs.items():
            if col not in working.columns:
                working = working.with_columns(pl.lit(None).cast(dtype).alias(col))
        return working
