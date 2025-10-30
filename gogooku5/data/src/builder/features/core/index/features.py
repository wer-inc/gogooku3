"""Index feature engineering utilities migrated from gogooku3."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import polars as pl


@dataclass
class IndexFeatureConfig:
    code_column: str = "code"
    date_column: str = "date"
    open_column: str = "open"
    high_column: str = "high"
    low_column: str = "low"
    close_column: str = "close"


class IndexFeatureEngineer:
    """Compute per-index time series features and breadth metrics."""

    def __init__(self, config: IndexFeatureConfig | None = None) -> None:
        self.config = config or IndexFeatureConfig()

    def _base_columns(self) -> Sequence[str]:
        return [
            self.config.code_column,
            self.config.date_column,
            self.config.open_column,
            self.config.high_column,
            self.config.low_column,
            self.config.close_column,
        ]

    def normalize_types(self, df: pl.DataFrame) -> pl.DataFrame:
        cfg = self.config
        out = df
        if cfg.date_column in out.columns:
            out = out.with_columns(
                pl.col(cfg.date_column)
                .cast(pl.Utf8, strict=False)
                .str.strptime(pl.Date, strict=False)
                .alias(cfg.date_column)
            )
        for column in (
            cfg.open_column,
            cfg.high_column,
            cfg.low_column,
            cfg.close_column,
        ):
            if column in out.columns:
                out = out.with_columns(pl.col(column).cast(pl.Float64, strict=False).alias(column))
        if cfg.code_column in out.columns:
            out = out.with_columns(pl.col(cfg.code_column).cast(pl.Utf8).alias(cfg.code_column))
        return out.sort([cfg.code_column, cfg.date_column])

    def build_returns(self, df: pl.DataFrame) -> pl.DataFrame:
        cfg = self.config
        eps = 1e-12
        out = df
        close = cfg.close_column
        code = cfg.code_column

        out = out.with_columns(
            pl.col(close).shift(1).over(code).alias("_prev_close"),
            pl.col(close).shift(5).over(code).alias("_close_5d"),
            pl.col(close).shift(20).over(code).alias("_close_20d"),
        )
        out = out.with_columns(
            (pl.col(close) / (pl.col("_prev_close") + eps)).log().alias("idx_r_1d"),
            (pl.col(close) / (pl.col("_close_5d") + eps)).log().alias("idx_r_5d"),
            (pl.col(close) / (pl.col("_close_20d") + eps)).log().alias("idx_r_20d"),
        )
        open_col = cfg.open_column
        if open_col in out.columns:
            out = out.with_columns(
                (pl.col(close) / (pl.col(open_col) + eps)).log().alias("idx_r_oc"),
                (pl.col(open_col) / (pl.col("_prev_close") + eps)).log().alias("idx_r_co"),
            )
        return out

    def add_atr(self, df: pl.DataFrame, *, mask_halt_day: bool = True) -> pl.DataFrame:
        cfg = self.config
        high, low, close = cfg.high_column, cfg.low_column, cfg.close_column
        code, date = cfg.code_column, cfg.date_column
        eps = 1e-12
        if not {high, low, close}.issubset(df.columns):
            return df

        out = df.with_columns(
            pl.max_horizontal(
                pl.col(high) - pl.col(low),
                (pl.col(high) - pl.col("_prev_close")).abs(),
                (pl.col(low) - pl.col("_prev_close")).abs(),
            ).alias("_TR")
        )
        out = out.with_columns(
            pl.col("_TR").rolling_mean(window_size=14, min_periods=5).over(code).alias("idx_atr14"),
            (pl.col("_TR") / (pl.col(close) + eps))
            .rolling_mean(window_size=14, min_periods=5)
            .over(code)
            .alias("idx_natr14"),
        )
        if mask_halt_day:
            halt = pl.date(2020, 10, 1)
            out = out.with_columns(
                pl.when(pl.col(date) == halt).then(None).otherwise(pl.col("idx_atr14")).alias("idx_atr14"),
                pl.when(pl.col(date) == halt).then(None).otherwise(pl.col("idx_natr14")).alias("idx_natr14"),
            )
        return out

    def build_features(self, indices: pl.DataFrame) -> pl.DataFrame:
        if indices.is_empty():
            return indices
        df = self.normalize_types(indices)
        df = self.build_returns(df)
        df = self.add_atr(df)
        drop_cols = [col for col in ["_prev_close", "_close_5d", "_close_20d", "_TR"] if col in df.columns]
        if drop_cols:
            df = df.drop(drop_cols)
        return df
