"""Margin daily feature engineering."""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

EPS = 1e-9


@dataclass
class MarginDailyConfig:
    code_column: str = "code"
    date_column: str = "date"
    long_column: str = "margin_balance"
    short_column: str = "short_balance"


class MarginDailyFeatureEngineer:
    """Derive daily margin trading features."""

    def __init__(self, config: MarginDailyConfig | None = None) -> None:
        self.config = config or MarginDailyConfig()

    def normalize(self, df: pl.DataFrame) -> pl.DataFrame:
        cfg = self.config
        out = df
        if cfg.date_column in out.columns:
            out = out.with_columns(
                pl.col(cfg.date_column)
                .cast(pl.Utf8, strict=False)
                .str.strptime(pl.Date, strict=False)
                .alias(cfg.date_column)
            )
        for column in (cfg.long_column, cfg.short_column):
            if column in out.columns:
                out = out.with_columns(pl.col(column).cast(pl.Float64, strict=False).alias(column))
        if cfg.code_column in out.columns:
            out = out.with_columns(pl.col(cfg.code_column).cast(pl.Utf8).alias(cfg.code_column))
        return out.sort([cfg.code_column, cfg.date_column])

    def build_features(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return df
        cfg = self.config
        code = cfg.code_column
        long_col, short_col = cfg.long_column, cfg.short_column

        out = self.normalize(df)
        if long_col not in out.columns:
            out = out.with_columns(pl.lit(0.0).alias(long_col))
        if short_col not in out.columns:
            out = out.with_columns(pl.lit(0.0).alias(short_col))

        out = out.with_columns(
            [
                (pl.col(long_col) - pl.col(short_col)).alias("margin_net"),
                (pl.col(long_col) + pl.col(short_col)).alias("margin_total"),
                (pl.col(long_col) / (pl.col(short_col) + EPS)).alias("margin_long_short_ratio"),
                ((pl.col(long_col) - pl.col(short_col)) / (pl.col(long_col) + pl.col(short_col) + EPS)).alias(
                    "margin_imbalance"
                ),
            ]
        )

        out = out.with_columns(
            [
                pl.col(long_col).diff().over(code).alias("margin_long_diff"),
                pl.col(short_col).diff().over(code).alias("margin_short_diff"),
                pl.col("margin_net").diff().over(code).alias("margin_net_diff"),
            ]
        )

        out = out.with_columns(
            [
                pl.col(long_col).rolling_mean(window_size=20, min_periods=5).over(code).alias("margin_long_ma20"),
                pl.col(long_col).rolling_std(window_size=20, min_periods=5).over(code).alias("margin_long_std20"),
            ]
        )

        out = out.with_columns(
            ((pl.col(long_col) - pl.col("margin_long_ma20")) / (pl.col("margin_long_std20") + EPS)).alias(
                "margin_long_z20"
            )
        )

        out = out.drop(["margin_long_ma20", "margin_long_std20"], strict=False)
        return out
