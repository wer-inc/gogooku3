"""Margin daily feature engineering."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ...utils.rolling import roll_mean_safe, roll_std_safe

EPS = 1e-9


@dataclass
class MarginDailyConfig:
    code_column: str = "code"
    date_column: str = "date"
    long_column: str = "margin_balance"
    short_column: str = "short_balance"
    application_date_column: str = "application_date"


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
        if cfg.application_date_column in out.columns:
            out = out.with_columns(
                pl.col(cfg.application_date_column)
                .cast(pl.Utf8, strict=False)
                .str.strptime(pl.Date, strict=False)
                .alias(cfg.application_date_column)
            )
        for column in (cfg.long_column, cfg.short_column):
            if column in out.columns:
                out = out.with_columns(pl.col(column).cast(pl.Float64, strict=False).alias(column))
        if cfg.code_column in out.columns:
            out = out.with_columns(pl.col(cfg.code_column).cast(pl.Utf8).alias(cfg.code_column))
        sort_keys = [cfg.code_column]
        if cfg.application_date_column in out.columns:
            sort_keys.append(cfg.application_date_column)
        sort_keys.append(cfg.date_column)
        return out.sort(sort_keys)

    def build_features(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return df
        cfg = self.config
        code = cfg.code_column
        long_col, short_col = cfg.long_column, cfg.short_column

        out = self.normalize(df)

        if cfg.application_date_column not in out.columns:
            out = out.with_columns(pl.col(cfg.date_column).alias(cfg.application_date_column))

        buy_col = "margin_buy_volume"
        sell_col = "margin_sell_volume"

        # Phase 2: Keep original balance columns (margin_balance, short_balance)
        # They are needed downstream and shouldn't be dropped
        out = out.with_columns(
            [
                pl.col(long_col).alias(buy_col),
                pl.col(short_col).alias(sell_col),
            ]
        )

        # NOTE: Intentionally NOT dropping long_col/short_col to preserve raw balances

        out = out.with_columns(
            [
                (pl.col(buy_col) - pl.col(sell_col)).alias("margin_net"),
                (pl.col(buy_col) + pl.col(sell_col)).alias("margin_total"),
                (pl.col(buy_col) / (pl.col(sell_col) + EPS)).alias("margin_long_short_ratio"),
                ((pl.col(buy_col) - pl.col(sell_col)) / (pl.col(buy_col) + pl.col(sell_col) + EPS)).alias(
                    "margin_imbalance"
                ),
            ]
        )

        out = out.with_columns(
            [
                pl.col(buy_col).diff().over(code).alias("margin_buy_diff"),
                pl.col(sell_col).diff().over(code).alias("margin_sell_diff"),
                pl.col("margin_net").diff().over(code).alias("margin_net_diff"),
            ]
        )

        out = out.with_columns(
            [
                pl.col(buy_col).rolling_mean(window_size=20, min_periods=5).over(code).alias("_margin_buy_ma20"),
                pl.col(buy_col).rolling_std(window_size=20, min_periods=5).over(code).alias("_margin_buy_std20"),
            ]
        )

        out = out.with_columns(
            ((pl.col(buy_col) - pl.col("_margin_buy_ma20")) / (pl.col("_margin_buy_std20") + EPS)).alias(
                "margin_buy_z20"
            )
        )

        out = out.drop(["_margin_buy_ma20", "_margin_buy_std20"], strict=False)

        # Additional leak-safe, scale-stable ratios
        out = out.with_columns(
            pl.when(pl.col("margin_long_short_ratio").is_not_null() & (pl.col("margin_long_short_ratio") > 0))
            .then(pl.col("margin_long_short_ratio").log())
            .otherwise(None)
            .alias("margin_long_short_ratio_log")
        )

        # Detect unusual swings in the long/short balance (shift(1) to avoid look-ahead)
        out = out.with_columns(
            [
                roll_mean_safe(pl.col("margin_long_short_ratio"), 20, min_periods=5, by=code).alias("_mls_ma20"),
                roll_std_safe(pl.col("margin_long_short_ratio"), 20, min_periods=5, by=code).alias("_mls_std20"),
            ]
        )
        out = out.with_columns(
            pl.when(pl.col("_mls_std20").abs() > EPS)
            .then((pl.col("margin_long_short_ratio") - pl.col("_mls_ma20")) / (pl.col("_mls_std20") + EPS))
            .otherwise(None)
            .alias("margin_ratio_spike_z20")
        )
        out = out.with_columns((pl.col("margin_ratio_spike_z20").abs() > 2.5).cast(pl.Int8).alias("margin_ratio_spike"))
        out = out.drop(["_mls_ma20", "_mls_std20"], strict=False)
        return out
