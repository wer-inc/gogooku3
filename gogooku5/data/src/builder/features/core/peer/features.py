"""Peer comparison features."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import polars as pl

EPS = 1e-9


@dataclass
class PeerFeatureConfig:
    date_column: str = "date"
    code_column: str = "code"
    group_column: str = "sector_code"
    numeric_columns: Sequence[str] = ("close",)


class PeerFeatureEngineer:
    """Compute peer-relative statistics within a group on each date."""

    def __init__(self, config: PeerFeatureConfig | None = None) -> None:
        self.config = config or PeerFeatureConfig()

    def add_features(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return df

        cfg = self.config
        required = {cfg.date_column, cfg.code_column, cfg.group_column}
        if not required.issubset(df.columns):
            return df

        available_cols = [col for col in cfg.numeric_columns if col in df.columns]
        if not available_cols:
            return df

        group_keys = [cfg.date_column, cfg.group_column]
        out = df
        out = out.with_columns(pl.count().over(group_keys).alias("_peer_count"))

        for column in available_cols:
            mean_col = f"{column}_peer_mean"
            std_col = f"{column}_peer_std"
            diff_col = f"{column}_peer_diff"
            ratio_col = f"{column}_peer_ratio"

            out = out.with_columns(
                [
                    pl.col(column).sum().over(group_keys).alias(f"__{column}_group_sum"),
                    (pl.col(column) ** 2).sum().over(group_keys).alias(f"__{column}_group_sq_sum"),
                ]
            )

            peer_count = pl.col("_peer_count") - 1
            peer_sum = pl.col(f"__{column}_group_sum") - pl.col(column)
            peer_sq_sum = pl.col(f"__{column}_group_sq_sum") - (pl.col(column) ** 2)

            # Exclude self from peer statistics; require at least one peer
            out = out.with_columns(
                [
                    pl.when(peer_count > 0).then(peer_sum / peer_count).otherwise(None).alias(mean_col),
                ]
            )
            variance = (
                pl.when(peer_count > 1)
                .then((peer_sq_sum - (peer_sum * peer_sum) / peer_count) / (peer_count - 1 + EPS))
                .otherwise(None)
            )
            out = out.with_columns(variance.alias(f"__{column}_peer_var"))
            out = out.with_columns(
                pl.when(pl.col(f"__{column}_peer_var").is_not_null())
                .then(
                    pl.when(pl.col(f"__{column}_peer_var") < 0.0)
                    .then(0.0)
                    .otherwise(pl.col(f"__{column}_peer_var"))
                    .sqrt()
                )
                .otherwise(None)
                .alias(std_col)
            )

            out = out.with_columns(
                [
                    pl.when(pl.col(mean_col).is_not_null())
                    .then(pl.col(column) - pl.col(mean_col))
                    .otherwise(None)
                    .alias(diff_col),
                    pl.when(pl.col(mean_col).abs() > EPS)
                    .then(pl.col(column) / pl.col(mean_col))
                    .otherwise(None)
                    .alias(ratio_col),
                ]
            )

        drop_cols: Iterable[str] = [c for c in out.columns if c.startswith("__") or c == "_peer_count"]
        if drop_cols:
            out = out.drop(drop_cols)
        return out
