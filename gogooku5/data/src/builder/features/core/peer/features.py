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
                    pl.col(column).mean().over(group_keys).alias(f"__{column}_group_mean"),
                    pl.col(column).std().over(group_keys).alias(f"__{column}_group_std"),
                ]
            )

            # Exclude self from peer statistics
            denominator = pl.when(pl.col("_peer_count") > 1).then(pl.col("_peer_count") - 1).otherwise(1)
            out = out.with_columns(
                [
                    ((pl.col(f"__{column}_group_mean") * pl.col("_peer_count") - pl.col(column)) / denominator).alias(
                        mean_col
                    ),
                    (pl.when(pl.col("_peer_count") > 1).then(pl.col(f"__{column}_group_std")).otherwise(None)).alias(
                        std_col
                    ),
                ]
            )

            out = out.with_columns(
                [
                    (pl.col(column) - pl.col(mean_col)).alias(diff_col),
                    (pl.col(column) / (pl.col(mean_col) + EPS)).alias(ratio_col),
                ]
            )

        drop_cols: Iterable[str] = [c for c in out.columns if c.startswith("__") or c == "_peer_count"]
        if drop_cols:
            out = out.drop(drop_cols)
        return out
