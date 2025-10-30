"""Simplified graph-based peer features."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
import polars as pl


@dataclass
class GraphFeatureConfig:
    code_column: str = "code"
    date_column: str = "date"
    return_column: str = "returns_1d"
    window_days: int = 60
    min_observations: int = 20
    correlation_threshold: float = 0.3
    shift_to_next_day: bool = True


class GraphFeatureEngineer:
    """Build simple correlation-based peer graph features per day."""

    def __init__(self, config: GraphFeatureConfig | None = None) -> None:
        self.config = config or GraphFeatureConfig()

    def add_features(self, df: pl.DataFrame) -> pl.DataFrame:
        cfg = self.config
        required = {cfg.code_column, cfg.date_column, cfg.return_column}
        if df.is_empty() or not required.issubset(df.columns):
            return df

        pdf = (
            df.select([cfg.code_column, cfg.date_column, cfg.return_column])
            .rename({cfg.code_column: "code", cfg.date_column: "date", cfg.return_column: "ret"})
            .to_pandas()
        )
        if not pd.api.types.is_datetime64_any_dtype(pdf["date"]):
            pdf["date"] = pd.to_datetime(pdf["date"])

        pdf = pdf.sort_values(["date", "code"]).reset_index(drop=True)
        dates = pdf["date"].drop_duplicates().sort_values().to_list()

        features: List[Dict[str, object]] = []
        window = timedelta(days=cfg.window_days)

        for current_date in dates:
            window_mask = (pdf["date"] <= current_date) & (pdf["date"] >= current_date - window)
            window_df = pdf.loc[window_mask]

            if window_df.empty:
                continue

            pivot = window_df.pivot_table(index="date", columns="code", values="ret", aggfunc="mean")
            pivot = pivot.dropna(thresh=cfg.min_observations, axis=0)
            if pivot.empty:
                continue

            corr = pivot.corr(min_periods=cfg.min_observations)
            if corr.empty:
                continue

            for code in corr.columns:
                if code not in corr.index:
                    continue
                peers = corr[code].drop(index=code).dropna()
                strong = peers[np.abs(peers) >= cfg.correlation_threshold]
                degree = int(strong.shape[0])
                mean_corr = float(np.abs(strong).mean()) if degree > 0 else 0.0
                max_corr = float(np.abs(strong).max()) if degree > 0 else 0.0
                features.append(
                    {
                        "code": code,
                        "date": current_date,
                        "graph_degree": degree,
                        "graph_peer_corr_mean": mean_corr,
                        "graph_peer_corr_max": max_corr,
                    }
                )

        if not features:
            return df

        feats_df = pd.DataFrame(features)
        if cfg.shift_to_next_day:
            feats_df["date"] = feats_df["date"] + timedelta(days=1)

        pl_feats = pl.from_pandas(feats_df).with_columns(
            [
                pl.col("date").cast(pl.Date),
                pl.col("graph_degree").cast(pl.Int32),
                pl.col("graph_peer_corr_mean").cast(pl.Float32),
                pl.col("graph_peer_corr_max").cast(pl.Float32),
            ]
        )

        out = df.join(
            pl_feats,
            left_on=[cfg.code_column, cfg.date_column],
            right_on=["code", "date"],
            how="left",
        )
        return out
