"""Sector aggregation features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import polars as pl

EPS = 1e-9


@dataclass
class SectorAggregationConfig:
    date_column: str = "date"
    code_column: str = "code"
    sector_column: str = "sector_code"
    returns_1d: str = "ret_prev_1d"  # Phase 2: changed from returns_1d
    returns_5d: str = "ret_prev_5d"  # Phase 2: changed from returns_5d


class SectorAggregationFeatures:
    def __init__(self, config: SectorAggregationConfig | None = None) -> None:
        self.config = config or SectorAggregationConfig()

    def _sector_column(self, df: pl.DataFrame) -> Optional[str]:
        cfg = self.config
        if cfg.sector_column in df.columns:
            return cfg.sector_column
        candidates = ["sector33_id", "sector33_code", "sec33"]
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def add_features(self, df: pl.DataFrame, *, min_members: int = 3) -> pl.DataFrame:
        if df.is_empty():
            return df
        cfg = self.config
        sector_col = self._sector_column(df)
        if sector_col is None:
            return df

        cross_keys = [cfg.date_column, sector_col]
        out = df.sort([sector_col, cfg.date_column])

        # P0 FIX: Use only ret_prev_* columns (no forward-looking bias)
        ret_1d_col = cfg.returns_1d if cfg.returns_1d in out.columns else None
        ret_5d_col = cfg.returns_5d if cfg.returns_5d in out.columns else None

        if ret_1d_col:
            out = out.with_columns(pl.col(ret_1d_col).median().over(cross_keys).alias("sec_ret_1d_eq"))
        if ret_5d_col:
            out = out.with_columns(pl.col(ret_5d_col).median().over(cross_keys).alias("sec_ret_5d_eq"))

        if "sec_ret_1d_eq" in out.columns:
            out = out.with_columns(
                pl.col("sec_ret_1d_eq")
                .rolling_sum(window_size=20, min_periods=20)
                .over(sector_col)
                .alias("sec_mom_20"),
                pl.col("sec_ret_1d_eq").ewm_mean(span=5, min_periods=5).over(sector_col).alias("sec_ema_5"),
                pl.col("sec_ret_1d_eq").ewm_mean(span=20, min_periods=20).over(sector_col).alias("sec_ema_20"),
            )
        if {"sec_ema_5", "sec_ema_20"}.issubset(out.columns):
            out = out.with_columns(
                ((pl.col("sec_ema_5") - pl.col("sec_ema_20")) / (pl.col("sec_ema_20").abs() + EPS)).alias(
                    "sec_gap_5_20"
                )
            )

        if "sec_ret_1d_eq" in out.columns:
            out = out.with_columns(
                pl.col("sec_ret_1d_eq")
                .rolling_std(window_size=20, min_periods=20)
                .over(sector_col)
                .mul(252**0.5)
                .alias("sec_vol_20")
            )
        if "sec_vol_20" in out.columns:
            out = out.with_columns(
                pl.col("sec_vol_20")
                .rolling_mean(window_size=252, min_periods=252)
                .over(sector_col)
                .alias("_sec_vol_20_mean_252"),
                pl.col("sec_vol_20")
                .rolling_std(window_size=252, min_periods=252)
                .over(sector_col)
                .alias("_sec_vol_20_std_252"),
            )
            out = out.with_columns(
                ((pl.col("sec_vol_20") - pl.col("_sec_vol_20_mean_252")) / (pl.col("_sec_vol_20_std_252") + EPS)).alias(
                    "sec_vol_20_z"
                )
            )

        out = out.with_columns(
            pl.count().over(cross_keys).alias("sec_member_cnt"),
        )
        out = out.with_columns((pl.col("sec_member_cnt") < min_members).cast(pl.Int8).alias("sec_small_flag"))

        if {cfg.returns_5d, "sec_ret_5d_eq"}.issubset(out.columns):
            out = out.with_columns((pl.col(cfg.returns_5d) - pl.col("sec_ret_5d_eq")).alias("rel_to_sec_5d"))

        temp_cols = [c for c in out.columns if c.startswith("_sec_") or c.startswith("_cov_") or c.startswith("_var_")]
        if temp_cols:
            out = out.drop(temp_cols)
        return out
