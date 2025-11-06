"""Market index regime feature engineering (TOPIX, Nikkei, etc.)."""
from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from typing import Sequence

import polars as pl

from ...utils.rolling import roll_mean_safe, roll_std_safe, roll_var_safe

EPS = 1e-12
REALIZED_VOL_WINDOWS: tuple[int, ...] = (20, 60)
RET_HORIZONS: tuple[int, ...] = (1, 5, 20)
TREND_SHORT = 20
TREND_LONG = 100
VOL_REGIME_WINDOW = 60
FEATURE_ALLOWLIST: tuple[str, ...] = (
    "date",
    "code",
    "close",
    "r_prev_1d",
    "r_prev_5d",
    "r_prev_20d",
    "trend_gap_20_100",
    "z_close_20",
    "atr14",
    "natr14",
    "yz_vol_20",
    "yz_vol_60",
    "pk_vol_20",
    "pk_vol_60",
    "rs_vol_20",
    "rs_vol_60",
    "vol_z_20",
    "regime_score",
)


@dataclass
class IndexFeatureConfig:
    code_column: str = "code"
    date_column: str = "date"
    open_column: str = "open"
    high_column: str = "high"
    low_column: str = "low"
    close_column: str = "close"


class IndexFeatureEngineer:
    """Compute leak-safe, compact market-regime features for indices."""

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
        """Ensure canonical schema and ordering."""
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

    def _add_returns(self, df: pl.DataFrame) -> pl.DataFrame:
        cfg = self.config
        close = cfg.close_column
        code = cfg.code_column

        shifts: list[pl.Expr] = []
        for horizon in RET_HORIZONS:
            shifts.append(pl.col(close).shift(horizon).over(code).alias(f"_close_{horizon}d"))
        df = df.with_columns(shifts)

        returns: list[pl.Expr] = []
        for horizon in RET_HORIZONS:
            past_col = f"_close_{horizon}d"
            returns.append(((pl.col(close) / (pl.col(past_col) + EPS)) - 1.0).alias(f"r_prev_{horizon}d"))
        return df.with_columns(returns)

    def _add_atr(self, df: pl.DataFrame) -> pl.DataFrame:
        cfg = self.config
        high, low, close = cfg.high_column, cfg.low_column, cfg.close_column
        code, date = cfg.code_column, cfg.date_column

        if not {high, low, close}.issubset(df.columns):
            return df

        tr = pl.max_horizontal(
            pl.col(high) - pl.col(low),
            (pl.col(high) - pl.col(close).shift(1).over(code)).abs(),
            (pl.col(low) - pl.col(close).shift(1).over(code)).abs(),
        ).alias("_tr")
        df = df.with_columns(tr)

        df = df.with_columns(
            [
                roll_mean_safe(pl.col("_tr"), 14, min_periods=5, by=code).alias("atr14"),
                roll_mean_safe(pl.col("_tr") / (pl.col(close) + EPS), 14, min_periods=5, by=code).alias("natr14"),
            ]
        )

        halt = pl.date(2020, 10, 1)
        df = df.with_columns(
            pl.when(pl.col(date) == halt).then(None).otherwise(pl.col("atr14")).alias("atr14"),
            pl.when(pl.col(date) == halt).then(None).otherwise(pl.col("natr14")).alias("natr14"),
        )
        return df

    def _add_trend(self, df: pl.DataFrame) -> pl.DataFrame:
        cfg = self.config
        close = cfg.close_column
        code = cfg.code_column

        sma20 = roll_mean_safe(pl.col(close), TREND_SHORT, min_periods=10, by=code).alias("_sma20")
        sma100 = roll_mean_safe(pl.col(close), TREND_LONG, min_periods=20, by=code).alias("_sma100")
        df = df.with_columns([sma20, sma100])

        df = df.with_columns(
            ((pl.col("_sma20") / (pl.col("_sma100") + EPS)) - 1.0).alias("trend_gap_20_100"),
            (
                (pl.col(close) - pl.col("_sma20"))
                / (roll_std_safe(pl.col(close), TREND_SHORT, min_periods=10, by=code) + EPS)
            ).alias("z_close_20"),
        )
        return df

    def _add_realized_vol(self, df: pl.DataFrame) -> pl.DataFrame:
        cfg = self.config
        code = cfg.code_column
        open_col, high, low, close = (
            cfg.open_column,
            cfg.high_column,
            cfg.low_column,
            cfg.close_column,
        )
        df = df.with_columns(
            [
                (pl.col(open_col) / (pl.col(close).shift(1).over(code) + EPS)).log().alias("_yz_u"),
                (pl.col(close) / (pl.col(open_col) + EPS)).log().alias("_yz_d"),
                (pl.col(close) / (pl.col(close).shift(1).over(code) + EPS)).log().alias("_yz_c"),
            ]
        )

        realized_exprs: list[pl.Expr] = []
        for win in REALIZED_VOL_WINDOWS:
            if win <= 1:
                continue
            min_p = max(win // 2, 5)
            k = 0.34 / (1.34 + (win + 1) / (win - 1))
            var_u = roll_var_safe(pl.col("_yz_u"), win, min_periods=min_p, by=code)
            var_d = roll_var_safe(pl.col("_yz_d"), win, min_periods=min_p, by=code)
            var_c = roll_var_safe(pl.col("_yz_c"), win, min_periods=min_p, by=code)
            yz_var = var_u + k * var_c + (1 - k) * var_d
            realized_exprs.append(yz_var.clip(lower_bound=0.0).sqrt().mul(sqrt(252.0)).alias(f"yz_vol_{win}"))

            log_hl = ((pl.col(high) + EPS) / (pl.col(low) + EPS)).log()
            pk_var = roll_mean_safe(log_hl.pow(2.0), win, min_periods=min_p, by=code) / (4.0 * log(2.0))
            realized_exprs.append(pk_var.clip(lower_bound=0.0).sqrt().mul(sqrt(252.0)).alias(f"pk_vol_{win}"))

            log_h_o = ((pl.col(high) + EPS) / (pl.col(open_col) + EPS)).log()
            log_h_c = ((pl.col(high) + EPS) / (pl.col(close) + EPS)).log()
            log_l_o = ((pl.col(low) + EPS) / (pl.col(open_col) + EPS)).log()
            log_l_c = ((pl.col(low) + EPS) / (pl.col(close) + EPS)).log()
            rs_term = (log_h_o * log_h_c) + (log_l_o * log_l_c)
            rs_var = roll_mean_safe(rs_term, win, min_periods=min_p, by=code)
            realized_exprs.append(rs_var.clip(lower_bound=0.0).sqrt().mul(sqrt(252.0)).alias(f"rs_vol_{win}"))

        if realized_exprs:
            df = df.with_columns(realized_exprs)
        return df

    def _add_regime_score(self, df: pl.DataFrame) -> pl.DataFrame:
        code = self.config.code_column
        if "yz_vol_20" in df.columns:
            vol_z = (
                (pl.col("yz_vol_20") - roll_mean_safe(pl.col("yz_vol_20"), VOL_REGIME_WINDOW, by=code))
                / (roll_std_safe(pl.col("yz_vol_20"), VOL_REGIME_WINDOW, min_periods=15, by=code) + EPS)
            ).alias("vol_z_20")
            df = df.with_columns(vol_z)
        else:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias("vol_z_20"))

        df = df.with_columns(
            (pl.col("trend_gap_20_100").fill_null(0.0) - pl.col("vol_z_20").fill_null(0.0)).alias("regime_score")
        )
        return df

    def build_features(self, indices: pl.DataFrame) -> pl.DataFrame:
        """Generate the P0 market-regime block for a given index panel."""
        if indices.is_empty():
            return indices

        df = self.normalize_types(indices)
        df = self._add_returns(df)
        df = self._add_atr(df)
        df = self._add_trend(df)
        df = self._add_realized_vol(df)
        df = self._add_regime_score(df)

        drop_cols = [
            c
            for c in df.columns
            if c.startswith("_close_") or c.startswith("_yz_") or c in {"_tr", "_sma20", "_sma100"}
        ]
        df = df.drop(drop_cols, strict=False)

        keep = [col for col in FEATURE_ALLOWLIST if col in df.columns]
        return df.select(keep)
