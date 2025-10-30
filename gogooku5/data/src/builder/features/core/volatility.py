"""Advanced volatility feature generation."""
from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt

import polars as pl

EPS = 1e-12


@dataclass
class AdvancedVolatilityConfig:
    code_column: str = "code"
    date_column: str = "date"
    open_column: str = "open"
    high_column: str = "high"
    low_column: str = "low"
    close_column: str = "close"
    windows: tuple[int, ...] = (20, 60)
    shift_to_next_day: bool = True


class AdvancedVolatilityFeatures:
    def __init__(self, config: AdvancedVolatilityConfig | None = None) -> None:
        self.config = config or AdvancedVolatilityConfig()

    def add_features(self, df: pl.DataFrame) -> pl.DataFrame:
        cfg = self.config
        required = {
            cfg.code_column,
            cfg.date_column,
            cfg.open_column,
            cfg.high_column,
            cfg.low_column,
            cfg.close_column,
        }
        if df.is_empty() or not required.issubset(df.columns):
            return df

        code = cfg.code_column
        date = cfg.date_column
        open_col = cfg.open_column
        high = cfg.high_column
        low = cfg.low_column
        close = cfg.close_column

        x = df.sort([code, date]).with_columns(
            [
                (pl.col(open_col) / pl.col(close).shift(1).over(code)).log().alias("yz_u"),
                (pl.col(close) / pl.col(open_col)).log().alias("yz_d"),
                (pl.col(close) / pl.col(close).shift(1).over(code)).log().alias("yz_c"),
            ]
        )

        exprs: list[pl.Expr] = []
        valid_windows: list[int] = []
        for win in cfg.windows:
            if win is None or win <= 1:
                continue
            win = int(win)
            valid_windows.append(win)
            k = 0.34 / (1.34 + (win + 1) / (win - 1))
            var_u = pl.col("yz_u").rolling_var(win).over(code)
            var_d = pl.col("yz_d").rolling_var(win).over(code)
            var_c = pl.col("yz_c").rolling_var(win).over(code)
            yz_var = var_u + k * var_c + (1 - k) * var_d
            exprs.append(yz_var.clip(lower_bound=0.0).sqrt().alias(f"yz_vol_{win}"))

            log_hl = ((pl.col(high) + EPS) / (pl.col(low) + EPS)).log()
            pk_var = log_hl.pow(2.0).rolling_mean(win, min_periods=win).over(code) / (4.0 * log(2.0))
            exprs.append(pk_var.clip(lower_bound=0.0).sqrt().mul(sqrt(252.0)).alias(f"pk_vol_{win}"))

            log_h_o = ((pl.col(high) + EPS) / (pl.col(open_col) + EPS)).log()
            log_h_c = ((pl.col(high) + EPS) / (pl.col(close) + EPS)).log()
            log_l_o = ((pl.col(low) + EPS) / (pl.col(open_col) + EPS)).log()
            log_l_c = ((pl.col(low) + EPS) / (pl.col(close) + EPS)).log()
            rs_term = (log_h_o * log_h_c) + (log_l_o * log_l_c)
            rs_var = rs_term.rolling_mean(win, min_periods=win).over(code)
            exprs.append(rs_var.clip(lower_bound=0.0).sqrt().mul(sqrt(252.0)).alias(f"rs_vol_{win}"))

        if exprs:
            x = x.with_columns(exprs)

        vov_exprs: list[pl.Expr] = []
        for win in valid_windows:
            vov_exprs.append(pl.col(f"yz_vol_{win}").rolling_std(win, min_periods=win).over(code).alias(f"vov_{win}"))
        if vov_exprs:
            x = x.with_columns(vov_exprs)

        vol_cols = [c for c in x.columns if c.startswith(("yz_vol_", "pk_vol_", "rs_vol_", "vov_"))]
        if not vol_cols:
            return df

        vol = x.select([code, date] + vol_cols)

        if cfg.shift_to_next_day:
            dates = df.select(date).unique().sort(date)[date].to_list()
            next_map = {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}
            vol = vol.with_columns(
                pl.col(date).replace(next_map, default=pl.col(date) + pl.duration(days=1)).alias(date)
            )

        out = df.join(vol, on=[code, date], how="left")
        main_win = valid_windows[0] if valid_windows else 20
        if f"yz_vol_{main_win}" in out.columns:
            out = out.with_columns(
                pl.when(pl.col(f"yz_vol_{main_win}").is_null())
                .then(0)
                .otherwise(1)
                .cast(pl.Int8)
                .alias("is_adv_vol_valid")
            )
        return out
