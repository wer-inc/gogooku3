"""Advanced volatility feature generation."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt

import polars as pl

from ..utils.rolling import roll_mean_safe, roll_std_safe, roll_var_safe

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

        # Phase 2 Patch C: All rolling operations exclude current day
        exprs: list[pl.Expr] = []
        valid_windows: list[int] = []
        for win in cfg.windows:
            if win is None or win <= 1:
                continue
            win = int(win)
            valid_windows.append(win)
            k = 0.34 / (1.34 + (win + 1) / (win - 1))
            # Phase 2 Patch C: Use safe rolling operations (shift(1) before rolling)
            min_p = max(win // 2, 5)  # Require at least half the window or 5 obs
            var_u = roll_var_safe(pl.col("yz_u"), win, min_periods=min_p, by=code)
            var_d = roll_var_safe(pl.col("yz_d"), win, min_periods=min_p, by=code)
            var_c = roll_var_safe(pl.col("yz_c"), win, min_periods=min_p, by=code)
            yz_var = var_u + k * var_c + (1 - k) * var_d
            exprs.append(yz_var.clip(lower_bound=0.0).sqrt().mul(sqrt(252.0)).alias(f"yz_vol_{win}"))

            log_hl = ((pl.col(high) + EPS) / (pl.col(low) + EPS)).log()
            pk_var = roll_mean_safe(log_hl.pow(2.0), win, min_periods=min_p, by=code) / (4.0 * log(2.0))
            exprs.append(pk_var.clip(lower_bound=0.0).sqrt().mul(sqrt(252.0)).alias(f"pk_vol_{win}"))

            log_h_o = ((pl.col(high) + EPS) / (pl.col(open_col) + EPS)).log()
            log_h_c = ((pl.col(high) + EPS) / (pl.col(close) + EPS)).log()
            log_l_o = ((pl.col(low) + EPS) / (pl.col(open_col) + EPS)).log()
            log_l_c = ((pl.col(low) + EPS) / (pl.col(close) + EPS)).log()
            rs_term = (log_h_o * log_h_c) + (log_l_o * log_l_c)
            rs_var = roll_mean_safe(rs_term, win, min_periods=min_p, by=code)
            exprs.append(rs_var.clip(lower_bound=0.0).sqrt().mul(sqrt(252.0)).alias(f"rs_vol_{win}"))

        if exprs:
            x = x.with_columns(exprs)

        # Phase 2 Patch C: Volatility of volatility (VoV) also excludes current day
        vov_exprs: list[pl.Expr] = []
        for win in valid_windows:
            min_p = max(win // 2, 5)
            vov_exprs.append(
                roll_std_safe(pl.col(f"yz_vol_{win}"), win, min_periods=min_p, by=code).alias(f"vov_{win}")
            )
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
