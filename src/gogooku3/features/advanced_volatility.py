from __future__ import annotations

"""
Advanced volatility features:
 - Yang–Zhang volatility (rolling) for equities
 - Volatility-of-Volatility (rolling std of YZ volatility)

Implements leak-safety by shifting EOD features to the next business day
using the equity panel's unique business dates (no external calendar required).
"""

from math import sqrt
from typing import Iterable

import polars as pl


def _next_bday_expr_from_equity(quotes: pl.DataFrame) -> pl.Expr:
    dates = quotes.select("Date").unique().sort("Date")["Date"].to_list()
    next_map = {}
    for i in range(len(dates) - 1):
        next_map[dates[i]] = dates[i + 1]
    # default: +1 day duration; Polars will handle Date + duration
    # Note: Polars 1.x uses replace() instead of map_dict()
    return pl.col("Date").replace(next_map, default=pl.col("Date") + pl.duration(days=1))


def add_advanced_vol_block(
    df: pl.DataFrame,
    *,
    windows: Iterable[int] = (20, 60),
    shift_to_next_day: bool = True,
) -> pl.DataFrame:
    """Attach Yang–Zhang volatility and volatility-of-volatility for equities.

    Args:
        df: Equity panel (must have Code, Date, Open, High, Low, Close)
        windows: Rolling windows for YZ vol
        shift_to_next_day: If True, shift EOD features to next business day

    Returns:
        DataFrame with yz_vol_{win} and vov_{win} columns added (leak-safe if shift enabled)
    """
    required = {"Code", "Date", "Open", "High", "Low", "Close"}
    if not required.issubset(set(df.columns)):
        return df

    x = df.sort(["Code", "Date"]).with_columns(
        [
            # Overnight return u_t = ln(Open_t / Close_{t-1})
            (pl.col("Open") / pl.col("Close").shift(1).over("Code")).log().alias("yz_u"),
            # Intraday return d_t = ln(Close_t / Open_t)
            (pl.col("Close") / pl.col("Open")).log().alias("yz_d"),
            # Close-to-close return c_t = ln(Close_t / Close_{t-1})
            (pl.col("Close") / pl.col("Close").shift(1).over("Code")).log().alias("yz_c"),
        ]
    )

    cols_to_attach: list[pl.Expr] = []
    for win in windows:
        if win is None or int(win) <= 1:
            continue
        win = int(win)
        k = 0.34 / (1.34 + (win + 1) / (win - 1))
        var_u = pl.col("yz_u").rolling_var(win).over("Code")
        var_d = pl.col("yz_d").rolling_var(win).over("Code")
        var_c = pl.col("yz_c").rolling_var(win).over("Code")
        yz_var = var_u + k * var_c + (1 - k) * var_d
        yz_vol = (yz_var.clip(lower_bound=0.0)).sqrt().alias(f"yz_vol_{win}")
        # Volatility-of-Volatility as rolling std of yz_vol over the same window
        vov = pl.col(f"yz_vol_{win}").rolling_std(win).over("Code").alias(f"vov_{win}")
        cols_to_attach.append(yz_vol)
        cols_to_attach.append(vov)

    x = x.with_columns(cols_to_attach)

    # Keep only columns to join back
    yz_cols = [c for c in x.columns if c.startswith("yz_vol_") or c.startswith("vov_")]
    yz = x.select(["Code", "Date"] + yz_cols)

    if shift_to_next_day:
        next_bd = _next_bday_expr_from_equity(df)
        yz = yz.with_columns([next_bd.alias("Date")])

    # Join back to main frame (left join on Code, Date)
    out = df.join(yz, on=["Code", "Date"], how="left")
    # Validity column per a main representative window (first in list)
    main_win = int(next(iter(windows))) if windows else 20
    if f"yz_vol_{main_win}" in out.columns:
        out = out.with_columns(
            [
                pl.when(pl.col(f"yz_vol_{main_win}").is_null()).then(0).otherwise(1).cast(pl.Int8).alias("is_adv_vol_valid")
            ]
        )
    return out

