from __future__ import annotations

from collections.abc import Iterable

import polars as pl


def add_ratio_adv_z(
    df: pl.DataFrame,
    value_col: str,
    adv_col: str,
    *,
    code_col: str = "Code",
    date_col: str = "Date",
    z_win: int = 260,
    prefix: str | None = None,
    ensure_sorted: bool = True,
) -> pl.DataFrame:
    """Add ADV ratio and rolling Z-score columns for a value column.

    - Adds `{prefix}_to_adv20` as value/ADV with small epsilon for stability.
    - Adds `{prefix}_z{z_win}` as per-code rolling Z-score.

    Parameters
    ----------
    value_col : str
        Column to normalize.
    adv_col : str
        Average dollar volume column to scale by.
    code_col : str
        Instrument identifier for per-series rolling stats.
    z_win : int
        Window length for rolling mean/std used in Z-score.
    prefix : str | None
        Prefix used to name the new columns; defaults to `value_col`.
    """
    pre = prefix or value_col
    # Ensure deterministic, as-of ordering per instrument
    base = df.sort([code_col, date_col]) if ensure_sorted and {code_col, date_col}.issubset(df.columns) else df
    # Ratio to ADV
    to_adv = (pl.col(value_col) / (pl.col(adv_col) + 1e-12)).alias(f"{pre}_to_adv20")

    # Rolling statistics per instrument
    mean = (
        pl.col(value_col)
        .rolling_mean(window_size=z_win, min_periods=max(1, z_win // 2))
        .over(code_col)
    )
    std = (
        pl.col(value_col)
        .rolling_std(window_size=z_win, min_periods=max(2, z_win // 2))
        .over(code_col)
    )
    z = ((pl.col(value_col) - mean) / (std + 1e-12)).alias(f"{pre}_z{z_win}")
    return base.with_columns([to_adv, z])


def add_multi_ratio_adv_z(
    df: pl.DataFrame, items: Iterable[tuple[str, str, str | None]]
) -> pl.DataFrame:
    """Apply `add_ratio_adv_z` across multiple (value_col, adv_col, prefix).

    Example
    -------
    >>> items = [("margin_long_tot", "dollar_volume_ma20", "margin_long")]
    >>> df = add_multi_ratio_adv_z(df, items)
    """
    out = df
    for value_col, adv_col, prefix in items:
        out = add_ratio_adv_z(out, value_col=value_col, adv_col=adv_col, prefix=prefix)
    return out


__all__ = ["add_ratio_adv_z", "add_multi_ratio_adv_z"]
