"""
Phase 2 Patch C: Safe rolling statistics utilities (left-closed, exclude current day).

All rolling operations must exclude the current day to prevent look-ahead bias.
Use these utilities instead of direct rolling operations.
"""

from __future__ import annotations

import polars as pl


def roll_mean_safe(
    expr: pl.Expr,
    window: int,
    min_periods: int | None = None,
    by: str | None = "code",
) -> pl.Expr:
    """
    Rolling mean excluding current day (left-closed).

    Args:
        expr: Column expression to compute rolling mean
        window: Window size in days
        min_periods: Minimum number of observations (default: window)
        by: Group by column (default: "code"). If None, no grouping applied.

    Returns:
        Expression with rolling mean (current day excluded)

    Example:
        df.with_columns(roll_mean_safe(pl.col("volume"), 20).alias("vol_ma_20"))
    """
    min_p = min_periods if min_periods is not None else window
    if by is None:
        # No grouping - simple time series rolling (e.g., TOPIX)
        return expr.shift(1).rolling_mean(window_size=window, min_periods=min_p)
    else:
        # Grouped rolling (e.g., by stock code)
        return expr.shift(1).over(by).rolling_mean(window_size=window, min_periods=min_p)


def roll_std_safe(
    expr: pl.Expr,
    window: int,
    min_periods: int | None = None,
    by: str | None = "code",
) -> pl.Expr:
    """
    Rolling standard deviation excluding current day (left-closed).

    Args:
        expr: Column expression to compute rolling std
        window: Window size in days
        min_periods: Minimum number of observations (default: window)
        by: Group by column (default: "code"). If None, no grouping applied.

    Returns:
        Expression with rolling std (current day excluded)
    """
    min_p = min_periods if min_periods is not None else window
    if by is None:
        return expr.shift(1).rolling_std(window_size=window, min_periods=min_p)
    else:
        return expr.shift(1).over(by).rolling_std(window_size=window, min_periods=min_p)


def roll_var_safe(
    expr: pl.Expr,
    window: int,
    min_periods: int | None = None,
    by: str | None = "code",
) -> pl.Expr:
    """
    Rolling variance excluding current day (left-closed).

    Args:
        expr: Column expression to compute rolling var
        window: Window size in days
        min_periods: Minimum number of observations (default: window)
        by: Group by column (default: "code"). If None, no grouping applied.

    Returns:
        Expression with rolling variance (current day excluded)
    """
    min_p = min_periods if min_periods is not None else window
    if by is None:
        return expr.shift(1).rolling_var(window_size=window, min_periods=min_p)
    else:
        return expr.shift(1).over(by).rolling_var(window_size=window, min_periods=min_p)


def roll_sum_safe(
    expr: pl.Expr,
    window: int,
    min_periods: int | None = None,
    by: str | None = "code",
) -> pl.Expr:
    """
    Rolling sum excluding current day (left-closed).

    Args:
        expr: Column expression to compute rolling sum
        window: Window size in days
        min_periods: Minimum number of observations (default: window)
        by: Group by column (default: "code"). If None, no grouping applied.

    Returns:
        Expression with rolling sum (current day excluded)
    """
    min_p = min_periods if min_periods is not None else window
    if by is None:
        return expr.shift(1).rolling_sum(window_size=window, min_periods=min_p)
    else:
        return expr.shift(1).over(by).rolling_sum(window_size=window, min_periods=min_p)


def roll_quantile_safe(
    expr: pl.Expr,
    window: int,
    quantile: float,
    min_periods: int | None = None,
    by: str | None = "code",
) -> pl.Expr:
    """
    Rolling quantile excluding current day (left-closed).

    Args:
        expr: Column expression to compute rolling quantile
        window: Window size in days
        quantile: Quantile value (0.0 to 1.0)
        min_periods: Minimum number of observations (default: window)
        by: Group by column (default: "code"). If None, no grouping applied.

    Returns:
        Expression with rolling quantile (current day excluded)
    """
    min_p = min_periods if min_periods is not None else window
    if by is None:
        return expr.shift(1).rolling_quantile(quantile=quantile, window_size=window, min_periods=min_p)
    else:
        return expr.shift(1).over(by).rolling_quantile(quantile=quantile, window_size=window, min_periods=min_p)


def ewm_mean_safe(
    expr: pl.Expr,
    span: int,
    by: str | None = "code",
) -> pl.Expr:
    """
    Exponential weighted moving average excluding current day.

    Args:
        expr: Column expression to compute EWM mean
        span: Span parameter (half-life in days)
        by: Group by column (default: "code"). If None, no grouping applied.

    Returns:
        Expression with EWM mean (current day excluded)

    Note:
        Polars ewm_mean doesn't have min_periods parameter.
        First value will be NaN.
    """
    alpha = 2.0 / (span + 1.0)
    if by is None:
        return expr.shift(1).ewm_mean(alpha=alpha, adjust=False)
    else:
        return expr.shift(1).over(by).ewm_mean(alpha=alpha, adjust=False)
