from __future__ import annotations

import logging
from collections.abc import Iterable

import polars as pl

LOGGER = logging.getLogger(__name__)


def _is_numeric_dtype(dtype: pl.DataType) -> bool:
    """Check if a Polars dtype is numeric (compatible with newer Polars versions)."""
    return dtype.is_numeric()


def add_cross_sectional_zscores(
    frame: pl.DataFrame,
    columns: Iterable[str],
    *,
    date_col: str,
    suffix: str = "_cs_z",
    clip_sigma: float | None = 5.0,
    epsilon: float = 1e-8,
) -> pl.DataFrame:
    """Append cross-sectional Z-score columns for the specified features.

    Each column ``c`` gains an additional ``c{suffix}`` column computed as
    ``(c - mean(c | date)) / std(c | date)``.  Zero-variance days fall back to
    a denominator of ``epsilon`` to avoid NaNs.
    """

    if not columns:
        return frame

    schema = frame.schema
    available = [c for c in columns if c in schema]
    missing = [c for c in columns if c not in schema]
    non_numeric = [c for c in available if not _is_numeric_dtype(schema[c])]
    numeric_cols = [c for c in available if _is_numeric_dtype(schema[c])]

    if missing:
        LOGGER.debug("Skipping z-score for missing columns: %s", missing)
    if non_numeric:
        LOGGER.warning("Skipping z-score for non-numeric columns: %s", non_numeric)

    if not numeric_cols:
        return frame

    exprs = []
    for col in numeric_cols:
        mean_expr = pl.col(col).mean().over(date_col)
        std_expr = pl.col(col).std().over(date_col)
        denominator = pl.when(std_expr.abs() < epsilon).then(1.0).otherwise(std_expr)
        z_expr = ((pl.col(col) - mean_expr) / denominator).alias(f"{col}{suffix}")
        exprs.append(z_expr)

    result = frame.with_columns(exprs)

    fill_exprs = []
    for col in numeric_cols:
        name = f"{col}{suffix}"
        expr = pl.col(name).fill_nan(0.0).fill_null(0.0)
        if clip_sigma is not None:
            expr = expr.clip(-clip_sigma, clip_sigma)
        fill_exprs.append(expr.alias(name))

    result = result.with_columns(fill_exprs)
    return result
