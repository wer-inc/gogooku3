from __future__ import annotations

from collections.abc import Iterable

import polars as pl


def winsorize(df: pl.DataFrame, cols: Iterable[str], k: float = 5.0) -> pl.DataFrame:
    """Winsorize columns at mean ± k·std, in-place by default.

    Designed for numeric columns with heavy tails. This function keeps the
    original column names (no extra copies) to avoid expanding the feature
    space while stabilizing extreme values.
    """
    out = df
    for c in cols:
        # compute stats once from the column to avoid group-wise leakage; for CS-Z use cs_standardize
        stats = out.select(mu=pl.col(c).mean(), sd=pl.col(c).std()).row(0)
        mu = float(stats[0]) if stats[0] is not None else 0.0
        sd = float(stats[1]) if stats[1] is not None else 0.0
        lo = mu - k * sd
        hi = mu + k * sd
        out = out.with_columns(pl.col(c).clip(lo, hi).alias(c))
    return out


def fit_winsor_stats(
    df_train: pl.DataFrame, cols: Iterable[str], k: float = 5.0
) -> dict[str, tuple[float, float]]:
    """Fit global (train-only) winsor thresholds per column.

    Returns a mapping col -> (lo, hi) computed from train distribution.
    """
    stats: dict[str, tuple[float, float]] = {}
    for c in cols:
        s = df_train.select(pl.col(c)).to_series()
        mu = float(s.mean() or 0.0)
        sd = float(s.std() or 0.0)
        lo, hi = mu - k * sd, mu + k * sd
        stats[c] = (lo, hi)
    return stats


def transform_winsor(
    df: pl.DataFrame, stats: dict[str, tuple[float, float]]
) -> pl.DataFrame:
    """Apply pre-fit winsor thresholds to a dataframe (no leakage)."""
    out = df
    for c, (lo, hi) in stats.items():
        if c in out.columns:
            out = out.with_columns(pl.col(c).clip(lo, hi).alias(c))
    return out


__all__ = ["winsorize", "fit_winsor_stats", "transform_winsor"]
