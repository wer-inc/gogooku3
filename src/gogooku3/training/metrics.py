from __future__ import annotations

import numpy as np
import polars as pl


def rank_ic(df: pl.DataFrame, *, pred_col: str, target_col: str, date_col: str = "Date") -> float:
    """Compute mean Spearman rank correlation across dates (RankIC)."""
    def _per_date(cdf: pl.DataFrame) -> float:
        a = cdf.select(pred_col).to_series().to_numpy()
        b = cdf.select(target_col).to_series().to_numpy()
        # Convert to ranks; handle ties via average rank
        ra = _rankdata(a)
        rb = _rankdata(b)
        if ra.std() == 0 or rb.std() == 0:
            return 0.0
        return float(np.corrcoef(ra, rb)[0, 1])

    vals = [
        _per_date(g)
        for _, g in df.group_by(date_col, maintain_order=True)
    ]
    return float(np.nanmean(np.asarray(vals, dtype=float)))


def sharpe_ratio(returns: np.ndarray, *, risk_free: float = 0.0, eps: float = 1e-12) -> float:
    """Compute simple Sharpe ratio for a vector of returns."""
    ex = returns - risk_free
    return float(np.mean(ex) / (np.std(ex) + eps))


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = x.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    avg = sums / np.maximum(counts, 1)
    return avg[inv]


__all__ = ["rank_ic", "sharpe_ratio"]

