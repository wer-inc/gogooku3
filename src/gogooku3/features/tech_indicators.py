from __future__ import annotations

"""Lightweight technical indicators and transforms.

Includes:
- KAMA (Kaufman's Adaptive Moving Average)
- VIDYA (Variable Index Dynamic Average)
- Fractional differencing
- Rolling quantiles
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def kama(series: pd.Series, window: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """Compute Kaufman's Adaptive Moving Average (KAMA).

    ER = |price_t - price_{t-w}| / sum_{i=t-w+1..t} |price_i - price_{i-1}|
    SC = (ER*(2/(fast+1) - 2/(slow+1)) + 2/(slow+1))^2
    KAMA_t = KAMA_{t-1} + SC*(price_t - KAMA_{t-1})
    """
    x = series.astype(float).to_numpy()
    n = x.size
    out = np.full(n, np.nan, dtype=float)
    if n == 0 or window < 1:
        return pd.Series(out, index=series.index)
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    kama_prev = x[0]
    out[0] = kama_prev
    for t in range(1, n):
        if t < window:
            out[t] = np.nan
            kama_prev = x[t]
            continue
        change = abs(x[t] - x[t - window])
        volatility = np.sum(np.abs(np.diff(x[t - window : t + 1]))) + 1e-12
        er = change / volatility
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama_prev = kama_prev + sc * (x[t] - kama_prev)
        out[t] = kama_prev
    return pd.Series(out, index=series.index)


def vidya(series: pd.Series, window: int = 14) -> pd.Series:
    """Compute VIDYA using CMO-based adaptive smoothing.

    alpha_t = |CMO_t|/100 * 2/(window+1)
    VIDYA_t = VIDYA_{t-1} + alpha_t * (price_t - VIDYA_{t-1})
    """
    x = series.astype(float).to_numpy()
    n = x.size
    out = np.full(n, np.nan, dtype=float)
    if n == 0 or window < 2:
        return pd.Series(out, index=series.index)
    # CMO = 100*(sum(up) - sum(down)) / (sum(up)+sum(down))
    def cmo(arr: np.ndarray) -> float:
        diff = np.diff(arr)
        up = np.sum(diff[diff > 0])
        down = np.sum(-diff[diff < 0])
        denom = up + down + 1e-12
        return 100.0 * (up - down) / denom

    v_prev = x[0]
    out[0] = v_prev
    k = 2.0 / (window + 1.0)
    for t in range(1, n):
        if t < window:
            out[t] = np.nan
            v_prev = x[t]
            continue
        c = abs(cmo(x[t - window : t + 1])) / 100.0
        alpha = c * k
        v_prev = v_prev + alpha * (x[t] - v_prev)
        out[t] = v_prev
    return pd.Series(out, index=series.index)


def fractional_diff(series: pd.Series, d: float = 0.4, window: int = 100) -> pd.Series:
    """Fractional differencing with finite window.

    Weights via recursive relation: w0=1; w_k = -w_{k-1} * (d - k + 1)/k
    y_t = sum_{k=0..min(t,window-1)} w_k * x_{t-k}
    """
    x = series.astype(float).to_numpy()
    n = x.size
    w = np.zeros(window, dtype=float)
    w[0] = 1.0
    for k in range(1, window):
        w[k] = -w[k - 1] * (d - (k - 1)) / k
    out = np.full(n, np.nan, dtype=float)
    for t in range(n):
        kmax = min(t + 1, window)
        out[t] = float(np.dot(w[:kmax], x[t : t - kmax : -1]))
    return pd.Series(out, index=series.index)


def rolling_quantiles(series: pd.Series, window: int = 63, quants: Sequence[float] = (0.1, 0.5, 0.9)) -> pd.DataFrame:
    r = series.rolling(window=window, min_periods=window)
    cols = {}
    for q in quants:
        cols[f"rq_{int(q*100):02d}"] = r.quantile(q)
    return pd.DataFrame(cols, index=series.index)

