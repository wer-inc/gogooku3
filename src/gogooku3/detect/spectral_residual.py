from __future__ import annotations

"""Spectral Residual based anomaly scoring (fast, dependency-light).

Reference: SR (Spectral Residual) saliency maps used in SR-CNN.
Implementation adapted to 1D time series and normalized to [0, 1].
"""

import numpy as np
import pandas as pd


def _spectral_residual_1d(x: np.ndarray, avg_window: int = 5) -> np.ndarray:
    n = x.size
    if n == 0:
        return np.zeros(0, dtype=float)
    # Avoid negative/zero values for log spectrum; shift baseline
    x_centered = x - np.median(x)
    eps = 1e-8
    X = np.fft.fft(x_centered)
    mag = np.abs(X) + eps
    log_mag = np.log(mag)
    # moving average on log magnitude
    k = avg_window
    kernel = np.ones(k) / k
    avg = np.convolve(log_mag, kernel, mode="same")
    spectral_residual = np.exp(log_mag - avg)
    # reconstruct with original phase
    Y = spectral_residual * np.exp(1j * np.angle(X))
    y = np.fft.ifft(Y).real
    saliency = np.abs(y)
    # normalize to [0,1]
    saliency -= saliency.min()
    denom = saliency.max() - saliency.min() + eps
    return saliency / denom


def spectral_residual_score(
    df: pd.DataFrame,
    value_col: str = "y",
    id_col: str = "id",
    ts_col: str = "ts",
    avg_window: int = 5,
) -> pd.DataFrame:
    """Compute spectral residual saliency per id and timestamp.

    Returns flat DataFrame: id, ts, score in [0,1].
    """
    if not {id_col, ts_col, value_col}.issubset(df.columns):
        raise ValueError(f"df must contain {id_col},{ts_col},{value_col}")
    d = df.copy()
    d[ts_col] = pd.to_datetime(d[ts_col])
    d.sort_values([id_col, ts_col], inplace=True)
    out = []
    for gid, g in d.groupby(id_col, sort=False):
        x = g[value_col].astype(float).to_numpy()
        s = _spectral_residual_1d(x, avg_window)
        out.append(pd.DataFrame({id_col: gid, ts_col: g[ts_col].values, "score": s}))
    if not out:
        return pd.DataFrame(columns=[id_col, ts_col, "score"])
    return pd.concat(out, ignore_index=True)

