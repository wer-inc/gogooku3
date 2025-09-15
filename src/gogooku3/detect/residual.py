from __future__ import annotations

"""Residual-based anomaly scoring with robust rolling normalization."""

from typing import Iterable

import numpy as np
import pandas as pd


def robust_scale_qmad(series: pd.Series, window: int = 63) -> pd.Series:
    """Return robust z-scores using rolling median and MAD (Q-MAD).

    z_t = (x_t - median(x_{t-w+1:t})) / (1.4826 * MAD(x_{t-w+1:t}))

    Parameters
    ----------
    series : pd.Series
        Input values (per id, ordered by time).
    window : int
        Rolling window length.
    """
    x = series.astype(float).to_numpy()
    n = x.size
    if n == 0:
        return pd.Series(dtype=float)

    z = np.zeros(n, dtype=float)
    for i in range(n):
        j0 = max(0, i - window + 1)
        chunk = x[j0 : i + 1]
        med = np.median(chunk)
        mad = np.median(np.abs(chunk - med)) * 1.4826 + 1e-9
        z[i] = (x[i] - med) / mad
    return pd.Series(z, index=series.index)


def residual_q_score(
    df_obs: pd.DataFrame,
    df_fcst: pd.DataFrame,
    horizon: int = 1,
    window: int = 63,
    id_col: str = "id",
    ts_col: str = "ts",
    y_col: str = "y",
    yhat_col: str = "y_hat",
) -> pd.DataFrame:
    """Compute residual-based robust anomaly scores per id and timestamp.

    Returns a flat DataFrame: id, ts, score in [0,1] (via |z|-score logistic).
    """
    if not {id_col, ts_col, y_col}.issubset(df_obs.columns):
        raise ValueError("df_obs must contain id, ts, y")
    if not {id_col, ts_col, "h", yhat_col}.issubset(df_fcst.columns):
        raise ValueError("df_fcst must contain id, ts, h, y_hat")

    df_o = df_obs.copy()
    df_f = df_fcst[df_fcst["h"] == horizon].copy()
    df_o[ts_col] = pd.to_datetime(df_o[ts_col])
    df_f[ts_col] = pd.to_datetime(df_f[ts_col])

    # Align by id+ts (forecast origin matching observed ts)
    m = pd.merge(df_o[[id_col, ts_col, y_col]], df_f[[id_col, ts_col, yhat_col]], on=[id_col, ts_col], how="inner")
    m.sort_values([id_col, ts_col], inplace=True)

    out = []
    for gid, g in m.groupby(id_col, sort=False):
        resid = g[y_col] - g[yhat_col]
        z = robust_scale_qmad(resid, window=window).abs()
        # squash to [0,1] via logistic on |z|
        prob = 1.0 / (1.0 + np.exp(-(z - 2.0)))
        out.append(pd.DataFrame({id_col: gid, ts_col: g[ts_col].values, "score": prob.values}))
    if not out:
        return pd.DataFrame(columns=[id_col, ts_col, "score"])
    return pd.concat(out, ignore_index=True)

