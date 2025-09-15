from __future__ import annotations

"""Simple change-point scoring via windowed mean-shift statistic.

For each timestamp t, compute the absolute difference between the mean in the
left and right windows, scaled by pooled standard deviation, then squash to [0,1].
This captures abrupt mean shifts without external dependencies.
"""

import numpy as np
import pandas as pd


def _mean_shift_score(x: np.ndarray, w: int) -> np.ndarray:
    n = x.size
    s = np.zeros(n, dtype=float)
    if n == 0:
        return s
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    cumsq = np.cumsum(np.insert(x * x, 0, 0.0))
    for i in range(n):
        l0 = max(0, i - w)
        r1 = min(n, i + w + 1)
        # left: [l0, i), right: (i, r1)
        ln = max(0, i - l0)
        rn = max(0, r1 - (i + 1))
        if ln < 2 or rn < 2:
            s[i] = 0.0
            continue
        lsum = cumsum[i] - cumsum[l0]
        rsum = cumsum[r1] - cumsum[i + 1]
        lmean = lsum / ln
        rmean = rsum / rn

        lss = cumsq[i] - cumsq[l0]
        rss = cumsq[r1] - cumsq[i + 1]
        lvar = max(0.0, (lss / ln) - lmean * lmean)
        rvar = max(0.0, (rss / rn) - rmean * rmean)
        # pooled std estimate
        denom = np.sqrt(0.5 * (lvar + rvar)) + 1e-9
        s[i] = abs(lmean - rmean) / denom
    # squash to [0, 1]
    return 1.0 / (1.0 + np.exp(-(s - 2.0)))


def change_point_score(
    df: pd.DataFrame,
    value_col: str = "y",
    id_col: str = "id",
    ts_col: str = "ts",
    window: int = 20,
) -> pd.DataFrame:
    """Compute change-point score per id and timestamp.

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
        s = _mean_shift_score(x, window)
        out.append(pd.DataFrame({id_col: gid, ts_col: g[ts_col].values, "score": s}))
    if not out:
        return pd.DataFrame(columns=[id_col, ts_col, "score"])
    return pd.concat(out, ignore_index=True)

