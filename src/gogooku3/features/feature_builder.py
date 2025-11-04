from __future__ import annotations

"""Default feature builder for TFT training.

Adds per-id technical series from y, and cross-sectional ranks per date.
"""

from collections.abc import Sequence

import pandas as pd

from .feature_params import FeatureParams
from .tech_indicators import fractional_diff, kama, rolling_quantiles, vidya


def add_default_features(
    df: pd.DataFrame,
    id_col: str = "id",
    ts_col: str = "ts",
    y_col: str = "y",
    params: FeatureParams | None = None,
) -> pd.DataFrame:
    p = params or FeatureParams()
    d = df.copy()
    d[ts_col] = pd.to_datetime(d[ts_col])  # type: ignore[assignment]
    d.sort_values([id_col, ts_col], inplace=True)
    parts = []
    for gid, g in d.groupby(id_col, sort=False):
        g = g.copy()
        y = g[y_col].astype(float)
        # KAMA: support multiple configs, else single
        if p.kama_set:
            for (kw, kf, ks) in p.kama_set:
                g[f"kama_{kw}_{kf}_{ks}"] = kama(y, window=int(kw), fast=int(kf), slow=int(ks))
        else:
            g[f"kama_{p.kama_window}_{p.kama_fast}_{p.kama_slow}"] = kama(
                y, window=p.kama_window, fast=p.kama_fast, slow=p.kama_slow
            )
        # VIDYA: support list windows, else single
        if p.vidya_windows:
            for vw in p.vidya_windows:
                g[f"vidya_{int(vw)}"] = vidya(y, window=int(vw))
        else:
            g[f"vidya_{p.vidya_window}"] = vidya(y, window=p.vidya_window)
        g[f"fdiff_{str(p.fd_d).replace('.','p')}_{p.fd_window}"] = fractional_diff(
            y, d=p.fd_d, window=p.fd_window
        )
        # Rolling quantiles: multiple windows or single
        if p.rq_windows:
            for w in p.rq_windows:
                rq = rolling_quantiles(y, window=int(w), quants=tuple(p.rq_quantiles))
                rq = rq.add_prefix(f"rq{int(w)}_")
                g = pd.concat([g, rq.reset_index(drop=True)], axis=1)
        else:
            rq = rolling_quantiles(y, window=p.rq_window, quants=tuple(p.rq_quantiles))
            g = pd.concat([g, rq.reset_index(drop=True)], axis=1)
        # Rolling std set
        for w in p.roll_std_windows:
            g[f"roll_std_{int(w)}"] = y.rolling(window=int(w), min_periods=int(w)).std()
        parts.append(g)
    d2 = pd.concat(parts, ignore_index=True)
    # Cross-sectional features per date
    d2.sort_values([ts_col, id_col], inplace=True)
    def _cs(df_day: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
        out = df_day.copy()
        for c in cols:
            if c not in out.columns:
                continue
            v = out[c].astype(float)
            # rank in [0,1]
            out[f"cs_rank_{c}"] = v.rank(method="average", na_option="keep") / v.count()
            # zscore
            mu = v.mean(); sd = v.std(ddof=0)
            out[f"cs_z_{c}"] = (v - mu) / (sd + 1e-12)
        return out
    d3 = d2.groupby(ts_col, group_keys=False).apply(lambda x: _cs(x, list(p.cs_features)))
    return d3
