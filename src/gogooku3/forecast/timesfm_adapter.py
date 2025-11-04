from __future__ import annotations

"""Lightweight TimesFM-style zero-shot adapter (with robust fallbacks).

This adapter exposes a simple API to get point and quantile forecasts for
single time series per id over multiple horizons. In environments without the
TimesFM package or network access, it falls back to a fast, deterministic
baseline (last-value / damped persistence with empirical residuals) so that
downstream wiring (Detect/Decide) can be exercised end-to-end.

Inputs (flat DataFrame):
  - id: str  (ticker)
  - ts: datetime-like (timestamp)
  - y: float (target)

Outputs (flat DataFrame):
  - id, ts, h, y_hat[, p10, p50, p90]

Notes:
  - This is intentionally minimal and dependency-light.
  - When real TimesFM is available, swap `predict_fn` to call it.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"])  # type: ignore[assignment]
    return out


@dataclass
class TimesFMAdapter:
    """Zero-shot style forecaster with a safe fallback.

    Parameters
    ----------
    horizons : Sequence[int]
        Horizons (days) to predict.
    context : int
        Context window length per id.
    quantiles : Sequence[float]
        Quantiles to emit; if empty, only point forecasts are returned.
    """

    horizons: Sequence[int]
    context: int = 256
    quantiles: Sequence[float] = (0.1, 0.5, 0.9)

    def predict(self, df_obs: pd.DataFrame) -> pd.DataFrame:
        """Produce flat forecasts for each id and horizon.

        Fallback baseline:
          - y_hat: damped persistence (0.95^h) * last_value
          - residuals: last N diffs to estimate scale
          - quantiles: symmetric around point using empirical MAD
        """
        if not {"id", "ts", "y"}.issubset(df_obs.columns):
            raise ValueError("df_obs must have columns: id, ts, y")

        df = _ensure_ts(df_obs)
        dfs: list[pd.DataFrame] = []

        for gid, g in df.sort_values(["id", "ts"]).groupby("id", sort=False):
            tail = g.tail(self.context)
            if tail.empty:
                continue
            last_ts = tail["ts"].max()
            last_y = float(tail["y"].iloc[-1])
            # Empirical residual scale via MAD of diffs
            diffs = np.diff(tail["y"].to_numpy())
            if diffs.size == 0:
                mad = 1e-6
            else:
                mad = np.median(np.abs(diffs - np.median(diffs))) * 1.4826 + 1e-6

            rows = []
            for h in self.horizons:
                # Simple damped persistence
                y_hat = (0.95 ** max(h, 0)) * last_y
                row = {
                    "id": gid,
                    "ts": last_ts,  # forecast origin timestamp
                    "h": int(h),
                    "y_hat": float(y_hat),
                }
                if self.quantiles:
                    # Symmetric quantiles around point forecast (approx)
                    z = np.array([-1.2816, 0.0, 1.2816])  # ~p10, p50, p90
                    q = y_hat + z * mad
                    # Map to requested quantiles by simple interpolation
                    rq = np.interp(self.quantiles, [0.1, 0.5, 0.9], q)
                    for p, v in zip(self.quantiles, rq):
                        row[f"p{int(p*100):02d}"] = float(v)
                rows.append(row)
            dfs.append(pd.DataFrame(rows))

        if not dfs:
            return pd.DataFrame(columns=["id", "ts", "h", "y_hat"])

        out = pd.concat(dfs, ignore_index=True)
        return out


def timesfm_predict(
    df_obs: pd.DataFrame,
    horizons: Iterable[int] = (1, 5, 10, 20, 30),
    context: int = 256,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
) -> pd.DataFrame:
    """Convenience function to run the adapter once."""
    adapter = TimesFMAdapter(horizons=list(horizons), context=context, quantiles=tuple(quantiles))
    return adapter.predict(df_obs)

