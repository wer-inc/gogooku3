from __future__ import annotations

"""Thin placeholder adapter for a TFT-like forecaster.

In this codebase, the primary complex model is ATFT-GAT-FAN. Until the
full TFT training/inference path is integrated, this adapter provides a
compatible interface and falls back to the simple zero-shot baseline so the
end-to-end pipeline wiring (Forecast→Detect→Decide) remains runnable.
"""

from dataclasses import dataclass, field
from typing import Iterable, Sequence, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

from .timesfm_adapter import TimesFMAdapter


@dataclass
class TFTAdapter:
    """Feature-rich forecaster interface (placeholder).

    Parameters
    ----------
    horizons : Sequence[int]
        Horizons (days) to predict.
    quantiles : Sequence[float]
        Quantiles to emit; if empty, only point forecasts are returned.
    """

    horizons: Sequence[int]
    quantiles: Sequence[float] = (0.1, 0.5, 0.9)
    _models: Dict[int, Dict[float, QuantileRegressor]] = field(default_factory=dict, init=False)
    _feature_cols: list[str] = field(default_factory=list, init=False)

    def fit(
        self,
        df_obs: pd.DataFrame,
        df_known_future: pd.DataFrame | None = None,
        df_static: pd.DataFrame | None = None,
    ) -> "TFTAdapter":
        """Train lightweight per-horizon quantile regressors.

        - Pooled across ids, using current-time features to predict y at t+h.
        - Avoids leakage by shifting target forward by h per id.
        - If data is insufficient, falls back to zero-shot baseline at predict().
        """
        req = {"id", "ts", "y"}
        if not req.issubset(df_obs.columns):
            raise ValueError("df_obs must contain id, ts, y")
        d = df_obs.copy()
        d["ts"] = pd.to_datetime(d["ts"])  # type: ignore[assignment]
        d.sort_values(["id", "ts"], inplace=True)

        # Feature columns: everything except id, ts, targets to be created
        base_cols = [c for c in d.columns if c not in ("id", "ts")]
        # Ensure 'y' is included as a feature if no other feature exists (fallback)
        self._feature_cols = [c for c in base_cols]
        if len(self._feature_cols) == 1 and self._feature_cols[0] == "y":
            # still fine; model will use current y to forecast
            pass

        self._models = {}
        # Build design per horizon by shifting targets within each id
        for h in self.horizons:
            parts = []
            for gid, g in d.groupby("id", sort=False):
                g = g.copy()
                g[f"target_{h}"] = g["y"].shift(-int(h))
                parts.append(g)
            dd = pd.concat(parts, ignore_index=True)
            dd = dd.dropna(subset=[f"target_{h}"])  # keep rows with future label
            if dd.empty:
                continue
            X = dd[self._feature_cols].astype(float).to_numpy()
            y = dd[f"target_{h}"].astype(float).to_numpy()
            models_h: Dict[float, QuantileRegressor] = {}
            for q in self.quantiles:
                # Use L2 regularization small alpha for stability; solver requires scipy
                model = QuantileRegressor(quantile=float(q), alpha=1e-4, solver="highs")
                try:
                    model.fit(X, y)
                    models_h[q] = model
                except Exception:
                    # if fit fails, skip this quantile
                    continue
            if models_h:
                self._models[int(h)] = models_h
        return self

    def predict(
        self,
        df_obs: pd.DataFrame,
        df_known_future: pd.DataFrame | None = None,
        df_static: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Predict quantiles for each horizon at the latest timestamp per id.

        Fallback to TimesFM-style baseline if models are unavailable.
        """
        if not self._models:
            return TimesFMAdapter(horizons=list(self.horizons), context=256, quantiles=tuple(self.quantiles)).predict(df_obs)

        req = {"id", "ts"}
        if not req.issubset(df_obs.columns):
            raise ValueError("df_obs must contain id, ts")
        d = df_obs.copy()
        d["ts"] = pd.to_datetime(d["ts"])  # type: ignore[assignment]
        d.sort_values(["id", "ts"], inplace=True)

        rows = []
        for gid, g in d.groupby("id", sort=False):
            last = g.tail(1)
            x = last[self._feature_cols].astype(float).to_numpy()
            origin = last["ts"].iloc[0]
            for h in self.horizons:
                qmap = self._models.get(int(h))
                if not qmap:
                    continue
                preds = {}
                for q in sorted(qmap.keys()):
                    try:
                        preds[q] = float(qmap[q].predict(x)[0])
                    except Exception:
                        continue
                if not preds:
                    continue
                # Ensure monotonic quantiles
                qs = sorted(preds.items())
                vals = np.maximum.accumulate([v for _, v in qs])
                for i, (qq, _) in enumerate(qs):
                    preds[qq] = float(vals[i])
                row = {"id": gid, "ts": origin, "h": int(h)}
                # p50 as y_hat if present else mean
                y_hat = preds.get(0.5, float(np.mean(list(preds.values()))))
                row["y_hat"] = y_hat
                for qq, v in preds.items():
                    row[f"p{int(qq*100):02d}"] = v
                rows.append(row)
        if not rows:
            return TimesFMAdapter(horizons=list(self.horizons), context=256, quantiles=tuple(self.quantiles)).predict(df_obs)
        return pd.DataFrame(rows)
