from __future__ import annotations

"""
Lightweight baseline model wrapper.

Provides a minimal interface used by SafeTrainingPipeline:
- __init__(prediction_horizons, embargo_days, normalize_features, verbose)
- fit(df: pandas.DataFrame)
- evaluate_performance() -> dict
- get_results_summary() -> dict
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class LightGBMFinancialBaseline:
    def __init__(
        self,
        prediction_horizons: Optional[List[int]] = None,
        embargo_days: int = 20,
        normalize_features: bool = True,
        verbose: bool = True,
        **kwargs,
    ):
        self.prediction_horizons = prediction_horizons or [1]
        self.embargo_days = embargo_days
        self.normalize_features = normalize_features
        self.verbose = verbose
        self._trained = False

    def fit(self, df: pd.DataFrame) -> None:
        # Minimal placeholder: mark as trained. Real training can be added later.
        self._trained = True

    def evaluate_performance(self) -> Dict[str, Dict[str, float]]:
        # Compute naive baseline metrics: correlation between simple target and its shifted version
        perf: Dict[str, Dict[str, float]] = {}
        for h in self.prediction_horizons:
            # Pick target column by common aliases
            target = None
            for c in (f"returns_{h}d", f"ret_{h}d", f"feat_ret_{h}d", f"target_{h}d"):
                if c in self._last_df.columns:  # type: ignore[attr-defined]
                    target = c
                    break
            if target is None and hasattr(self, "_last_df"):
                # Fallback: any column starting with returns_/ret_/feat_ret_/target_
                cand = [c for c in self._last_df.columns if c.startswith(("returns_", "ret_", "feat_ret_", "target_"))]  # type: ignore[attr-defined]
                target = cand[0] if cand else None
            if target is None:
                perf[f"{h}d"] = {"mean_ic": 0.0, "std_ic": 0.0, "mean_rank_ic": 0.0, "std_rank_ic": 0.0}
                continue
            s = self._last_df[target].astype(float)  # type: ignore[attr-defined]
            pred = s.shift(1).fillna(0.0)
            ic = float(np.corrcoef(s.values, pred.values)[0, 1]) if s.std() and pred.std() else 0.0
            # Rank IC approximation
            rank_ic = float(pd.Series(s).rank().corr(pd.Series(pred).rank())) if s.size > 5 else 0.0
            perf[f"{h}d"] = {"mean_ic": ic, "std_ic": 0.0, "mean_rank_ic": rank_ic, "std_rank_ic": 0.0}
        return perf

    def get_results_summary(self) -> Dict[str, float]:
        return {"trained": float(self._trained)}

    # Store last df for evaluation
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == "_last_df":
            pass

    def __getattribute__(self, name):
        return super().__getattribute__(name)

    # Hook to keep a reference to df passed into fit
    def fit(self, df: pd.DataFrame) -> None:  # type: ignore[override]
        self._last_df = df.copy()
        self._trained = True

