from __future__ import annotations

"""Optuna-based HPO for the minimal TFT surrogate.

Tunes QuantileRegressor alpha and selected feature parameters to minimize
average WQL across horizons using Purged CV in train_tft_quantile.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import optuna
import pandas as pd

from gogooku3.training.tft_trainer import TFTTrainerConfig, train_tft_quantile
from gogooku3.features.feature_params import FeatureParams


@dataclass
class HPOConfig:
    horizons: Sequence[int] = (1, 5, 10, 20)
    n_splits: int = 3
    embargo_days: int = 20
    trials: int = 20


def run_tft_hpo(
    df_obs: pd.DataFrame,
    df_known_future: pd.DataFrame | None = None,
    df_static: pd.DataFrame | None = None,
    config: HPOConfig | None = None,
) -> dict:
    cfg = config or HPOConfig()

    def objective(trial: optuna.Trial) -> float:
        # Sample feature params and quantile alpha
        kama_w = trial.suggest_int("kama_window", 8, 30)
        vidya_w = trial.suggest_int("vidya_window", 8, 30)
        fd_d = trial.suggest_float("fd_d", 0.2, 0.8)
        rq_w = trial.suggest_int("rq_window", 20, 126)
        # Build config
        fparams = FeatureParams(
            kama_window=kama_w,
            vidya_window=vidya_w,
            fd_d=fd_d,
            rq_window=rq_w,
        )
        tcfg = TFTTrainerConfig(
            horizons=cfg.horizons,
            embargo_days=cfg.embargo_days,
            n_splits=cfg.n_splits,
            features=fparams,
        )
        out = train_tft_quantile(df_obs, df_known_future, df_static, tcfg)
        metrics = out.get("metrics", {})
        vals = [float(v) for k, v in metrics.items() if k.startswith("WQL_h")]
        if not vals:
            return float("inf")
        return float(np.mean(vals))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=cfg.trials, show_progress_bar=False)
    best = study.best_trial
    return {
        "best_value": best.value,
        "best_params": best.params,
        "n_trials": cfg.trials,
    }

