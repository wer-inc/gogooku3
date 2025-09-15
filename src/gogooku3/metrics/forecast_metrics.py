from __future__ import annotations

"""Lightweight forecast metrics: sMAPE, MAE, and WQL (pinball)."""

import numpy as np
import pandas as pd


def smape(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    denom = (np.abs(y) + np.abs(yhat))
    return float(np.mean(2.0 * np.abs(y - yhat) / np.clip(denom, 1e-12, None)))


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.mean(np.abs(y - yhat)))


def pinball_loss(y: np.ndarray, qhat: np.ndarray, q: float) -> float:
    y = np.asarray(y, dtype=float)
    qhat = np.asarray(qhat, dtype=float)
    diff = y - qhat
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))


def weighted_quantile_loss(y: np.ndarray, preds: pd.DataFrame) -> float:
    """Compute WQL from predicted quantile columns present in `preds`.

    Supports columns like p05, p10, p50, p90. Each available quantile is
    equally weighted. If none present, returns NaN.
    """
    cols = [c for c in preds.columns if c.startswith("p") and c[1:].isdigit()]
    if not cols:
        return float("nan")
    losses = []
    for c in cols:
        q = int(c[1:]) / 100.0
        losses.append(pinball_loss(y, preds[c].to_numpy(), q))
    return float(np.mean(losses))

