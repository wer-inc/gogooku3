from __future__ import annotations

"""Calibration and simple ensembling utilities for anomaly scores."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray, a: float = 1.0, b: float = 0.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(a * x + b)))


@dataclass
class PlattParams:
    a: float = 1.0
    b: float = 0.0


def calibrate_sigmoid(
    scores: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, PlattParams]:
    """Calibrate anomaly scores to probabilities using Platt scaling.

    If labels are None, returns identity parameters.
    labels are expected to be binary point labels (0/1) aligned to scores.
    """
    scores = scores.astype(float)
    if labels is None or labels.size == 0:
        return _sigmoid(scores), PlattParams(1.0, 0.0)

    y = labels.astype(float).reshape(-1)
    x = scores.reshape(-1)
    # Fit logistic regression with Newton update (2 params)
    a, b = 1.0, 0.0
    for _ in range(25):
        p = _sigmoid(a * x + b)
        # gradient
        da = np.sum((y - p) * x)
        db = np.sum(y - p)
        # Hessian terms
        w = p * (1 - p)
        haa = np.sum(w * x * x) + 1e-8
        hbb = np.sum(w) + 1e-8
        hab = np.sum(w * x)  # symmetric
        # Solve 2x2: H * d = g
        det = haa * hbb - hab * hab
        if abs(det) < 1e-12:
            break
        da_upd = ( hbb * da - hab * db) / det
        db_upd = (-hab * da + haa * db) / det
        a += da_upd
        b += db_upd
        if max(abs(da_upd), abs(db_upd)) < 1e-6:
            break
    return _sigmoid(a * x + b), PlattParams(a=float(a), b=float(b))


def stack_and_score(
    scores: list[pd.DataFrame],
    id_col: str = "id",
    ts_col: str = "ts",
    score_col: str = "score",
    calibrate: bool = False,
    labels: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Stack multiple component scores and produce a calibrated ensemble.

    - Joins on id+ts and averages the component scores.
    - Optionally fits Platt scaling if point labels are provided as DataFrame
      with columns [id, ts, label]. Range labels should be converted upstream.
    """
    if not scores:
        return pd.DataFrame(columns=[id_col, ts_col, score_col])

    base = scores[0][[id_col, ts_col]].copy()
    base[score_col] = 0.0
    k = 0
    for s in scores:
        base = base.merge(s[[id_col, ts_col, score_col]], on=[id_col, ts_col], how="inner", suffixes=(None, None))
        base[score_col] = base[score_col] + s[score_col].values  # type: ignore[index]
        k += 1
    base[score_col] = base[score_col] / max(k, 1)

    if calibrate and labels is not None and {id_col, ts_col, "label"}.issubset(labels.columns):
        df = base.merge(labels[[id_col, ts_col, "label"]], on=[id_col, ts_col], how="left")
        probs, _ = calibrate_sigmoid(df[score_col].fillna(0).to_numpy(), df["label"].fillna(0).to_numpy())
        base[score_col] = probs

    # Clip to [0,1]
    base[score_col] = base[score_col].clip(0.0, 1.0)
    return base


class DetectionEnsemble:
    """Ensemble detection engine for multiple anomaly detection methods."""

    def __init__(self, methods: list[str] | None = None):
        """Initialize detection ensemble.

        Args:
            methods: List of detection methods to use
        """
        self.methods = methods or ["residual", "change_point", "spectral"]

    def detect(self, data: pd.DataFrame, symbol: str = "unknown", threshold: float = 0.25) -> dict:
        """Run ensemble detection on time series data.

        Args:
            data: DataFrame with time series data (expects columns: timestamp, value)
            symbol: Symbol identifier
            threshold: Anomaly detection threshold

        Returns:
            Detection results dictionary
        """
        from .residual import ResidualAnomalyDetector
        from .change_point import ChangePointDetector
        from .spectral_residual import SpectralResidualDetector
        from .ranges import score_to_ranges, VUS_PR

        # Initialize detection methods
        detectors = {}
        if "residual" in self.methods:
            detectors["residual"] = ResidualAnomalyDetector(window_size=30)
        if "change_point" in self.methods:
            detectors["change_point"] = ChangePointDetector(min_size=5, jump=1)
        if "spectral" in self.methods:
            detectors["spectral"] = SpectralResidualDetector(window_size=10, k=3)

        # Prepare data format (standardize column names)
        if "timestamp" not in data.columns and "ts" in data.columns:
            data = data.rename(columns={"ts": "timestamp"})
        if "value" not in data.columns and "y" in data.columns:
            data = data.rename(columns={"y": "value"})

        # Run individual detection methods
        method_scores = []
        for method_name, detector in detectors.items():
            try:
                # Create detection input format: id, ts, y
                detect_df = pd.DataFrame({
                    "id": symbol,
                    "ts": data["timestamp"],
                    "value": data["value"]
                })

                # Run detection (mock for now - actual implementations need refinement)
                scores = np.random.beta(2, 5, len(data))  # Mock anomaly scores
                score_df = pd.DataFrame({
                    "id": symbol,
                    "ts": data["timestamp"],
                    "score": scores
                })
                method_scores.append(score_df)
            except Exception as e:
                # Skip failed methods
                continue

        if not method_scores:
            # Fallback if all methods fail
            return {
                "anomaly_score": 0.0,
                "vus_pr_score": 0.0,
                "is_anomaly": False,
                "detected_ranges": [],
                "methods": self.methods,
                "symbol": symbol
            }

        # Ensemble: Average scores across methods
        ensemble_df = method_scores[0].copy()
        if len(method_scores) > 1:
            # Stack and average scores from all methods
            all_scores = stack_and_score(method_scores, calibrate=False)
            ensemble_df["score"] = all_scores["score"]

        # Convert point scores to ranges
        ranges = score_to_ranges(ensemble_df, threshold=threshold, min_len=1)

        # Calculate VUS-PR (mock labels for now)
        vus_pr_evaluator = VUS_PR(min_iou=0.25)
        mock_labels = []  # TODO: Integrate with actual labels
        vus_pr_result = vus_pr_evaluator.evaluate(ranges, mock_labels)

        # Extract key metrics
        anomaly_score = ensemble_df["score"].max() if not ensemble_df.empty else 0.0
        is_anomaly = anomaly_score >= threshold

        return {
            "anomaly_score": float(anomaly_score),
            "vus_pr_score": vus_pr_result["vus_pr"],
            "is_anomaly": bool(is_anomaly),
            "detected_ranges": [
                {"start": int((r.start - pd.Timestamp.min).days), "end": int((r.end - pd.Timestamp.min).days)}
                for r in ranges
            ],
            "methods": self.methods,
            "symbol": symbol
        }

