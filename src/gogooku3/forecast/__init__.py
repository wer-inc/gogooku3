"""Forecast layer adapters and utilities.

- TimesFMAdapter: zero-shot style interface (fallback to naive if not available)
- TFTAdapter: placeholder for feature-rich forecaster (delegates to existing models where possible)
- ChampionChallengerFramework: Model deployment and A/B testing framework

All functions operate on flat pandas DataFrames with columns:
  id (str), ts (datetime-like), y (float), and optional feature columns.

Outputs are flat DataFrames with columns:
  id, ts, h (horizon in days), y_hat, and optional quantiles p10/p50/p90.
"""

from .timesfm_adapter import TimesFMAdapter, timesfm_predict
from .tft_adapter import TFTAdapter
from .champion_challenger import ChampionChallengerFramework, ModelMetrics, PerformanceTracker

__all__ = [
    "TimesFMAdapter",
    "timesfm_predict",
    "TFTAdapter",
    "ChampionChallengerFramework",
    "ModelMetrics",
    "PerformanceTracker",
]

