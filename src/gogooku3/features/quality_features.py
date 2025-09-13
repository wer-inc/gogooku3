from __future__ import annotations

"""
Lightweight quality feature generator.

Attempts to use the restored peer feature extractor from src/features/peer_features.py
when available; otherwise returns the input DataFrame unchanged.

API compatible subset used by SafeTrainingPipeline:
- generate_quality_features(df: pandas.DataFrame) -> pandas.DataFrame
- validate_features(df: pandas.DataFrame) -> dict
"""

import importlib.util
from pathlib import Path
from typing import Any, Dict


class QualityFinancialFeaturesGenerator:
    def __init__(self, use_cross_sectional_quantiles: bool = True, sigma_threshold: float = 2.0):
        self.use_cross_sectional_quantiles = use_cross_sectional_quantiles
        self.sigma_threshold = sigma_threshold
        self._peer_extractor = None

        # Try to load PeerFeatureExtractor from src/features/peer_features.py if present
        repo_root = Path(__file__).resolve().parents[3]
        peer_path = repo_root / "src" / "features" / "peer_features.py"
        if peer_path.exists():
            spec = importlib.util.spec_from_file_location("peer_features", str(peer_path))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[attr-defined]
                if hasattr(mod, "PeerFeatureExtractor"):
                    self._peer_extractor = mod.PeerFeatureExtractor()

    def generate_quality_features(self, df):  # pandas.DataFrame
        if self._peer_extractor is None:
            return df
        try:
            return self._peer_extractor.add_peer_features(df, method="mixed", verbose=False)
        except Exception:
            # On any failure, return original df to keep pipeline robust
            return df

    def validate_features(self, df) -> Dict[str, Any]:
        # Minimal validation stub; pipeline expects keys below
        zero_var = []
        high_missing = []
        try:
            import pandas as pd  # noqa
            numeric = df.select_dtypes(include=["number"])  # type: ignore[attr-defined]
            var = numeric.var().fillna(0)
            zero_var = [c for c, v in var.items() if v == 0]
            missing = df.isna().mean()  # type: ignore[attr-defined]
            high_missing = [c for c, r in missing.items() if r > 0.5]
        except Exception:
            pass
        return {"zero_variance_features": zero_var, "high_missing_features": high_missing}

