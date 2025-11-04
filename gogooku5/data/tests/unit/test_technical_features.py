from __future__ import annotations

from datetime import date

import polars as pl
from builder.features.core.technical import TechnicalFeatureEngineer


def test_technical_feature_engineer_add_features() -> None:
    df = pl.DataFrame(
        {
            "code": ["A", "A", "A", "B", "B", "B"],
            "date": [
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 1, 3),
            ],
            "close": [100.0, 101.0, 102.0, 50.0, 51.0, 52.0],
            "open": [99.5, 100.5, 101.0, 49.5, 50.5, 51.0],
            "high": [101.0, 102.5, 103.0, 51.5, 52.0, 53.0],
            "low": [99.0, 100.0, 100.5, 49.0, 50.0, 51.0],
            "volume": [1000, 1100, 1200, 800, 850, 900],
            "returns_1d": [0.01, 0.0099, 0.0098, 0.02, 0.0196, 0.0192],
        }
    )

    engineer = TechnicalFeatureEngineer()
    out = engineer.add_features(df)

    assert "kama_10_2_30" in out.columns
    assert "log_returns_1d" in out.columns
    assert "feat_ret_5d" in out.columns
    assert "sma_20" in out.columns
    assert "ema_60" in out.columns
    assert "ma_gap_5_20" in out.columns
    assert "volume_ratio_5" in out.columns
    assert "atr_14" in out.columns
