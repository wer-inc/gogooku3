from __future__ import annotations

from datetime import date

import polars as pl
from builder.features.core.volatility import AdvancedVolatilityFeatures


def test_advanced_volatility_features_add_features() -> None:
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
            "open": [100.0, 101.0, 102.0, 50.0, 51.0, 52.0],
            "high": [102.0, 103.0, 104.0, 51.0, 52.0, 53.0],
            "low": [99.0, 100.0, 101.0, 49.0, 50.0, 51.0],
            "close": [101.0, 102.0, 103.0, 50.5, 51.5, 52.5],
        }
    )

    features = AdvancedVolatilityFeatures()
    out = features.add_features(df)

    assert "yz_vol_20" in out.columns
    assert "vov_20" in out.columns
