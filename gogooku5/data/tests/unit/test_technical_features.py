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
        }
    )

    engineer = TechnicalFeatureEngineer()
    out = engineer.add_features(df)

    assert "kama_10_2_30" in out.columns
