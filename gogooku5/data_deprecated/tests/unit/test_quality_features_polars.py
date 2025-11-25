from __future__ import annotations

from datetime import date

import polars as pl
from builder.features.core.quality_features_polars import (
    QualityFinancialFeaturesGeneratorPolars,
)


def test_quality_features_generator_adds_expected_columns() -> None:
    df = pl.DataFrame(
        {
            "code": ["1301", "1301", "1305", "1305"],
            "sector_code": ["FOOD", "FOOD", "ETF", "ETF"],
            "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 1), date(2024, 1, 2)],
            "close": [100.0, 101.0, 55.0, 56.0],
            "open": [99.0, 100.0, 54.0, 55.0],
        }
    )

    generator = QualityFinancialFeaturesGeneratorPolars()
    out = generator.generate_quality_features(df)

    assert "close_cs_rank" in out.columns
    assert "close_roll_mean_20d" in out.columns
    assert "close_sector_mean" in out.columns
