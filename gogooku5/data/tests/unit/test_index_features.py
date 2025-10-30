from __future__ import annotations

from datetime import date

import polars as pl
from builder.features.core.index.features import IndexFeatureEngineer


def test_index_feature_engineer_build_features() -> None:
    df = pl.DataFrame(
        {
            "code": ["0000", "0000", "1300", "1300"],
            "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 1), date(2024, 1, 2)],
            "open": [100.0, 101.0, 50.0, 51.0],
            "high": [102.0, 103.0, 52.0, 53.0],
            "low": [99.0, 100.0, 49.0, 50.0],
            "close": [101.0, 102.0, 51.0, 52.0],
        }
    )

    engineer = IndexFeatureEngineer()
    out = engineer.build_features(df)

    assert "idx_r_1d" in out.columns
    assert "idx_atr14" in out.columns
    assert out.filter(pl.col("code") == "0000").select("idx_r_1d").item(1, 0) is not None
