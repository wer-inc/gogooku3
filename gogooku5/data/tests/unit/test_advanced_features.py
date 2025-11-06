from __future__ import annotations

from datetime import date

import polars as pl
from builder.features.core.advanced import AdvancedFeatureEngineer
from polars.testing import assert_frame_equal


def test_advanced_feature_engineer_add_features() -> None:
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
            "volume": [1000, 1100, 1200, 800, 850, 900],
            "returns_1d": [0.01, 0.012, 0.011, 0.009, 0.008, 0.007],
            "returns_5d": [0.05, 0.06, 0.055, 0.03, 0.035, 0.04],
            "dollar_volume": [1e5, 1.1e5, 1.2e5, 4e4, 4.2e4, 4.4e4],
        }
    )

    engineer = AdvancedFeatureEngineer()
    out = engineer.add_features(df)

    assert "rsi_14" in out.columns
    assert "rsi_2" in out.columns  # Multi-period RSI support
    assert "macd_hist_slope" in out.columns


def test_advanced_features_unsorted_input_matches_sorted() -> None:
    base = pl.DataFrame(
        {
            "code": ["A", "A", "A", "B", "B", "B"],
            "date": [
                date(2024, 1, 3),
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 1),
            ],
            "close": [102.0, 100.0, 101.0, 51.0, 52.0, 50.0],
            "volume": [1200, 1000, 1100, 850, 900, 800],
            "returns_1d": [0.011, 0.01, 0.012, 0.008, 0.007, 0.009],
            "returns_5d": [0.055, 0.05, 0.06, 0.035, 0.04, 0.03],
            "dollar_volume": [1.2e5, 1e5, 1.1e5, 4.2e4, 4.4e4, 4e4],
        }
    )

    engineer = AdvancedFeatureEngineer()
    sorted_result = engineer.add_features(base.sort(["code", "date"]))
    unsorted_result = engineer.add_features(base)

    joined = unsorted_result.sort(["code", "date"]).select(sorted_result.columns)

    assert_frame_equal(sorted_result.sort(["code", "date"]), joined, check_dtypes=True)
