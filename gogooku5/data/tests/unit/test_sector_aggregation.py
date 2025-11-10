from __future__ import annotations

from datetime import date

import polars as pl
from builder.features.core.sector.aggregation import SectorAggregationFeatures


def test_sector_aggregation_features_add_features() -> None:
    df = pl.DataFrame(
        {
            "code": ["A", "B", "A", "B"],
            "sector_code": ["SEC1", "SEC1", "SEC1", "SEC1"],
            "date": [date(2024, 1, 1), date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 2)],
            "ret_prev_1d": [0.01, 0.015, 0.02, 0.01],  # P0 FIX: Use ret_prev_* (backward-looking)
            "ret_prev_5d": [0.05, 0.04, 0.06, 0.03],  # P0 FIX: Use ret_prev_* (backward-looking)
        }
    )

    features = SectorAggregationFeatures()
    out = features.add_features(df)

    assert "sec_ret_1d_eq" in out.columns
    assert "sec_mom_20" in out.columns
    assert "sec_member_cnt" in out.columns
