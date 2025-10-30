from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from builder.features.core.peer.features import PeerFeatureEngineer


def test_peer_feature_engineer_add_features() -> None:
    df = pl.DataFrame(
        {
            "code": ["A", "B", "C", "A", "B", "C"],
            "sector_code": ["S1", "S1", "S1", "S1", "S1", "S1"],
            "date": [
                date(2024, 1, 1),
                date(2024, 1, 1),
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 1, 2),
                date(2024, 1, 2),
            ],
            "close": [100.0, 102.0, 98.0, 101.0, 103.0, 99.0],
        }
    )

    engineer = PeerFeatureEngineer()
    out = engineer.add_features(df)

    assert "close_peer_mean" in out.columns
    first_day = out.filter(pl.col("date") == date(2024, 1, 1))
    mean_value = first_day.filter(pl.col("code") == "A").select("close_peer_mean").item(0, 0)
    assert mean_value == pytest.approx((102.0 + 98.0) / 2)
