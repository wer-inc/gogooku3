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


def test_peer_features_handles_single_member_groups() -> None:
    df = pl.DataFrame(
        {
            "code": ["A", "A"],
            "sector_code": ["S1", "S1"],
            "date": [date(2024, 1, 1), date(2024, 1, 2)],
            "close": [100.0, 101.0],
        }
    )

    engineer = PeerFeatureEngineer()
    out = engineer.add_features(df)

    mean_nulls = out.select("close_peer_mean").null_count().item(0, 0)
    ratio_nulls = out.select("close_peer_ratio").null_count().item(0, 0)
    assert mean_nulls == out.height
    assert ratio_nulls == out.height


def test_peer_features_peer_std_excludes_self() -> None:
    df = pl.DataFrame(
        {
            "code": ["A", "B", "A", "B"],
            "sector_code": ["S1", "S1", "S1", "S1"],
            "date": [
                date(2024, 1, 1),
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 1, 2),
            ],
            "close": [100.0, 102.0, 101.0, 99.0],
        }
    )

    engineer = PeerFeatureEngineer()
    out = engineer.add_features(df)

    sample = out.filter((pl.col("date") == date(2024, 1, 1)) & (pl.col("code") == "A"))
    peer_std = sample.select("close_peer_std").item(0, 0)
    assert peer_std is None

    day_two = out.filter(pl.col("date") == date(2024, 1, 2))
    peer_mean_a = day_two.filter(pl.col("code") == "A").select("close_peer_mean").item(0, 0)
    assert peer_mean_a == pytest.approx(99.0)
