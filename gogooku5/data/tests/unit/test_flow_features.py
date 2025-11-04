from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from builder.features.core.flow.enhanced import FlowFeatureEngineer


def test_flow_features_rescale_partial_weeks_and_dedup() -> None:
    base = pl.DataFrame(
        {
            "code": ["AAA", "AAA", "AAA"],
            "date": [date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)],
        }
    )

    flows = pl.DataFrame(
        {
            "PublishedDate": ["2024-01-04", "2024-01-05"],
            "StartDate": ["2024-01-01", "2024-01-01"],
            "EndDate": ["2024-01-05", "2024-01-05"],
            "ForeignersPurchases": [500.0, 1000.0],
            "ForeignersSales": [300.0, 400.0],
            "IndividualsPurchases": [None, 200.0],
            "IndividualsSales": [None, 150.0],
        }
    )

    engineer = FlowFeatureEngineer()
    out = engineer.add_features(base, flows)

    assert "institutional_accumulation" in out.columns
    release = out.filter(pl.col("date") == date(2024, 1, 5))
    assert release.select("institutional_accumulation").item(0, 0) == pytest.approx(600.0)
    assert release.select("foreign_sentiment").item(0, 0) == pytest.approx((600.0) / (1000.0 + 400.0))

    pre_release = out.filter(pl.col("date") < date(2024, 1, 5))
    assert pre_release.select("institutional_accumulation").drop_nulls().height == 0
