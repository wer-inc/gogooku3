from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from builder.features.core.margin.daily import MarginDailyFeatureEngineer


def test_margin_daily_feature_engineer() -> None:
    df = pl.DataFrame(
        {
            "code": ["1301", "1301", "1305", "1305"],
            "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 1), date(2024, 1, 2)],
            "margin_balance": [100000.0, 100200.0, 50000.0, 50010.0],
            "short_balance": [50000.0, 51000.0, 20000.0, 19900.0],
        }
    )

    engineer = MarginDailyFeatureEngineer()
    out = engineer.build_features(df)

    assert "margin_net" in out.columns
    net = out.filter(pl.col("code") == "1301").select("margin_net").item(1, 0)
    assert net == pytest.approx(49200.0)
    assert "margin_long_z20" in out.columns
