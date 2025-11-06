from __future__ import annotations

from datetime import date, timedelta

import polars as pl
from builder.features.core.index.features import IndexFeatureEngineer
from pytest import approx


def test_index_feature_engineer_build_features() -> None:
    base = date(2024, 1, 1)
    horizon = 120
    dates = [base + timedelta(days=i) for i in range(horizon)]
    close = [100.0 + float(i) for i in range(horizon)]
    open_px = [c - 0.5 for c in close]
    high = [c + 0.8 for c in close]
    low = [c - 1.2 for c in close]

    df = pl.DataFrame(
        {
            "code": ["0000"] * horizon + ["0101"] * horizon,
            "date": dates + dates,
            "open": open_px + [p * 0.5 for p in open_px],
            "high": high + [p * 0.5 for p in high],
            "low": low + [p * 0.5 for p in low],
            "close": close + [p * 0.5 for p in close],
        }
    )

    engineer = IndexFeatureEngineer()
    out = engineer.build_features(df)

    expected_cols = {
        "date",
        "code",
        "close",
        "r_prev_1d",
        "r_prev_5d",
        "r_prev_20d",
        "trend_gap_20_100",
        "z_close_20",
        "atr14",
        "natr14",
        "yz_vol_20",
        "yz_vol_60",
        "pk_vol_20",
        "pk_vol_60",
        "rs_vol_20",
        "rs_vol_60",
        "vol_z_20",
        "regime_score",
    }
    assert set(out.columns) == expected_cols
    assert "idx_r_1d" not in out.columns

    topix = out.filter(pl.col("code") == "0000").sort("date")

    # First day has no prior return (left-closed)
    assert topix.select("r_prev_1d").item(0, 0) is None

    second_ret = topix.select("r_prev_1d").item(1, 0)
    expected_second = (close[1] / close[0]) - 1.0
    assert second_ret == approx(expected_second, abs=1e-12)

    # ATR and regime score become available once window is filled
    atr_sample = topix.select("atr14").item(40, 0)
    assert atr_sample is not None
    regime_sample = topix.select("regime_score").item(60, 0)
    assert regime_sample is not None
