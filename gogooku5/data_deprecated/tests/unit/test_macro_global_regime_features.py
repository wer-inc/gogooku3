from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest
from builder.features.macro.global_regime import prepare_vvmd_features


def _make_regime_frame(days: int = 12) -> pl.DataFrame:
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(days)]
    return pl.DataFrame(
        {
            "Date": dates,
            "spy_close": [100.0 + i for i in range(days)],
            "spy_open": [99.5 + i for i in range(days)],
            "spy_volume": [1_000_000.0 + 5_000 * i for i in range(days)],
            "qqq_close": [300.0 + 1.2 * i for i in range(days)],
            "qqq_volume": [800_000.0 + 4_000 * i for i in range(days)],
            "vix_close": [18.0 + 0.1 * i for i in range(days)],
            "dxy_close": [105.0 - 0.05 * i for i in range(days)],
            "btc_close": [30_000.0 + 50 * i for i in range(days)],
            "fx_usdjpy_close": [140.0 - 0.1 * i for i in range(days)],
            "credit_hyg_close": [75.0 + 0.02 * i for i in range(days)],
            "credit_lqd_close": [110.0 + 0.01 * i for i in range(days)],
            "rates_tlt_close": [90.0 - 0.05 * i for i in range(days)],
            "rates_ief_close": [95.0 - 0.02 * i for i in range(days)],
            "vix9d_close": [19.0 + 0.12 * i for i in range(days)],
            "vix3m_close": [22.0 + 0.08 * i for i in range(days)],
        }
    )


def test_prepare_vvmd_features_includes_cross_market_extensions() -> None:
    raw = _make_regime_frame()
    features = prepare_vvmd_features(raw)

    expected_columns = {
        "macro_vvmd_vrp_spy",
        "macro_vvmd_vrp_spy_z_252d",
        "macro_vvmd_credit_spread_ratio",
        "macro_vvmd_rates_term_ratio",
        "macro_vvmd_vix_term_slope",
        "macro_vvmd_vix_term_ratio",
        "macro_vvmd_spy_overnight_ret",
        "macro_vvmd_fx_usdjpy_ret_1d",
        "macro_vvmd_fx_usdjpy_z_20d",
    }
    missing = expected_columns - set(features.columns)
    assert not missing, f"Missing expected columns: {missing}"

    vrp_rows = features.filter(pl.col("macro_vvmd_vrp_spy").is_not_null())
    assert vrp_rows.height > 0, "VRP should produce non-null rows once sufficient history accrues"

    sample = vrp_rows.row(0, named=True)
    sample_date = sample["Date"]
    realized_vol = sample["macro_vvmd_vol_spy_rv20"]
    vix_value = raw.filter(pl.col("Date") == sample_date)["vix_close"][0]
    expected_vrp = (vix_value / 100.0) ** 2 - (realized_vol**2)
    assert sample["macro_vvmd_vrp_spy"] == pytest.approx(expected_vrp, rel=1e-6)

    fx_rows = features.filter(pl.col("macro_vvmd_fx_usdjpy_ret_1d").is_not_null())
    assert fx_rows.height > 0, "USDJPY returns should exist with rolling window satisfied"
