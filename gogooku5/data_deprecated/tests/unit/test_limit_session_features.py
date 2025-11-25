from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from builder.features.events import LimitEventFeatureEngineer
from builder.features.session import SessionFeatureEngineer


def test_limit_event_features_basic() -> None:
    df = pl.DataFrame(
        {
            "code": ["0001"] * 5,
            "date": [date(2024, 1, i) for i in range(1, 6)],
            "upper_limit": [0, 1, 0, 0, 0],
            "lower_limit": [0, 0, 0, 1, 0],
            "adjustmentclose": [100.0, 110.0, 108.0, 90.0, 92.0],
            "adjustmenthigh": [101.0, 110.0, 109.0, 91.0, 93.0],
            "adjustmentlow": [99.0, 109.0, 107.0, 90.0, 91.0],
        }
    ).with_columns(pl.col("date").cast(pl.Date))

    engineered = LimitEventFeatureEngineer().add_features(df)
    rows = engineered.sort(["code", "date"]).to_dicts()

    # Row 0: no history → Null roll sums, Null days_since_limit
    assert rows[0]["limit_up_flag"] == 0
    assert rows[0]["limit_down_flag"] == 0
    assert rows[0]["limit_any_flag"] == 0
    assert rows[0]["limit_up_5d"] is None
    assert rows[0]["limit_down_5d"] is None
    assert rows[0]["days_since_limit"] is None
    assert rows[0]["price_locked_flag"] == 0

    # Row 1: limit up event with price lock
    assert rows[1]["limit_up_flag"] == 1
    assert rows[1]["limit_down_flag"] == 0
    assert rows[1]["limit_any_flag"] == 1
    assert rows[1]["price_locked_flag"] == 1
    assert rows[1]["days_since_limit"] == 0

    # Row 2: roll sum still None (window=5) but days_since_limit increments
    assert rows[2]["limit_up_flag"] == 0
    assert rows[2]["limit_up_5d"] is None
    assert rows[2]["days_since_limit"] == 1

    # Row 3: limit down event resets days_since_limit and price lock triggers
    assert rows[3]["limit_down_flag"] == 1
    assert rows[3]["limit_any_flag"] == 1
    assert rows[3]["price_locked_flag"] == 1
    assert rows[3]["days_since_limit"] == 0

    # Row 4: distance since last event increments
    assert rows[4]["days_since_limit"] == 1


def test_session_features_eod_shift() -> None:
    df = pl.DataFrame(
        {
            "code": ["0001"] * 5,
            "date": [date(2024, 1, i) for i in range(1, 6)],
            "morning_open": [100.0, 102.0, 101.0, 103.0, 104.0],
            "morning_high": [101.0, 103.0, 102.0, 104.0, 105.0],
            "morning_low": [99.0, 101.0, 100.0, 102.0, 103.0],
            "morning_close": [100.5, 102.5, 101.2, 103.5, 104.5],
            "morning_volume": [1000.0, 1200.0, 1100.0, 1300.0, 1400.0],
            "morning_upper_limit": [0, 1, 0, 0, 0],
            "morning_lower_limit": [0, 0, 0, 1, 0],
            "afternoon_open": [101.0, 103.0, 102.0, 104.0, 105.0],
            "afternoon_high": [102.0, 104.0, 103.0, 105.0, 106.0],
            "afternoon_low": [100.0, 102.0, 101.0, 103.0, 104.0],
            "afternoon_volume": [1500.0, 1600.0, 1550.0, 1650.0, 1700.0],
            "adjustmentclose": [100.0, 102.0, 101.0, 103.0, 104.0],
            "adjustmenthigh": [101.0, 103.0, 102.0, 104.0, 105.0],
            "adjustmentlow": [99.0, 101.0, 100.0, 102.0, 103.0],
            "adjustmentvolume": [2000.0, 2100.0, 2050.0, 2150.0, 2200.0],
        }
    ).with_columns(pl.col("date").cast(pl.Date))

    engineered = SessionFeatureEngineer().add_features(df, intraday_mode=False)
    rows = engineered.sort(["code", "date"]).to_dicts()

    # Expect early rows to be null due to shift(1)
    assert rows[0]["am_gap_prev_close"] is None
    assert rows[1]["am_gap_prev_close"] is None

    # Row 2: derived from previous day
    expected_am_gap = (102.0 / 100.0) - 1.0
    assert rows[2]["am_gap_prev_close"] == pytest.approx(expected_am_gap, rel=1e-6)

    expected_am_range = (103.0 - 101.0) / 102.0
    assert rows[2]["am_range"] == pytest.approx(expected_am_range, rel=1e-6)

    expected_am_to_full = (103.0 - 101.0) / (103.0 - 101.0)
    assert rows[2]["am_to_full_range"] == pytest.approx(expected_am_to_full, rel=1e-6)

    expected_vol_share = 1200.0 / 2100.0
    assert rows[2]["am_vol_share"] == pytest.approx(expected_vol_share, rel=1e-6)

    # Morning upper limit on day 2 should appear as am_limit_up_flag on day 3
    assert rows[2]["am_limit_up_flag"] == 1
    assert rows[2]["am_limit_any_flag"] == 1
    assert rows[3]["am_limit_up_flag"] == 0
    assert rows[3]["am_limit_down_flag"] == 0
    assert rows[4]["am_limit_down_flag"] == 1
    assert rows[4]["am_limit_any_flag"] == 1

    # PM gap/range should reference previous day values
    expected_pm_gap = (103.0 / 102.5) - 1.0
    assert rows[2]["pm_gap_am_close"] == pytest.approx(expected_pm_gap, rel=1e-6)

    expected_pm_range = (104.0 - 102.0) / 103.0
    assert rows[2]["pm_range"] == pytest.approx(expected_pm_range, rel=1e-6)
