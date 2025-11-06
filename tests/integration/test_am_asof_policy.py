from __future__ import annotations

from datetime import date

import polars as pl

from scripts.data.ml_dataset_builder import MLDatasetBuilder


def _base_quotes() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Code": ["1301", "1301", "1301"],
            "Date": [
                date(2024, 1, 4),
                date(2024, 1, 5),
                date(2024, 1, 8),
            ],
            "Open": [100.0, 105.0, 103.0],
            "High": [110.0, 112.0, 109.0],
            "Low": [95.0, 101.0, 100.0],
            "Close": [105.0, 103.0, 108.0],
            "Volume": [1_000.0, 1_200.0, 1_050.0],
        }
    )


def _am_quotes() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Code": ["1301", "1301"],
            "Date": [date(2024, 1, 4), date(2024, 1, 5)],
            "MorningOpen": [101.0, 104.0],
            "MorningHigh": [108.0, 107.0],
            "MorningLow": [100.0, 102.0],
            "MorningClose": [106.0, 105.0],
            "MorningVolume": [500.0, 650.0],
            "MorningTurnoverValue": [50_000.0, 65_000.0],
        }
    )


def test_am_features_t_plus_one() -> None:
    builder = MLDatasetBuilder()
    base = _base_quotes()
    am_df = _am_quotes()
    business_days = ["2024-01-04", "2024-01-05", "2024-01-08"]

    enriched = builder.add_am_session_features(
        base,
        am_df,
        business_days=business_days,
        asof_policy="T+1",
    ).sort(["Code", "Date"])

    jan4 = enriched.filter(pl.col("Date") == date(2024, 1, 4)).to_dicts()[0]
    assert jan4["am_body"] is None
    assert jan4["is_am_valid"] == 0

    jan5 = enriched.filter(pl.col("Date") == date(2024, 1, 5)).to_dicts()[0]
    assert jan5["is_am_valid"] == 1
    assert round(jan5["am_gap_prev_close"], 6) == round(106.0 / 105.0 - 1.0, 6)
    assert round(jan5["am_body"], 6) == round(106.0 / 101.0 - 1.0, 6)
    assert round(jan5["am_range"], 6) == round(108.0 / 100.0 - 1.0, 6)
    assert round(jan5["am_pos_in_am_range"], 6) == round((106.0 - 100.0) / (108.0 - 100.0), 6)
    assert round(jan5["am_to_full_range_prev"], 6) == round((108.0 / 100.0 - 1.0) / (110.0 / 95.0 - 1.0), 6)

    jan8 = enriched.filter(pl.col("Date") == date(2024, 1, 8)).to_dicts()[0]
    assert jan8["is_am_valid"] == 1
    assert round(jan8["am_body"], 6) == round(105.0 / 104.0 - 1.0, 6)
    assert round(jan8["am_to_full_range_prev"], 6) == round((107.0 / 102.0 - 1.0) / (112.0 / 101.0 - 1.0), 6)


def test_am_features_same_day_policy() -> None:
    builder = MLDatasetBuilder()
    base = _base_quotes()
    am_df = _am_quotes()
    business_days = ["2024-01-04", "2024-01-05", "2024-01-08"]

    enriched = builder.add_am_session_features(
        base,
        am_df,
        business_days=business_days,
        asof_policy="SAME_DAY_PM",
    ).sort(["Code", "Date"])

    jan4 = enriched.filter(pl.col("Date") == date(2024, 1, 4)).to_dicts()[0]
    assert jan4["is_am_valid"] == 1
    assert round(jan4["am_body"], 6) == round(106.0 / 101.0 - 1.0, 6)
    assert jan4["am_to_full_range_prev"] is None
    assert "am_source_date" not in jan4
