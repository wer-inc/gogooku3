from __future__ import annotations

from datetime import date

import polars as pl

from gogooku3.features.earnings_events import add_earnings_event_block


def _make_quotes(dates: list[date]) -> pl.DataFrame:
    return pl.DataFrame({"Code": ["1301"] * len(dates), "Date": dates})


def _business_days(dates: list[date]) -> list[str]:
    return [d.isoformat() for d in dates]


def test_fallback_schedule_with_different_asof_hours() -> None:
    dates = [date(2024, 1, 4), date(2024, 1, 5), date(2024, 1, 8)]
    quotes = _make_quotes(dates)
    announcements = pl.DataFrame({"Code": ["1301"], "Date": [date(2024, 1, 5)]})
    business_days = _business_days(dates)

    enriched_19 = add_earnings_event_block(
        quotes,
        announcements,
        business_days=business_days,
        windows=[1, 3, 5],
        asof_hour=19,
    )
    prev_day = enriched_19.filter(pl.col("Date") == date(2024, 1, 4)).to_dicts()[0]
    event_day = enriched_19.filter(pl.col("Date") == date(2024, 1, 5)).to_dicts()[0]
    next_day = enriched_19.filter(pl.col("Date") == date(2024, 1, 8)).to_dicts()[0]

    assert next_day["e_days_to"] is None
    assert next_day["e_next_available_ts"] is None
    assert prev_day["e_days_to"] == 1
    assert prev_day["e_win_pre1"] == 1
    assert event_day["e_days_to"] == 0
    assert event_day["e_is_E0"] == 1
    assert next_day["e_days_since"] == 1

    enriched_15 = add_earnings_event_block(
        quotes,
        announcements,
        business_days=business_days,
        windows=[1, 3, 5],
        asof_hour=15,
    )
    prev_day_15 = enriched_15.filter(pl.col("Date") == date(2024, 1, 4)).to_dicts()[0]
    event_day_15 = enriched_15.filter(pl.col("Date") == date(2024, 1, 5)).to_dicts()[0]

    assert prev_day_15["e_days_to"] is None
    assert prev_day_15["e_win_pre1"] == 0
    assert event_day_15["e_days_to"] == 0
    assert event_day_15["e_is_E0"] == 1


def test_same_day_publication_defers_visibility() -> None:
    dates = [date(2024, 1, 5), date(2024, 1, 8)]
    quotes = _make_quotes(dates)
    announcements = pl.DataFrame(
        {
            "Code": ["1301"],
            "Date": [date(2024, 1, 5)],
            "PublishedDateTime": ["2024-01-05T19:00:00"],
        }
    )
    business_days = _business_days([date(2024, 1, 5), date(2024, 1, 8)])

    enriched = add_earnings_event_block(
        quotes,
        announcements,
        business_days=business_days,
        windows=[1, 3, 5],
        asof_hour=15,
    )
    event_row = enriched.filter(pl.col("Date") == date(2024, 1, 5)).to_dicts()[0]
    follow_row = enriched.filter(pl.col("Date") == date(2024, 1, 8)).to_dicts()[0]

    assert event_row["e_days_to"] is None
    assert follow_row["e_next_available_ts"] is None
    assert follow_row["e_days_since"] == 1
    assert follow_row["e_win_post1"] == 1

    enriched_19 = add_earnings_event_block(
        quotes,
        announcements,
        business_days=business_days,
        windows=[1, 3, 5],
        asof_hour=19,
    )
    event_row_19 = enriched_19.filter(pl.col("Date") == date(2024, 1, 5)).to_dicts()[0]
    assert event_row_19["e_days_to"] == 0
    assert event_row_19["e_is_E0"] == 1


def test_available_timestamp_never_exceeds_asof() -> None:
    dates = [date(2024, 1, 4), date(2024, 1, 5)]
    quotes = _make_quotes(dates)
    announcements = pl.DataFrame({"Code": ["1301"], "Date": [date(2024, 1, 5)]})
    business_days = _business_days(dates)

    enriched = add_earnings_event_block(
        quotes,
        announcements,
        business_days=business_days,
        windows=[1, 3, 5],
        asof_hour=19,
    )

    asof_ts = enriched.select((pl.col("Date").cast(pl.Datetime("us")) + pl.duration(hours=19)).alias("asof"))[
        "asof"
    ].to_list()
    avail_ts = enriched["e_next_available_ts"].to_list()

    for avail, asof in zip(avail_ts, asof_ts):
        if avail is None:
            continue
        assert avail <= asof
