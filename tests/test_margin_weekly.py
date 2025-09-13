from __future__ import annotations

import polars as pl

from src.gogooku3.features.margin_weekly import add_margin_weekly_block


def _make_business_days(start: str, end: str) -> list[str]:
    dr = pl.date_range(start, end, eager=True)
    # Keep only weekdays (Mon-Fri)
    return [d.strftime("%Y-%m-%d") for d in dr if d.weekday() < 5]


def test_asof_no_leak_and_updates():
    # Daily grid 2024-01-03 .. 2024-01-20 (weekends auto-skipped by filter)
    bdays = _make_business_days("2024-01-03", "2024-01-20")
    quotes = (
        pl.DataFrame(
            {
                "Code": ["AAA"] * len(bdays),
                "Date": bdays,
                # Use constant volume to simplify ADV20
                "Volume": [1000.0] * len(bdays),
            }
        ).with_columns(pl.col("Date").str.strptime(pl.Date))
    )

    # Weekly margin: 2024-01-05 and 2024-01-12 (Fridays)
    weekly = (
        pl.DataFrame(
            {
                "Code": ["AAA", "AAA"],
                "Date": ["2024-01-05", "2024-01-12"],
                "PublishedDate": [None, None],
                "LongMarginTradeVolume": [5000.0, 7000.0],
                "ShortMarginTradeVolume": [4000.0, 2000.0],
                "LongNegotiableMarginTradeVolume": [2000.0, 3000.0],
                "ShortNegotiableMarginTradeVolume": [1000.0, 900.0],
                "LongStandardizedMarginTradeVolume": [3000.0, 4000.0],
                "ShortStandardizedMarginTradeVolume": [3000.0, 1100.0],
                "IssueType": [2, 2],
            }
        )
        .with_columns(
            [pl.col("Date").str.strptime(pl.Date), pl.col("PublishedDate").cast(pl.Date)]
        )
    )

    # Build block end-to-end with lag=3 (Fri → next Wed effective)
    out = add_margin_weekly_block(
        quotes=quotes,
        weekly_df=weekly,
        lag_bdays_weekly=3,
        adv_window_days=5,  # small for the synthetic window
    )

    # Effective starts should be 2024-01-10 and 2024-01-17
    # Check validity flag transitions (no leak before first effective date)
    first_valid = (
        out.filter(pl.col("Date") == pl.lit("2024-01-10").str.strptime(pl.Date))
        .select("is_margin_valid")
        .item()
    )
    prev_invalid = (
        out.filter(pl.col("Date") == pl.lit("2024-01-09").str.strptime(pl.Date))
        .select("is_margin_valid")
        .item()
    )
    assert prev_invalid == 0
    assert first_valid == 1

    # Values between 2024-01-10 and 16 come from first week record
    s_16 = out.filter(pl.col("Date") == pl.lit("2024-01-16").str.strptime(pl.Date)).select(
        ["margin_long_tot", "margin_short_tot", "is_margin_valid"]
    )
    assert s_16.item(0, 0) == 5000.0  # long
    assert s_16.item(0, 1) == 4000.0  # short
    assert s_16.item(0, 2) == 1

    # From 2024-01-17 it should switch to the second week record
    s_17 = out.filter(pl.col("Date") == pl.lit("2024-01-17").str.strptime(pl.Date)).select(
        ["margin_long_tot", "margin_short_tot", "margin_impulse"]
    )
    assert s_17.item(0, 0) == 7000.0
    assert s_17.item(0, 1) == 2000.0
    assert s_17.item(0, 2) == 1  # impulse day

    # No leak: a day before 2024-01-10 should have null margin_long_tot
    s_09 = (
        out.filter(pl.col("Date") == pl.lit("2024-01-09").str.strptime(pl.Date))
        .select("margin_long_tot")
        .item()
    )
    assert s_09 is None
