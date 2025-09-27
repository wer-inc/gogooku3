import polars as pl

from src.features.calendar_utils import build_next_bday_expr_from_dates
from src.gogooku3.features.margin_daily import build_daily_effective
from src.gogooku3.features.short_selling_sector import build_sector_short_features


def test_margin_daily_tplus1_holiday_jump():
    # Published on Friday; next business day is Tuesday (skipping weekend + holiday)
    daily = pl.DataFrame(
        {
            "Code": ["1301"],
            "PublishedDate": ["2024-01-05"],
            "ApplicationDate": ["2024-01-05"],
            "ShortMarginOutstanding": [100.0],
            "LongMarginOutstanding": [120.0],
        }
    ).with_columns([
        pl.col("PublishedDate").str.strptime(pl.Date),
        pl.col("ApplicationDate").str.strptime(pl.Date),
    ])

    nb = build_next_bday_expr_from_dates(["2024-01-05", "2024-01-09"])  # 1/08 相当を休場とみなしスキップ
    eff = build_daily_effective(daily, next_business_day=nb)
    assert eff.select(pl.col("effective_start")).item() == pl.date(2024, 1, 9)


def test_sector_short_tplus1_holiday_jump():
    ss = pl.DataFrame(
        {
            "Date": ["2024-01-05"],
            "Sector33Code": ["1050"],
            "Selling": [100000.0],
            "ShortSelling": [25000.0],
            "ShortSellingWithPriceRestriction": [5000.0],
        }
    ).with_columns([
        pl.col("Date").str.strptime(pl.Date),
    ])

    nb = build_next_bday_expr_from_dates(["2024-01-05", "2024-01-09"])  # holiday skip
    feats = build_sector_short_features(ss, calendar_next_bday=nb, enable_z_scores=False)
    assert feats.select(pl.col("effective_date")).item() == pl.date(2024, 1, 9)

