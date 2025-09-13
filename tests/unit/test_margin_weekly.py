import sys
from pathlib import Path

# Ensure project root is importable so `import src...` works
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import polars as pl
import pytest

from src.gogooku3.features.margin_weekly import add_margin_weekly_block


@pytest.mark.unit
def test_margin_step_function_with_published_date():
    # Daily quotes grid (weekdays only)
    def business_days(start: str, end: str):
        from datetime import datetime, timedelta

        s = datetime.strptime(start, "%Y-%m-%d")
        e = datetime.strptime(end, "%Y-%m-%d")
        cur = s
        out = []
        while cur <= e:
            if cur.weekday() < 5:  # Mon-Fri
                out.append(cur.strftime("%Y-%m-%d"))
            cur += timedelta(days=1)
        return out

    dates = business_days("2025-09-01", "2025-09-15")
    quotes = pl.DataFrame(
        {
            "Code": ["1301"] * len(dates),
            "Date": pl.Series(dates).str.strptime(pl.Date),
            # Minimal OHLCV to satisfy pipeline expectations
            "Open": [100.0] * len(dates),
            "High": [101.0] * len(dates),
            "Low": [99.0] * len(dates),
            "Close": [100.0] * len(dates),
            "Volume": [1000.0] * len(dates),
        }
    )

    # Two weekly snapshots with PublishedDate (so effective_start = T+1 business day)
    weekly = pl.DataFrame(
        {
            "Code": ["1301", "1301"],
            "Date": pl.Series(["2025-09-01", "2025-09-08"]).str.strptime(pl.Date),
            "PublishedDate": pl.Series(["2025-09-01", "2025-09-08"]).str.strptime(pl.Date),
            # Use distinct L/S to observe change
            "LongMarginTradeVolume": [1000.0, 900.0],
            "ShortMarginTradeVolume": [200.0, 300.0],
            # Provide positive values to avoid 0/0 in aux ratios (not asserted)
            "LongNegotiableMarginTradeVolume": [10.0, 10.0],
            "ShortNegotiableMarginTradeVolume": [5.0, 5.0],
            "LongStandardizedMarginTradeVolume": [20.0, 20.0],
            "ShortStandardizedMarginTradeVolume": [8.0, 8.0],
            "IssueType": ["2", "2"],
        }
    )

    out = add_margin_weekly_block(quotes=quotes, weekly_df=weekly, lag_bdays_weekly=3, adv_window_days=20)

    # Effective starts: 2025-09-02 and 2025-09-09 (T+1 rule)
    # 1) Before first effective_start: is_margin_valid == 0
    row_0901 = out.filter((pl.col("Code") == "1301") & (pl.col("Date") == pl.date(2025, 9, 1))).select("is_margin_valid").to_series()
    assert row_0901.len() == 1 and (row_0901.item() == 0 or row_0901.item() is None)

    # 2) Between 2025-09-02 and 2025-09-08 inclusive, values reflect first snapshot (1000/200 -> ratio 5.0)
    mid = out.filter((pl.col("Date") >= pl.date(2025, 9, 2)) & (pl.col("Date") <= pl.date(2025, 9, 8)))
    assert (mid["is_margin_valid"] == 1).all()
    assert pytest.approx(5.0, rel=1e-6) == float(mid.select(pl.col("margin_credit_ratio").mean()).item())

    # 3) From 2025-09-09 onward, values reflect second snapshot (900/300 -> ratio 3.0)
    later = out.filter(pl.col("Date") >= pl.date(2025, 9, 9))
    assert (later["is_margin_valid"] == 1).all()
    assert pytest.approx(3.0, rel=1e-6) == float(later.select(pl.col("margin_credit_ratio").mean()).item())

    # 4) margin_impulse is 1 only on effective_start days
    impulses = out.filter(pl.col("margin_impulse") == 1).get_column("Date").to_list()
    from datetime import date as _date
    assert set(impulses) == {_date(2025, 9, 2), _date(2025, 9, 9)}
