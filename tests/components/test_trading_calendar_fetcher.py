import datetime as dt
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.components.trading_calendar_fetcher import TradingCalendarFetcher


@pytest.fixture
def fetcher(tmp_path):
    # isolate cache directory during tests
    instance = TradingCalendarFetcher()
    instance.cache_dir = tmp_path
    instance.cache_dir.mkdir(parents=True, exist_ok=True)
    return instance


def test_extract_subscription_bounds_with_end():
    text = '{"message": "Your subscription covers the following dates: 2015-09-23 ~ 2024-12-31."}'
    start, end = TradingCalendarFetcher._extract_subscription_bounds(text)
    assert start == dt.date(2015, 9, 23)
    assert end == dt.date(2024, 12, 31)


def test_extract_subscription_bounds_without_end():
    text = '{"message": "Your subscription covers the following dates: 2015-09-23 ~ ."}'
    start, end = TradingCalendarFetcher._extract_subscription_bounds(text)
    assert start == dt.date(2015, 9, 23)
    assert end is None


def test_clamp_requested_range_adjusts_start(fetcher):
    fetcher.subscription_start = dt.date(2015, 9, 23)
    clamped_from, clamped_to, adjusted = fetcher._clamp_requested_range(
        "2013-11-07", "2025-09-19"
    )
    assert clamped_from == "2015-09-23"
    assert clamped_to == "2025-09-19"
    assert adjusted is True


def test_clamp_requested_range_raises_when_outside(fetcher):
    fetcher.subscription_start = dt.date(2015, 9, 23)
    fetcher.subscription_end = dt.date(2015, 12, 31)
    with pytest.raises(ValueError):
        fetcher._clamp_requested_range("2014-01-01", "2015-03-01")


def test_parse_date_accepts_compact_format():
    parsed = TradingCalendarFetcher._parse_date("20150923")
    assert parsed == dt.date(2015, 9, 23)
