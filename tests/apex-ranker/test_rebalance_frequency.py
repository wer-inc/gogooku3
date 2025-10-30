from datetime import date

import pytest
from apex_ranker.backtest import normalise_frequency, should_rebalance


def test_normalise_frequency_accepts_supported_modes() -> None:
    assert normalise_frequency("daily") == "daily"
    assert normalise_frequency("Weekly") == "weekly"
    assert normalise_frequency("MONTHLY") == "monthly"


def test_normalise_frequency_rejects_invalid_modes() -> None:
    with pytest.raises(ValueError):
        normalise_frequency("quarterly")


def test_should_rebalance_daily_always_true_when_invested() -> None:
    last = date(2025, 1, 6)
    assert should_rebalance(date(2025, 1, 7), last, "daily")
    assert should_rebalance(date(2025, 1, 8), last, "daily")


def test_should_rebalance_first_call_always_true() -> None:
    assert should_rebalance(date(2025, 1, 6), None, "monthly")
    assert should_rebalance(date(2025, 1, 6), None, "weekly")


def test_should_rebalance_weekly_only_on_friday() -> None:
    last = date(2024, 12, 27)  # Friday
    assert should_rebalance(date(2025, 1, 3), last, "weekly")  # Friday
    assert not should_rebalance(date(2025, 1, 6), date(2025, 1, 3), "weekly")  # Monday


def test_should_rebalance_monthly_on_first_trading_day() -> None:
    last = date(2024, 12, 30)
    assert should_rebalance(date(2025, 1, 6), last, "monthly")
    assert not should_rebalance(date(2025, 1, 7), date(2025, 1, 6), "monthly")
    assert should_rebalance(date(2025, 2, 3), date(2025, 1, 6), "monthly")
