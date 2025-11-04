from __future__ import annotations

import pytest

from builder.utils.datetime import business_date_range


def test_business_date_range_holiday_fallback(monkeypatch):
    # Simulate environment without jpholiday to exercise pandas fallback
    monkeypatch.setattr("builder.utils.datetime.jpholiday", None, raising=False)
    with pytest.raises(RuntimeError):
        business_date_range("2024-01-01", "2024-01-03")
