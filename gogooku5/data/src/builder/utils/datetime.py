"""Datetime helpers for dataset builder."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

try:
    import jpholiday
except ImportError:  # pragma: no cover - optional dependency
    jpholiday = None


def _is_business_day(dt: datetime) -> bool:
    """Return True if the given datetime represents a JP trading day."""

    if dt.weekday() >= 5:  # Saturday/Sunday
        return False
    if jpholiday is not None and jpholiday.is_holiday(dt.date()):
        return False
    return True

def date_range(start: str, end: str) -> List[str]:
    """Return a list of ISO date strings inclusive of start/end."""

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    delta = end_dt - start_dt
    if delta.days < 0:
        raise ValueError("start date must be before end date")
    return [(start_dt + timedelta(days=offset)).strftime("%Y-%m-%d") for offset in range(delta.days + 1)]


def business_date_range(start: str, end: str) -> List[str]:
    """Return ISO dates that fall on JP trading days between start/end (inclusive)."""

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    if end_dt < start_dt:
        raise ValueError("start date must be before end date")

    dates: List[str] = []
    current = start_dt
    while current <= end_dt:
        if _is_business_day(current):
            dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    # Fallback: if no business day detected (e.g. single holiday),
    # include start date to avoid empty grids.
    if not dates:
        dates.append(start_dt.strftime("%Y-%m-%d"))
    return dates
