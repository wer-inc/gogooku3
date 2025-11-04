"""Datetime helpers for dataset builder."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

try:
    import jpholiday
except ImportError:  # pragma: no cover - optional dependency
    jpholiday = None

try:  # pragma: no cover - optional dependency
    from pandas.tseries.holiday import JapanHolidayCalendar
except ImportError:  # pragma: no cover
    JapanHolidayCalendar = None


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
    if jpholiday is None and JapanHolidayCalendar is None:
        raise RuntimeError(
            "Japanese holiday calendar unavailable. Install 'jpholiday' or pandas holiday extras to compute business days."
        )

    if jpholiday is None and JapanHolidayCalendar is not None:
        calendar = JapanHolidayCalendar()
        holidays = {
            day.strftime("%Y-%m-%d")
            for day in calendar.holidays(start=start_dt, end=end_dt)
        }
        while current <= end_dt:
            iso_current = current.strftime("%Y-%m-%d")
            if current.weekday() < 5 and iso_current not in holidays:
                dates.append(iso_current)
            current += timedelta(days=1)
    else:
        while current <= end_dt:
            if _is_business_day(current):
                dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)

    # Fallback: if no business day detected (e.g. single holiday),
    # include start date to avoid empty grids.
    if not dates:
        dates.append(start_dt.strftime("%Y-%m-%d"))
    return dates


def shift_trading_days(date: str, days: int) -> str:
    """
    Shift a date by N trading days (JP business days).

    Args:
        date: ISO date string (YYYY-MM-DD)
        days: Number of trading days to shift (negative for backward)

    Returns:
        ISO date string after shifting

    Examples:
        shift_trading_days("2025-01-06", -60)  # ~60 trading days before
        shift_trading_days("2025-01-06", 5)    # 5 trading days after
    """
    dt = datetime.strptime(date, "%Y-%m-%d")

    if jpholiday is None and JapanHolidayCalendar is None:
        raise RuntimeError(
            "Japanese holiday calendar unavailable. Install 'jpholiday' or pandas holiday extras."
        )

    direction = 1 if days > 0 else -1
    remaining = abs(days)
    current = dt

    while remaining > 0:
        current += timedelta(days=direction)
        if _is_business_day(current):
            remaining -= 1

    return current.strftime("%Y-%m-%d")
