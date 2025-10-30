"""Datetime helpers for dataset builder."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import List


def date_range(start: str, end: str) -> List[str]:
    """Return a list of ISO date strings inclusive of start/end."""

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    delta = end_dt - start_dt
    if delta.days < 0:
        raise ValueError("start date must be before end date")
    return [(start_dt + timedelta(days=offset)).strftime("%Y-%m-%d") for offset in range(delta.days + 1)]
