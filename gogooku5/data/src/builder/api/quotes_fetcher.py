"""High-level helpers for fetching quote data."""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Iterable, List, Literal

from .jquants_fetcher import JQuantsFetcher


class QuotesFetcher:
    """Convenience wrapper around J-Quants quote endpoints."""

    def __init__(self, *, client: JQuantsFetcher | None = None) -> None:
        self.client = client or JQuantsFetcher()

    def fetch_batch(self, *, codes: Iterable[str], start: str, end: str) -> List[dict[str, str]]:
        """Legacy method: by-code fetching (use fetch_batch_optimized for better performance)."""
        result: List[dict[str, str]] = []
        for code in codes:
            rows = self.client.fetch_quotes_paginated(code=code, from_=start, to=end)
            result.extend(rows)
        return result

    def fetch_by_date(self, *, dates: Iterable[str], codes: set[str] | None = None) -> List[dict[str, str]]:
        """Fetch quotes by date axis (営業日ごと取得)."""
        result: List[dict[str, str]] = []
        for date in dates:
            # API expects YYYYMMDD format
            date_api = date.replace("-", "")
            rows = self.client.fetch_quotes_by_date_paginated(date=date_api)
            # Filter by codes if specified
            if codes:
                rows = [row for row in rows if row.get("Code") in codes]
            result.extend(rows)
        return result

    def fetch_batch_optimized(
        self,
        *,
        codes: Iterable[str],
        start: str,
        end: str,
        axis_override: Literal["by_date", "by_code"] | None = None,
    ) -> List[dict[str, str]]:
        """
        Optimized batch fetching with automatic axis selection.

        Strategy (when axis_override=None):
        - Short-term (<= 30 days): Use by-date axis (faster for many stocks)
        - Long-term (> 30 days): Use by-code axis (faster for few days)

        Args:
            codes: Stock codes to fetch
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            axis_override: Force specific axis ("by_date" or "by_code")
                          Reads from FETCH_AXIS env var if not provided

        Returns:
            List of quote dictionaries

        Example:
            >>> # Force by-date axis for large symbol count scenarios
            >>> fetcher.fetch_batch_optimized(codes=all_codes, start="2024-01-01", end="2024-12-31", axis_override="by_date")
        """
        codes_set = set(codes)
        days = self._calculate_business_days(start, end)

        # Check for axis override (parameter takes precedence over env var)
        if axis_override is None:
            env_axis = os.getenv("FETCH_AXIS")
            if env_axis in ("by_date", "by_code"):
                axis_override = env_axis  # type: ignore

        # Apply axis override if specified
        if axis_override == "by_date":
            date_list = self._generate_date_list(start, end)
            return self.fetch_by_date(dates=date_list, codes=codes_set)
        elif axis_override == "by_code":
            return self.fetch_batch(codes=codes, start=start, end=end)

        # Auto-select axis (simple heuristic: if period is short, use by-date)
        if days <= 30:
            # By-date is more efficient for short periods
            date_list = self._generate_date_list(start, end)
            return self.fetch_by_date(dates=date_list, codes=codes_set)
        else:
            # By-code is more efficient for long periods
            return self.fetch_batch(codes=codes, start=start, end=end)

    def _calculate_business_days(self, start: str, end: str) -> int:
        """Estimate number of business days between two dates."""
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        total_days = (end_dt - start_dt).days
        # Rough estimate: ~71% of days are business days
        return max(1, int(total_days * 0.71))

    def _generate_date_list(self, start: str, end: str) -> List[str]:
        """Generate list of dates between start and end (inclusive)."""
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")

        dates = []
        current = start_dt
        while current <= end_dt:
            # Skip weekends (simple heuristic)
            if current.weekday() < 5:  # Monday=0, Friday=4
                dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        return dates
