"""High-level helpers for fetching quote data."""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from threading import Lock
from time import perf_counter
from typing import Iterable, List, Literal

from .jquants_fetcher import JQuantsFetcher

LOGGER = logging.getLogger(__name__)


class QuotesFetcher:
    """Convenience wrapper around J-Quants quote endpoints."""

    def __init__(self, *, client: JQuantsFetcher | None = None) -> None:
        self.client = client or JQuantsFetcher()

    def fetch_batch(self, *, codes: Iterable[str], start: str, end: str) -> List[dict[str, str]]:
        """Legacy method: by-code fetching (use fetch_batch_optimized for better performance)."""

        codes_list = list(codes)
        total = len(codes_list)
        if total == 0:
            LOGGER.warning("fetch_batch called with empty symbol list (%s to %s)", start, end)
            return []

        result: List[dict[str, str]] = []
        timer_start = perf_counter()
        for idx, code in enumerate(codes_list, start=1):
            if idx == 1 or idx == total or idx % 25 == 0:
                elapsed = perf_counter() - timer_start
                LOGGER.info(
                    "[QUOTES] Fetch by-code progress %d/%d (%.1fs elapsed)",
                    idx,
                    total,
                    elapsed,
                )
            rows = self.client.fetch_quotes_paginated(code=code, from_=start, to=end)
            result.extend(rows)
        return result

    def fetch_by_date(self, *, dates: Iterable[str], codes: set[str] | None = None) -> List[dict[str, str]]:
        """Fetch quotes by date axis (営業日ごと取得) with parallel execution."""

        date_list = list(dates)
        total = len(date_list)
        if total == 0:
            LOGGER.warning("fetch_by_date called with empty date list")
            return []

        # Check for parallel mode (default: enabled for >1 date)
        enable_parallel = os.getenv("QUOTES_PARALLEL_FETCH", "1") == "1"
        max_workers = int(os.getenv("QUOTES_PARALLEL_WORKERS", "16"))

        if enable_parallel and total > 1:
            return self._fetch_by_date_parallel(date_list, codes, max_workers)
        else:
            # Sequential fallback (original behavior)
            return self._fetch_by_date_sequential(date_list, codes)

    def _fetch_by_date_sequential(self, date_list: List[str], codes: set[str] | None) -> List[dict[str, str]]:
        """Sequential fetch (original implementation)."""
        result: List[dict[str, str]] = []
        timer_start = perf_counter()
        total = len(date_list)

        for idx, date in enumerate(date_list, start=1):
            if idx == 1 or idx == total or idx % 20 == 0:
                elapsed = perf_counter() - timer_start
                LOGGER.info(
                    "[QUOTES] Fetch by-date progress %d/%d (%.1fs elapsed)",
                    idx,
                    total,
                    elapsed,
                )
            date_api = date.replace("-", "")
            rows = self.client.fetch_quotes_by_date_paginated(date=date_api)
            if codes:
                rows = [row for row in rows if row.get("Code") in codes]
            result.extend(rows)
        return result

    def _fetch_by_date_parallel(self, date_list: List[str], codes: set[str] | None, max_workers: int) -> List[dict[str, str]]:
        """Parallel fetch using ThreadPoolExecutor."""
        total = len(date_list)
        timer_start = perf_counter()
        result: List[dict[str, str]] = []
        result_lock = Lock()
        counter = {"done": 0}
        counter_lock = Lock()

        def fetch_single_date(date: str) -> List[dict[str, str]]:
            """Fetch quotes for a single date."""
            date_api = date.replace("-", "")
            rows = self.client.fetch_quotes_by_date_paginated(date=date_api)
            if codes:
                rows = [row for row in rows if row.get("Code") in codes]

            # Update progress
            with counter_lock:
                counter["done"] += 1
                done = counter["done"]
                if done == 1 or done == total or done % 10 == 0:
                    elapsed = perf_counter() - timer_start
                    LOGGER.info(
                        "[QUOTES] Parallel fetch by-date progress %d/%d (%.1fs elapsed)",
                        done,
                        total,
                        elapsed,
                    )
            return rows

        LOGGER.info("[QUOTES] Parallel fetch enabled: %d dates with %d workers", total, max_workers)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_single_date, date): date for date in date_list}

            for future in as_completed(futures):
                try:
                    rows = future.result()
                    with result_lock:
                        result.extend(rows)
                except Exception as e:
                    date = futures[future]
                    LOGGER.warning("Failed to fetch quotes for date %s: %s", date, e)

        elapsed = perf_counter() - timer_start
        LOGGER.info(
            "[QUOTES] Parallel fetch completed: %d records from %d dates in %.1fs",
            len(result),
            total,
            elapsed,
        )
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
        codes_list = list(codes)
        codes_set = set(codes_list)
        total_codes = len(codes_list)
        days = self._calculate_business_days(start, end)

        # Check for axis override (parameter takes precedence over env var)
        if axis_override is None:
            env_axis = os.getenv("FETCH_AXIS")
            if env_axis in ("by_date", "by_code"):
                axis_override = env_axis  # type: ignore

        # Apply axis override if specified
        if axis_override == "by_date":
            date_list = self._generate_date_list(start, end)
            LOGGER.info(
                "[QUOTES] Axis override=%s using by-date (%d days × %d codes)",
                axis_override,
                len(date_list),
                total_codes,
            )
            return self.fetch_by_date(dates=date_list, codes=codes_set)
        elif axis_override == "by_code":
            LOGGER.info(
                "[QUOTES] Axis override=by_code for %d codes (%s to %s)",
                total_codes,
                start,
                end,
            )
            return self.fetch_batch(codes=codes_list, start=start, end=end)

        # Auto-select axis (simple heuristic: if period is short, use by-date)
        if days <= 30:
            # By-date is more efficient for short periods
            date_list = self._generate_date_list(start, end)
            LOGGER.info(
                "[QUOTES] Auto-select by-date axis (%d days ≤ 30 for %d codes)",
                len(date_list),
                total_codes,
            )
            return self.fetch_by_date(dates=date_list, codes=codes_set)
        else:
            # By-code is more efficient for long periods
            LOGGER.info(
                "[QUOTES] Auto-select by-code axis (%d business days) for %d codes",
                days,
                total_codes,
            )
            return self.fetch_batch(codes=codes_list, start=start, end=end)

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
