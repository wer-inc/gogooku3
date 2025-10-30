"""High-level helpers for fetching quote data."""
from __future__ import annotations

from typing import Iterable, List

from .jquants_fetcher import JQuantsFetcher


class QuotesFetcher:
    """Convenience wrapper around J-Quants quote endpoints."""

    def __init__(self, *, client: JQuantsFetcher | None = None) -> None:
        self.client = client or JQuantsFetcher()

    def fetch_batch(self, *, codes: Iterable[str], start: str, end: str) -> List[dict[str, str]]:
        result: List[dict[str, str]] = []
        for code in codes:
            rows = self.client.fetch_quotes_paginated(code=code, from_=start, to=end)
            result.extend(rows)
        return result
