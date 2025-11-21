"""Trading calendar utilities."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import APIClient

CALENDAR_ENDPOINT = "/markets/trading_calendar"


class TradingCalendarFetcher(APIClient):
    """Fetch trading calendars and compute session metadata."""

    def __init__(self, *, base_url: str = "https://api.jquants.com/v1") -> None:
        super().__init__(base_url=base_url)

    def fetch_calendar(self, *, year: int, market_code: Optional[str] = None) -> Dict[str, Any]:
        """Return the raw trading calendar for a given year."""

        params: Dict[str, Any] = {"year": year}
        if market_code:
            params["marketCode"] = market_code
        response = self.request("GET", CALENDAR_ENDPOINT, params=params)
        return response.json()
