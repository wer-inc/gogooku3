"""Manage listed securities metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, List

from .jquants_fetcher import JQuantsFetcher
from .market_filter import Instrument, MarketFilter


class ListedManager:
    """Cache and filter listed securities information."""

    def __init__(
        self,
        *,
        fetcher: JQuantsFetcher | None = None,
        market_filter: MarketFilter | None = None,
    ) -> None:
        self.fetcher = fetcher or JQuantsFetcher()
        self.market_filter = market_filter
        self._listed: List[Instrument] = []

    def refresh(self) -> List[Instrument]:
        """Fetch the latest listed info and cache it."""

        payload = self.fetcher.fetch_listed_info()
        raw_listed = payload.get("info", [])
        self._listed = [self._normalize_instrument(item) for item in raw_listed]
        if self.market_filter:
            self._listed = self.market_filter.filter(self._listed)
        return self._listed

    def listed(self) -> List[Instrument]:
        """Return cached listed data, refreshing if necessary."""

        if not self._listed:
            return self.refresh()
        return list(self._listed)

    def codes(self) -> List[str]:
        """Return the list of codes for currently cached instruments."""

        return [str(item.get("code", "")) for item in self.listed() if item.get("code")]

    def sectors(self) -> List[str]:
        """Return unique sector codes for cached instruments."""

        return sorted({str(item.get("sector_code", "UNKNOWN")) for item in self.listed()})

    def load_from_iterable(self, instruments: Iterable[Instrument]) -> None:
        """Manually seed listed metadata (useful for tests)."""

        self._listed = list(instruments)
        if self.market_filter:
            self._listed = self.market_filter.filter(self._listed)

    @staticmethod
    def _normalize_instrument(entry: Instrument) -> Instrument:
        """Normalize API payload keys to snake_case for downstream use."""

        normalized: Instrument = dict(entry)
        code = entry.get("Code") or entry.get("code")
        if code:
            normalized["code"] = str(code)

        sector = (
            entry.get("Sector33Code")
            or entry.get("Sector17Code")
            or entry.get("sector_code")
            or entry.get("sectorCode")
        )
        if sector:
            normalized["sector_code"] = str(sector)

        market = entry.get("MarketCode") or entry.get("marketCode") or entry.get("market_code")
        if market:
            normalized["market_code"] = str(market)

        margin = entry.get("MarginCode") or entry.get("marginCode")
        if margin:
            normalized["margin_code"] = str(margin)

        listed_date = entry.get("Date") or entry.get("date")
        if listed_date:
            try:
                normalized["listed_date"] = datetime.fromisoformat(str(listed_date))
            except ValueError:
                normalized["listed_date"] = listed_date

        return normalized
