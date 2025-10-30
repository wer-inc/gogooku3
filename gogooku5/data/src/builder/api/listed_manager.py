"""Manage listed securities metadata."""
from __future__ import annotations

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
        self._listed = payload.get("info", [])
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

        return sorted({str(item.get("sectorCode", "UNKNOWN")) for item in self.listed()})

    def load_from_iterable(self, instruments: Iterable[Instrument]) -> None:
        """Manually seed listed metadata (useful for tests)."""

        self._listed = list(instruments)
        if self.market_filter:
            self._listed = self.market_filter.filter(self._listed)
