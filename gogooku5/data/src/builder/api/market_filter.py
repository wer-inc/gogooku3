"""Utilities for filtering securities by market codes."""
from __future__ import annotations

from typing import Iterable, List, MutableMapping

MarketCode = str
Instrument = MutableMapping[str, object]


class MarketFilter:
    """Filter listed instruments by market code or criteria."""

    def __init__(self, allowed_markets: Iterable[MarketCode]) -> None:
        self.allowed_markets = {code for code in allowed_markets}

    def filter(self, instruments: Iterable[Instrument]) -> List[Instrument]:
        """Return instruments whose `market_code` is allowed."""

        return [inst for inst in instruments if self._is_allowed(inst)]

    def group_by_market(self, instruments: Iterable[Instrument]) -> dict[MarketCode, List[Instrument]]:
        """Group instruments by their market code."""

        grouped: dict[MarketCode, List[Instrument]] = {}
        for inst in instruments:
            code = str(inst.get("market_code", ""))
            grouped.setdefault(code, []).append(inst)
        return grouped

    def _is_allowed(self, instrument: Instrument) -> bool:
        code = instrument.get("market_code")
        return code in self.allowed_markets
