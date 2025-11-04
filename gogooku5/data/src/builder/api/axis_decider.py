"""Logic for determining fetch axes (codes, sectors, etc.)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class AxisDecider:
    """Determine which axes (symbols, sectors, markets) to query."""

    symbols: List[str]
    sectors: List[str]

    def choose_symbols(self, *, limit: int | None = None) -> List[str]:
        """Return a symbol list capped by `limit` if provided."""

        return self.symbols[:limit] if limit else list(self.symbols)

    def choose_sectors(self, include_delisted: bool = False) -> List[str]:
        """Return sectors to process, optionally excluding delisted entries."""

        if include_delisted:
            return list(self.sectors)
        return [sector for sector in self.sectors if not sector.endswith("_DELISTED")]

    @staticmethod
    def from_listed_symbols(listed: Iterable[dict[str, str]]) -> "AxisDecider":
        """Build an AxisDecider from listed symbol metadata."""

        symbols = [entry["code"] for entry in listed if entry.get("code")]
        sectors = sorted({entry.get("sector_code", entry.get("Sector33Code", "UNKNOWN")) for entry in listed})
        return AxisDecider(symbols=symbols, sectors=sectors)
