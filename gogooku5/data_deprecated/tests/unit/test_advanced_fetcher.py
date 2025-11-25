from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import polars as pl
import pytest
from builder.api import AdvancedJQuantsFetcher
from builder.config import DatasetBuilderSettings


@dataclass
class _StubAsyncFetcher:
    calls: Dict[str, Tuple[Any, ...]]

    async def authenticate(self, session: Any) -> None:  # pragma: no cover - simple stub
        self.calls["authenticate"] = tuple()

    async def fetch_topix_data(self, session: Any, from_date: str, to_date: str) -> pl.DataFrame:
        self.calls["topix"] = (from_date, to_date)
        return pl.DataFrame({"Date": [from_date], "Close": [1.0]})

    async def fetch_indices_ohlc(
        self, session: Any, from_date: str, to_date: str, codes: List[str] | None = None
    ) -> pl.DataFrame:
        self.calls["indices"] = (from_date, to_date, tuple(codes) if codes else None)
        return pl.DataFrame({"Code": ["TOPIX"], "Date": [from_date], "Close": [1.0]})

    async def get_trades_spec(self, session: Any, from_date: str, to_date: str) -> pl.DataFrame:
        self.calls["trades_spec"] = (from_date, to_date)
        return pl.DataFrame({"Date": [from_date], "value": [1.0]})

    async def get_listed_info(self, session: Any, date: str | None = None) -> pl.DataFrame:
        self.calls["listed_info"] = (date,)
        return pl.DataFrame({"Code": ["1301"], "SectorCode": ["FOOD"]})

    async def get_weekly_margin_interest(self, session: Any, from_date: str, to_date: str) -> pl.DataFrame:
        self.calls["margin_weekly"] = (from_date, to_date)
        return pl.DataFrame({"Date": [from_date], "LongMarginTradeVolume": [1.0]})

    async def get_daily_margin_interest(self, session: Any, from_date: str, to_date: str) -> pl.DataFrame:
        self.calls["margin_daily"] = (from_date, to_date)
        return pl.DataFrame({"Date": [from_date], "ShortMarginOutstanding": [1.0]})

    async def get_futures_daily(self, session: Any, from_date: str, to_date: str) -> pl.DataFrame:
        self.calls["futures"] = (from_date, to_date)
        return pl.DataFrame({"Date": [from_date], "Volume": [100]})

    async def get_index_option(self, session: Any, from_date: str, to_date: str) -> pl.DataFrame:
        self.calls["options"] = (from_date, to_date)
        return pl.DataFrame({"Date": [from_date], "Volume": [50]})

    async def get_short_selling(self, session: Any, from_date: str, to_date: str) -> pl.DataFrame:
        self.calls["short"] = (from_date, to_date)
        return pl.DataFrame({"Date": [from_date], "ShortSellShares": [10]})

    async def get_short_selling_positions(self, session: Any, from_date: str, to_date: str) -> pl.DataFrame:
        self.calls["short_positions"] = (from_date, to_date)
        return pl.DataFrame({"Date": [from_date], "Ratio": [0.1]})

    async def get_sector_short_selling(
        self,
        session: Any,
        from_date: str,
        to_date: str,
        *,
        business_days: List[str] | None = None,
    ) -> pl.DataFrame:
        self.calls["sector_short"] = (from_date, to_date, tuple(business_days) if business_days else None)
        return pl.DataFrame({"Date": [from_date], "SectorCode": ["33A"], "ShortRatio": [0.2]})

    async def get_breakdown(self, session: Any, from_date: str, to_date: str) -> pl.DataFrame:
        self.calls["breakdown"] = (from_date, to_date)
        return pl.DataFrame({"Date": [from_date], "Individual": [1.0]})

    async def get_prices_am(self, session: Any, from_date: str, to_date: str) -> pl.DataFrame:
        self.calls["prices_am"] = (from_date, to_date)
        return pl.DataFrame({"Date": [from_date], "Code": ["1301"], "Open": [100.0]})

    async def get_dividends(self, session: Any, from_date: str, to_date: str) -> pl.DataFrame:
        self.calls["dividends"] = (from_date, to_date)
        return pl.DataFrame({"Date": [from_date], "Dividend": [5.0]})

    async def get_fs_details(self, session: Any, from_date: str, to_date: str) -> pl.DataFrame:
        self.calls["fs_details"] = (from_date, to_date)
        return pl.DataFrame({"Date": [from_date], "NetSales": [1000]})

    async def get_earnings_announcements(self, session: Any, from_date: str, to_date: str) -> pl.DataFrame:
        self.calls["earnings"] = (from_date, to_date)
        return pl.DataFrame({"Date": [from_date], "EPS": [2.0]})


class _StubSession:
    """Async context manager mimicking aiohttp.ClientSession."""

    def __init__(self) -> None:
        self.closed = False

    async def __aenter__(self) -> "_StubSession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.closed = True


@pytest.fixture()
def settings(monkeypatch: pytest.MonkeyPatch, tmp_path) -> DatasetBuilderSettings:
    monkeypatch.setenv("JQUANTS_AUTH_EMAIL", "user@example.com")
    monkeypatch.setenv("JQUANTS_AUTH_PASSWORD", "secret")
    monkeypatch.setenv("DATA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("DATA_CACHE_DIR", str(tmp_path / "cache"))
    return DatasetBuilderSettings(
        jquants_auth_email="user@example.com",
        jquants_auth_password="secret",
        data_output_dir=tmp_path / "output",
        data_cache_dir=tmp_path / "cache",
    )


@pytest.fixture()
def stub_fetcher(monkeypatch: pytest.MonkeyPatch, settings: DatasetBuilderSettings):
    calls: Dict[str, Tuple[Any, ...]] = {}
    stub = _StubAsyncFetcher(calls=calls)

    fetcher = AdvancedJQuantsFetcher(settings=settings)

    monkeypatch.setattr(fetcher, "_create_async_fetcher", lambda: stub)
    monkeypatch.setattr("builder.api.advanced_fetcher.aiohttp.ClientSession", _StubSession)

    return fetcher, calls


def test_fetch_topix(stub_fetcher) -> None:
    fetcher, calls = stub_fetcher
    df = fetcher.fetch_topix(start="2024-01-01", end="2024-01-02")
    assert not df.is_empty()
    assert calls["topix"] == ("2024-01-01", "2024-01-02")


def test_fetch_indices(stub_fetcher) -> None:
    fetcher, calls = stub_fetcher
    df = fetcher.fetch_indices(start="2024-01-01", end="2024-01-02", codes=["TOPIX"])
    assert "indices" in calls
    assert calls["indices"] == ("2024-01-01", "2024-01-02", ("TOPIX",))
    assert df.select("Code").item(0, 0) == "TOPIX"


def test_fetch_margin_and_short_data(stub_fetcher) -> None:
    fetcher, calls = stub_fetcher
    fetcher.fetch_margin_weekly(start="2024-01-01", end="2024-01-31")
    fetcher.fetch_margin_daily(start="2024-01-01", end="2024-01-02")
    fetcher.fetch_short_selling(start="2024-01-01", end="2024-01-02")
    fetcher.fetch_short_positions(start="2024-01-01", end="2024-01-02")
    fetcher.fetch_sector_short_selling(start="2024-01-01", end="2024-01-02", business_days=["20240101"])

    assert calls["margin_weekly"] == ("2024-01-01", "2024-01-31")
    assert calls["margin_daily"] == ("2024-01-01", "2024-01-02")
    assert calls["short"] == ("2024-01-01", "2024-01-02")
    assert calls["short_positions"] == ("2024-01-01", "2024-01-02")
    assert calls["sector_short"][0:2] == ("2024-01-01", "2024-01-02")


def test_fetch_miscellaneous(stub_fetcher) -> None:
    fetcher, calls = stub_fetcher
    fetcher.fetch_futures(start="2024-01-01", end="2024-01-02")
    fetcher.fetch_options(start="2024-01-01", end="2024-01-02")
    fetcher.fetch_trades_spec(start="2024-01-01", end="2024-01-02")
    fetcher.fetch_listed_info(as_of="2024-01-01")
    fetcher.fetch_trading_breakdown(start="2024-01-01", end="2024-01-02")
    fetcher.fetch_prices_am(start="2024-01-01", end="2024-01-02")
    fetcher.fetch_dividends(start="2024-01-01", end="2024-01-02")
    fetcher.fetch_fs_details(start="2024-01-01", end="2024-01-02")
    fetcher.fetch_earnings(start="2024-01-01", end="2024-01-02")

    assert calls["futures"] == ("2024-01-01", "2024-01-02")
    assert calls["options"] == ("2024-01-01", "2024-01-02")
    assert calls["trades_spec"] == ("2024-01-01", "2024-01-02")
    assert calls["listed_info"] == ("2024-01-01",)
    assert calls["breakdown"] == ("2024-01-01", "2024-01-02")
    assert calls["prices_am"] == ("2024-01-01", "2024-01-02")
    assert calls["dividends"] == ("2024-01-01", "2024-01-02")
    assert calls["fs_details"] == ("2024-01-01", "2024-01-02")
    assert calls["earnings"] == ("2024-01-01", "2024-01-02")
