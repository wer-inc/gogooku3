from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from builder.api.data_sources import DataSourceManager
from builder.config import DatasetBuilderSettings
from builder.utils import CacheManager


class StubAdvancedFetcher:
    def __init__(self) -> None:
        self.calls = 0
        self.flow_calls = 0

    def fetch_margin_daily(self, *, start: str, end: str) -> pl.DataFrame:
        self.calls += 1
        return pl.DataFrame(
            {
                "Code": ["1301", "1305"],
                "ApplicationDate": ["2024-01-01", "2024-01-01"],
                "LongMarginOutstanding": [100_000.0, 0.0],
                "ShortMarginOutstanding": [50_000.0, 0.0],
            }
        )

    def fetch_trades_spec(self, *, start: str, end: str) -> pl.DataFrame:
        self.flow_calls += 1
        return pl.DataFrame(
            {
                "Code": ["1301"],
                "PublishedDate": ["2024-01-01"],
                "ForeignersPurchaseValue": [150.0],
                "ForeignersSalesValue": [100.0],
                "IndividualPurchaseValue": [60.0],
                "IndividualSalesValue": [80.0],
            }
        )


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DatasetBuilderSettings:
    monkeypatch.setenv("JQUANTS_AUTH_EMAIL", "user@example.com")
    monkeypatch.setenv("JQUANTS_AUTH_PASSWORD", "secret")
    monkeypatch.setenv("DATA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("DATA_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CACHE_TTL_DAYS_MARGIN_DAILY", "5")
    return DatasetBuilderSettings(
        jquants_auth_email="user@example.com",
        jquants_auth_password="secret",
        data_output_dir=tmp_path / "output",
        data_cache_dir=tmp_path / "cache",
    )


def test_margin_daily_cached(settings: DatasetBuilderSettings, tmp_path: Path) -> None:
    cache = CacheManager(settings=settings)
    fetcher = StubAdvancedFetcher()
    manager = DataSourceManager(settings=settings, cache=cache, fetcher=fetcher)

    df1 = manager.margin_daily(start="2024-01-01", end="2024-01-31")
    assert df1.columns == ["code", "date", "margin_balance", "short_balance"]
    assert df1.height == 2
    assert fetcher.calls == 1

    df2 = manager.margin_daily(start="2024-01-01", end="2024-01-31")
    assert df2.equals(df1)
    # Cached result should prevent additional fetches within TTL
    assert fetcher.calls == 1


def test_trades_spec_cached(settings: DatasetBuilderSettings, tmp_path: Path) -> None:
    cache = CacheManager(settings=settings)
    fetcher = StubAdvancedFetcher()
    manager = DataSourceManager(settings=settings, cache=cache, fetcher=fetcher)

    df1 = manager.trades_spec(start="2024-01-01", end="2024-01-31")
    assert "ForeignersPurchaseValue" in df1.columns or not df1.is_empty()
    assert fetcher.flow_calls == 1

    df2 = manager.trades_spec(start="2024-01-01", end="2024-01-31")
    assert df2.equals(df1)
    assert fetcher.flow_calls == 1


def test_macro_vix_cached(settings: DatasetBuilderSettings, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = CacheManager(settings=settings)
    manager = DataSourceManager(settings=settings, cache=cache, fetcher=StubAdvancedFetcher())

    sample_history = pl.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02"],
            "Close": [20.0, 21.0],
        }
    )

    monkeypatch.setattr("builder.api.data_sources.load_vix_history", lambda *a, **k: sample_history)
    df1 = manager.macro_vix(start="2024-01-01", end="2024-01-02")
    assert "macro_vix_close" in df1.columns

    df2 = manager.macro_vix(start="2024-01-01", end="2024-01-02")
    assert df2.equals(df1)
