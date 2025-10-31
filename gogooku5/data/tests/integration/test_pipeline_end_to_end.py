from __future__ import annotations

from pathlib import Path
from typing import Iterator

import polars as pl
import pytest
from builder.config import DatasetBuilderSettings
from builder.pipelines.dataset_builder import DatasetBuilder


class MockJQuantsFetcher:
    def __init__(self) -> None:
        self.auth_called = 0
        self.quote_calls: list[tuple[str, str, str]] = []

    def fetch_listed_info(self) -> dict[str, list[dict[str, str]]]:
        return {
            "info": [
                {"code": "1301", "sectorCode": "FOOD"},
                {"code": "1305", "sectorCode": "ETF"},
            ]
        }

    def fetch_quotes_paginated(self, *, code: str, from_: str, to: str) -> list[dict[str, str]]:
        self.quote_calls.append((code, from_, to))
        return [
            {
                "code": code,
                "date": "2024-01-01",
                "close": "100",
                "open": "95",
                "high": "105",
                "low": "94",
                "volume": "1000",
            }
        ]

    # Legacy interface no longer used but kept for compatibility
    def fetch_margin_daily_window(self, *, dates: list[str]) -> list[dict[str, str]]:
        return []


class MockDataSources:
    def __init__(self) -> None:
        self.margin_calls = 0
        self._cache: dict[tuple[str, str], pl.DataFrame] = {}
        self._trades_cache: dict[tuple[str, str], pl.DataFrame] = {}
        self.trades_calls = 0
        self.vix_calls = 0
        self._vix_cache: dict[tuple[str, str], pl.DataFrame] = {}

    def margin_daily(self, *, start: str, end: str) -> pl.DataFrame:
        key = (start, end)
        if key not in self._cache:
            self.margin_calls += 1
            self._cache[key] = pl.DataFrame(
                {
                    "code": ["1301", "1305"],
                    "date": ["2024-01-01", "2024-01-01"],
                    "margin_balance": [100_000.0, 0.0],
                    "short_balance": [50_000.0, 0.0],
                }
            )
        return self._cache[key]

    def trades_spec(self, *, start: str, end: str) -> pl.DataFrame:
        key = (start, end)
        if key not in self._trades_cache:
            self.trades_calls += 1
            self._trades_cache[key] = pl.DataFrame(
                {
                    "Code": ["1301"],
                    "PublishedDate": ["2024-01-01"],
                    "ForeignersPurchaseValue": [120.0],
                    "ForeignersSalesValue": [80.0],
                    "IndividualPurchaseValue": [60.0],
                    "IndividualSalesValue": [70.0],
                }
            )
        return self._trades_cache[key]

    def macro_vix(self, *, start: str, end: str, force_refresh: bool = False) -> pl.DataFrame:
        key = (start, end)
        if not force_refresh and key in self._vix_cache:
            return self._vix_cache[key]
        self.vix_calls += 1
        df = pl.DataFrame(
            {
                "Date": ["2024-01-01"],
                "macro_vix_close": [20.0],
                "macro_vix_spike": [0],
            }
        )
        self._vix_cache[key] = df
        return df


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DatasetBuilderSettings:
    monkeypatch.setenv("JQUANTS_AUTH_EMAIL", "user@example.com")
    monkeypatch.setenv("JQUANTS_AUTH_PASSWORD", "secret")
    output = tmp_path / "output"
    cache = tmp_path / "cache"
    return DatasetBuilderSettings(
        jquants_auth_email="user@example.com",
        jquants_auth_password="secret",
        data_output_dir=output,
        data_cache_dir=cache,
    )


@pytest.fixture
def builder(settings: DatasetBuilderSettings) -> Iterator[DatasetBuilder]:
    fetcher = MockJQuantsFetcher()
    data_sources = MockDataSources()
    dataset_builder = DatasetBuilder(settings=settings, fetcher=fetcher, data_sources=data_sources)
    dataset_builder.data_sources = data_sources
    yield dataset_builder


def test_dataset_builder_end_to_end(builder: DatasetBuilder, settings: DatasetBuilderSettings) -> None:
    output_path = builder.build(start="2024-01-01", end="2024-01-02")

    assert output_path.exists()
    assert output_path.is_symlink()
    df = pl.read_parquet(output_path.resolve(strict=True))
    assert df.shape[0] == 2
    assert "margin_balance" in df.columns
    assert df.filter(pl.col("Code") == "1301").select("margin_balance").item(0, 0) == pytest.approx(100000.0)
    assert "margin_net" in df.columns
    assert "foreign_sentiment" in df.columns
    assert "smart_flow_indicator" in df.columns
    assert "macro_vix_close" in df.columns

    # Ensure cache hit on subsequent build
    builder.build(start="2024-01-01", end="2024-01-02")
    data_sources: MockDataSources = builder.data_sources  # type: ignore[assignment]
    assert data_sources.margin_calls == 1
    assert data_sources.trades_calls == 1
    assert data_sources.vix_calls == 1
