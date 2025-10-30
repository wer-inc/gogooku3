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
        self.margin_calls: list[str] = []

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

    def fetch_margin_daily_window(self, *, dates: list[str]) -> list[dict[str, str]]:
        self.margin_calls.extend(dates)
        return [
            {
                "code": "1301",
                "date": "2024-01-01",
                "margin_balance": "100000",
                "short_balance": "50000",
            }
        ]


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
    dataset_builder = DatasetBuilder(settings=settings, fetcher=fetcher)
    yield dataset_builder


def test_dataset_builder_end_to_end(builder: DatasetBuilder, settings: DatasetBuilderSettings) -> None:
    output_path = builder.build(start="2024-01-01", end="2024-01-02")

    assert output_path.exists()
    df = pl.read_parquet(output_path)
    assert df.shape[0] == 2
    assert "margin_balance" in df.columns
    assert df.filter(pl.col("code") == "1301").select("margin_balance").item(0, 0) == pytest.approx(100000.0)
    assert "margin_net" in df.columns

    # Ensure cache hit on subsequent build
    builder.build(start="2024-01-01", end="2024-01-02")
