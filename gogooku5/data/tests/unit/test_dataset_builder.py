from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from builder.pipelines.dataset_builder import DatasetBuilder


class StubFetcher:
    def __init__(self) -> None:
        self.quote_calls = 0

    def fetch_listed_info(self) -> dict[str, list[dict[str, str]]]:
        return {
            "info": [
                {"code": "1301", "sectorCode": "FOOD"},
                {"code": "1305", "sectorCode": "ETF"},
            ]
        }

    def fetch_daily_quotes(self, *, code: str, from_: str, to: str) -> dict[str, list[dict[str, str]]]:
        self.quote_calls += 1
        return {
            "daily_quotes": [
                {
                    "code": code,
                    "date": from_,
                    "close": "100",
                    "open": "95",
                    "high": "105",
                    "low": "94",
                    "volume": "1000",
                }
            ]
        }

    def fetch_quotes_paginated(self, *, code: str, from_: str, to: str) -> list[dict[str, str]]:
        return self.fetch_daily_quotes(code=code, from_=from_, to=to)["daily_quotes"]

    def fetch_margin_daily_window(self, *, dates: list[str]) -> list[dict[str, str]]:
        return [
            {
                "code": "1301",
                "date": date,
                "margin_balance": "100000",
                "short_balance": "50000",
            }
            for date in dates
        ]

    def authenticate(self) -> str:  # pragma: no cover - not used in stub
        return "token"


class StubStorage:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def ensure_remote_symlink(self, *, target: str) -> None:
        self.calls.append(target)


@pytest.fixture
def builder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DatasetBuilder:
    monkeypatch.setenv("JQUANTS_AUTH_EMAIL", "user@example.com")
    monkeypatch.setenv("JQUANTS_AUTH_PASSWORD", "secret")
    monkeypatch.setenv("DATA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("DATA_CACHE_DIR", str(tmp_path / "cache"))

    storage = StubStorage()
    fetcher = StubFetcher()
    ds_builder = DatasetBuilder(fetcher=fetcher, storage=storage)
    ds_builder.fetcher = fetcher
    ds_builder.storage = storage
    return ds_builder


def test_dataset_builder_writes_parquet(builder: DatasetBuilder) -> None:
    output_path = builder.build(start="2024-01-01", end="2024-01-31")

    assert output_path.exists()
    df = pl.read_parquet(output_path)
    assert df.shape[0] == 2  # two codes * one row each
    base_cols = {"code", "sector_code", "date", "close", "open", "high", "low", "volume"}
    assert base_cols.issubset(set(df.columns))
    assert df.schema["date"] == pl.Date
    quality_cols = [col for col in df.columns if col.endswith("_cs_rank")]
    assert quality_cols  # quality features should be generated
    assert "margin_net" in df.columns
    assert df.filter(pl.col("code") == "1301").select("margin_net").item(0, 0) is not None
    assert "close_peer_mean" in df.columns

    # Cache should prevent additional quote fetches on subsequent runs
    fetcher: StubFetcher = builder.fetcher  # type: ignore[assignment]
    initial_calls = fetcher.quote_calls
    builder.build(start="2024-01-01", end="2024-01-31")
    assert fetcher.quote_calls == initial_calls

    storage: StubStorage = builder.storage  # type: ignore[assignment]
    assert storage.calls
