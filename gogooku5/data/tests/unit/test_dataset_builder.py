from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from builder.config import get_settings
from builder.pipelines.dataset_builder import DatasetBuilder
from builder.utils.artifacts import DatasetArtifactWriter
from builder.utils import business_date_range


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

    def authenticate(self) -> str:  # pragma: no cover - not used in stub
        return "token"


class StubDataSources:
    def __init__(self) -> None:
        self.margin_calls = 0
        self.trades_calls = 0
        self.vix_calls = 0
        self._margin_cache: dict[tuple[str, str], pl.DataFrame] = {}
        self._trades_cache: dict[tuple[str, str], pl.DataFrame] = {}
        self._vix_cache: dict[tuple[str, str], pl.DataFrame] = {}

    def margin_daily(self, *, start: str, end: str) -> pl.DataFrame:
        key = (start, end)
        if key not in self._margin_cache:
            self.margin_calls += 1
            self._margin_cache[key] = pl.DataFrame(
                {
                    "code": ["1301", "1305"],
                    "date": ["2024-01-01", "2024-01-01"],
                    "margin_balance": [100_000.0, 0.0],
                    "short_balance": [50_000.0, 0.0],
                }
            )
        return self._margin_cache[key]

    def trades_spec(self, *, start: str, end: str) -> pl.DataFrame:
        key = (start, end)
        if key not in self._trades_cache:
            self.trades_calls += 1
            self._trades_cache[key] = pl.DataFrame(
                {
                    "Code": ["1301"],
                    "PublishedDate": ["2024-01-01"],
                    "ForeignersPurchaseValue": [150.0],
                    "ForeignersSalesValue": [100.0],
                    "IndividualPurchaseValue": [60.0],
                    "IndividualSalesValue": [80.0],
                    "InvestmentPurchaseValue": [40.0],
                    "InvestmentSalesValue": [30.0],
                }
            )
        return self._trades_cache[key]

    def macro_vix(self, *, start: str, end: str, force_refresh: bool = False) -> pl.DataFrame:
        key = (start, end)
        if not force_refresh and key in self._vix_cache:
            return self._vix_cache[key]
        self.vix_calls += 1
        data = pl.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-02"],
                "macro_vix_close": [20.0, 21.5],
                "macro_vix_spike": [0, 1],
            }
        )
        self._vix_cache[key] = data
        return data


class StubStorage:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.settings = get_settings()
        self._writer = DatasetArtifactWriter(settings=self.settings)

    def write_dataset(self, df: pl.DataFrame, *, start_date: str | None, end_date: str | None):
        return self._writer.write(df, start_date=start_date, end_date=end_date)

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
    data_sources = StubDataSources()
    ds_builder = DatasetBuilder(fetcher=fetcher, storage=storage, data_sources=data_sources)
    ds_builder.fetcher = fetcher
    ds_builder.storage = storage
    ds_builder.data_sources = data_sources  # ensure stub usage
    return ds_builder


def test_dataset_builder_writes_parquet(builder: DatasetBuilder) -> None:
    output_path = builder.build(start="2024-01-01", end="2024-01-31")

    assert output_path.exists()
    assert output_path.is_symlink()
    resolved = output_path.resolve(strict=True)
    df = pl.read_parquet(resolved)
    unique_dates = df.select("Date").unique().height
    unique_codes = df.select("Code").unique().height
    assert df.shape[0] == unique_dates * unique_codes
    expected_business = set(business_date_range("2024-01-01", "2024-01-31"))
    actual_dates = {d.isoformat() for d in df["Date"].to_list()}
    assert expected_business.issubset(actual_dates)
    base_cols = {"Code", "SectorCode", "Date", "Close", "Open", "High", "Low", "Volume"}
    assert base_cols.issubset(set(df.columns))
    assert df.schema["Date"] == pl.Date
    quality_cols = [col for col in df.columns if col.endswith("_cs_rank")]
    assert quality_cols  # quality features should be generated
    assert "margin_net" in df.columns
    assert df.filter(pl.col("Code") == "1301").select("margin_net").item(0, 0) is not None
    assert "close_peer_mean" in df.columns
    assert "foreign_sentiment" in df.columns
    assert "smart_flow_indicator" in df.columns
    assert "macro_vix_close" in df.columns

    # Cache should prevent additional quote fetches on subsequent runs
    fetcher: StubFetcher = builder.fetcher  # type: ignore[assignment]
    initial_calls = fetcher.quote_calls
    data_sources: StubDataSources = builder.data_sources  # type: ignore[assignment]
    initial_margin_calls = data_sources.margin_calls
    initial_trades_calls = data_sources.trades_calls
    initial_vix_calls = data_sources.vix_calls
    builder.build(start="2024-01-01", end="2024-01-31")
    assert fetcher.quote_calls == initial_calls
    assert data_sources.margin_calls == initial_margin_calls
    assert data_sources.trades_calls == initial_trades_calls
    assert data_sources.vix_calls == initial_vix_calls

    storage: StubStorage = builder.storage  # type: ignore[assignment]
    assert storage.calls
    metadata_symlink = storage.settings.data_output_dir / storage.settings.latest_metadata_symlink
    assert metadata_symlink.exists()
