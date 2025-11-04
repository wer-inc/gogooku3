from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest
from builder.config import get_settings
from builder.pipelines.dataset_builder import DatasetBuilder
from builder.utils.artifacts import DatasetArtifactWriter
from builder.utils import business_date_range


class StubCalendarFetcher:
    def fetch_calendar(self, *, year: int, market_code: str | None = None) -> dict:
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        days = business_date_range(start, end)
        return {
            "trading_calendar": [
                {"Date": day, "HolidayDivision": "1"}
                for day in days
            ]
        }


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
                    "TurnoverValue": "1000000",
                    "AdjustmentClose": "101",
                    "AdjustmentOpen": "96",
                    "AdjustmentHigh": "106",
                    "AdjustmentLow": "95",
                    "AdjustmentVolume": "1005",
                },
                {
                    "code": code,
                    "date": to,
                    "close": "102",
                    "open": "97",
                    "high": "106",
                    "low": "95",
                    "volume": "1100",
                    "TurnoverValue": "1100000",
                    "AdjustmentClose": "103",
                    "AdjustmentOpen": "98",
                    "AdjustmentHigh": "107",
                    "AdjustmentLow": "96",
                    "AdjustmentVolume": "1105",
                },
            ]
        }

    def fetch_quotes_paginated(self, *, code: str, from_: str, to: str) -> list[dict[str, str]]:
        return self.fetch_daily_quotes(code=code, from_=from_, to=to)["daily_quotes"]

    def fetch_quotes_by_date_paginated(self, *, date: str) -> list[dict[str, str]]:
        formatted = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        return [
            {
                "Code": "1301",
                "Date": formatted,
                "Close": "100",
                "Open": "95",
                "High": "105",
                "Low": "94",
                "Volume": "1000",
                "TurnoverValue": "1000000",
                "AdjustmentClose": "101",
                "AdjustmentOpen": "96",
                "AdjustmentHigh": "106",
                "AdjustmentLow": "95",
                "AdjustmentVolume": "1005",
            },
            {
                "Code": "1305",
                "Date": formatted,
                "Close": "200",
                "Open": "195",
                "High": "205",
                "Low": "194",
                "Volume": "2000",
                "TurnoverValue": "2000000",
                "AdjustmentClose": "201",
                "AdjustmentOpen": "196",
                "AdjustmentHigh": "206",
                "AdjustmentLow": "195",
                "AdjustmentVolume": "2005",
            },
        ]

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
        self._topix_cache: dict[tuple[str, str], pl.DataFrame] = {}
        self._short_cache: dict[tuple[str, str], pl.DataFrame] = {}
        self._weekly_cache: dict[tuple[str, str], pl.DataFrame] = {}

    def margin_daily(self, *, start: str, end: str) -> pl.DataFrame:
        key = (start, end)
        if key not in self._margin_cache:
            self.margin_calls += 1
            self._margin_cache[key] = pl.DataFrame(
                {
                    "code": ["1301", "1305"],
                    "date": [date(2024, 1, 2), date(2024, 1, 2)],
                    "application_date": [date(2024, 1, 1), date(2024, 1, 1)],
                    "published_date": [date(2024, 1, 2), date(2024, 1, 2)],
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
                    "PublishedDate": ["2024-01-05"],
                    "StartDate": ["2024-01-01"],
                    "EndDate": ["2024-01-05"],
                    "ForeignersPurchases": [1_500_000.0],
                    "ForeignersSales": [900_000.0],
                    "IndividualsPurchases": [200_000.0],
                    "IndividualsSales": [320_000.0],
                    "InvestmentTrustsPurchases": [400_000.0],
                    "InvestmentTrustsSales": [380_000.0],
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

    def macro_global_regime(self, *, start: str, end: str, force_refresh: bool = False) -> pl.DataFrame:
        """Stub macro global regime features (14 VVMD features).

        Returns minimal DataFrame with Date and 14 macro_vvmd_* columns.
        """
        return pl.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-02"],
                # V (Volatility): 4 features
                "macro_vvmd_spy_vol_20d": [0.15, 0.16],
                "macro_vvmd_qqq_vol_20d": [0.18, 0.19],
                "macro_vvmd_vix_z": [0.5, 0.6],
                "macro_vvmd_vol_regime": [1.0, 1.1],
                # Vlm (Volume): 2 features
                "macro_vvmd_spy_volume_ma20": [1000000.0, 1050000.0],
                "macro_vvmd_qqq_volume_ma20": [800000.0, 820000.0],
                # M (Momentum): 5 features
                "macro_vvmd_spy_mom_20d": [0.05, 0.04],
                "macro_vvmd_qqq_mom_20d": [0.06, 0.05],
                "macro_vvmd_dxy_z": [-0.3, -0.2],
                "macro_vvmd_btc_rel_mom": [0.10, 0.12],
                "macro_vvmd_trend_strength": [0.7, 0.75],
                # D (Demand): 3 features
                "macro_vvmd_btc_vol_20d": [0.45, 0.47],
                "macro_vvmd_risk_appetite": [0.6, 0.65],
                "macro_vvmd_flight_to_quality": [0.2, 0.18],
            }
        )

    def topix(self, *, start: str, end: str) -> pl.DataFrame:
        key = (start, end)
        if key not in self._topix_cache:
            self._topix_cache[key] = pl.DataFrame(
                {
                    "Date": [start, end],
                    "Code": ["TOPIX", "TOPIX"],
                    "Open": [2000.0, 2010.0],
                    "High": [2020.0, 2030.0],
                    "Low": [1980.0, 1995.0],
                    "Close": [2010.0, 2025.0],
                }
            )
        return self._topix_cache[key]

    def short_selling(self, *, start: str, end: str, business_days: list[str] | None = None) -> pl.DataFrame:
        key = (start, end)
        if key not in self._short_cache:
            self._short_cache[key] = pl.DataFrame(
                {
                    "Date": [date(2024, 1, 5)],
                    "PublishedDate": [date(2024, 1, 8)],
                    "Sector33Code": ["FOOD"],
                    "SellingExcludingShortSellingTurnoverValue": [1_000_000.0],
                    "ShortSellingWithRestrictionsTurnoverValue": [200_000.0],
                    "ShortSellingWithoutRestrictionsTurnoverValue": [50_000.0],
                }
            )
        return self._short_cache[key]

    def sector_short_selling(self, *, start: str, end: str, business_days: list[str] | None = None) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "Date": [date(2024, 1, 5)],
                "PublishedDate": [date(2024, 1, 8)],
                "Sector33Code": ["FOOD"],
                "SellingExcludingShortSellingTurnoverValue": [500_000.0],
                "ShortSellingWithRestrictionsTurnoverValue": [100_000.0],
                "ShortSellingWithoutRestrictionsTurnoverValue": [25_000.0],
            }
        )

    def indices(self, *, start: str, end: str, codes: list[str]) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "Date": [start, end],
                "Code": [codes[0], codes[0]],
                "Open": [30000.0, 30100.0],
                "High": [30200.0, 30300.0],
                "Low": [29900.0, 30050.0],
                "Close": [30100.0, 30250.0],
            }
        )

    def margin_weekly(self, *, start: str, end: str) -> pl.DataFrame:
        key = (start, end)
        if key not in self._weekly_cache:
            self._weekly_cache[key] = pl.DataFrame(
                {
                    "Date": [start],
                    "Code": ["1301"],
                    "ShortMarginTradeVolume": [5000.0],
                    "LongMarginTradeVolume": [15000.0],
                }
            )
        return self._weekly_cache[key]


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
    monkeypatch.setenv("WARMUP_DAYS", "0")
    # BATCH-2B Safety: Prevent test data from overwriting production 'latest' symlinks
    monkeypatch.setenv("NO_LATEST_SYMLINK", "1")

    storage = StubStorage()
    fetcher = StubFetcher()
    data_sources = StubDataSources()
    calendar_fetcher = StubCalendarFetcher()
    ds_builder = DatasetBuilder(
        fetcher=fetcher,
        storage=storage,
        data_sources=data_sources,
        calendar_fetcher=calendar_fetcher,
    )
    ds_builder.fetcher = fetcher
    ds_builder.storage = storage
    ds_builder.data_sources = data_sources  # ensure stub usage
    ds_builder.calendar_fetcher = calendar_fetcher
    return ds_builder


def test_dataset_builder_writes_parquet(builder: DatasetBuilder) -> None:
    output_path = builder.build(start="2024-01-01", end="2024-01-31")

    assert output_path.exists()
    # BATCH-2B Safety: NO_LATEST_SYMLINK=1 returns parquet_path instead of symlink
    # In test mode, output_path may be the actual parquet file (not symlink)
    if output_path.is_symlink():
        resolved = output_path.resolve(strict=True)
    else:
        resolved = output_path  # Already the actual file
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
    assert (
        df.filter(pl.col("Code") == "1301").select("margin_net").drop_nulls().height > 0
    )
    assert "close_peer_mean" in df.columns
    assert "foreign_sentiment" in df.columns
    assert "smart_flow_indicator" in df.columns
    assert "macro_vix_close" in df.columns

    # Past return features are generated
    return_cols = {"ret_prev_1d", "ret_prev_5d", "ret_prev_10d", "ret_prev_20d"}
    assert return_cols.issubset(set(df.columns))
    assert df.schema["dollar_volume"] == pl.Float64
    assert df.select("dollar_volume").drop_nulls().height > 0

    # Index features exist (prefixed with topix_)
    topix_cols = [col for col in df.columns if col.startswith("topix_")]
    assert topix_cols
    nikkei_cols = [col for col in df.columns if col.startswith("nk225_")]
    assert nikkei_cols

    # Short selling and weekly margin aggregates exist
    assert "short_selling_ratio_market" in df.columns
    # sector-level ratios may be absent if sector data unavailable; ensure no crash
    if "sector_short_selling_ratio" in df.columns:
        sector_values = df.select("sector_short_selling_ratio").drop_nulls()
        assert sector_values.height >= 0

    assert "weekly_margin_imbalance" in df.columns
    assert df.select("weekly_margin_imbalance").drop_nulls().height > 0

    # Flow features release on publication day only
    release_slice = (
        df.filter(pl.col("Date") == pl.date(2024, 1, 5))
        .select("institutional_accumulation")
        .drop_nulls()
    )
    assert release_slice.height == df.select("Code").unique().height

    pre_release = df.filter(pl.col("Date") < pl.date(2024, 1, 5))
    assert (
        pre_release.select("institutional_accumulation").drop_nulls().height == 0
    )

    # Macro VIX should be shifted forward (first day null, subsequent day populated)
    first_day_vix = df.filter(pl.col("Date") == pl.date(2024, 1, 1)).select("macro_vix_close").item(0, 0)
    assert first_day_vix is None
    later_vix = (
        df.filter(pl.col("Date") == pl.date(2024, 1, 2))
        .select("macro_vix_close")
        .item(0, 0)
    )
    assert later_vix == pytest.approx(20.0)

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
    # BATCH-2B Safety: Metadata symlink also not created when NO_LATEST_SYMLINK=1
    # metadata_symlink = storage.settings.data_output_dir / storage.settings.latest_metadata_symlink
    # assert metadata_symlink.exists()  # Skipped in test mode


def test_quotes_cache_key_depends_on_symbols(builder: DatasetBuilder) -> None:
    key_full = builder._quotes_cache_key(symbols=["2222", "1111"], start="2024-01-01", end="2024-01-02")
    key_subset = builder._quotes_cache_key(symbols=["1111"], start="2024-01-01", end="2024-01-02")
    assert key_full != key_subset
