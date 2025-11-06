from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Iterator

import polars as pl
import pytest
from builder.config import DatasetBuilderSettings
from builder.pipelines.dataset_builder import DatasetBuilder
from builder.utils.storage import StorageClient


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

    def check_rate_limit(self, *, code: str, date: str) -> None:
        return None

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
                "AdjustmentFactor": "1.0",
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
                "AdjustmentFactor": "1.0",
            },
        ]


class MockDataSources:
    def __init__(self) -> None:
        self.margin_calls = 0
        self._cache: dict[tuple[str, str], pl.DataFrame] = {}
        self._trades_cache: dict[tuple[str, str], pl.DataFrame] = {}
        self.trades_calls = 0
        self.vix_calls = 0
        self._vix_cache: dict[tuple[str, str], pl.DataFrame] = {}
        self._dividend_cache: dict[tuple[str, str], pl.DataFrame] = {}
        self._fs_cache: dict[tuple[str, str], pl.DataFrame] = {}
        self._bd_cache: dict[tuple[str, str], pl.DataFrame] = {}

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

    def macro_global_regime(self, *, start: str, end: str, force_refresh: bool = False) -> pl.DataFrame:
        """Mock macro global regime features (14 VVMD features).

        Returns minimal DataFrame with Date and 14 macro_vvmd_* columns.
        """
        return pl.DataFrame(
            {
                "Date": ["2024-01-01"],
                # V (Volatility): 4 features
                "macro_vvmd_spy_vol_20d": [0.15],
                "macro_vvmd_qqq_vol_20d": [0.18],
                "macro_vvmd_vix_z": [0.5],
                "macro_vvmd_vol_regime": [1.0],
                # Vlm (Volume): 2 features
                "macro_vvmd_spy_volume_ma20": [1000000.0],
                "macro_vvmd_qqq_volume_ma20": [800000.0],
                # M (Momentum): 5 features
                "macro_vvmd_spy_mom_20d": [0.05],
                "macro_vvmd_qqq_mom_20d": [0.06],
                "macro_vvmd_dxy_z": [-0.3],
                "macro_vvmd_btc_rel_mom": [0.10],
                "macro_vvmd_trend_strength": [0.7],
                # D (Demand): 3 features
                "macro_vvmd_btc_vol_20d": [0.45],
                "macro_vvmd_risk_appetite": [0.6],
                "macro_vvmd_flight_to_quality": [0.2],
                # Cross-market extensions
                "macro_vvmd_vrp_spy": [0.02],
                "macro_vvmd_vrp_spy_z_252d": [0.5],
                "macro_vvmd_vrp_spy_high_flag": [0],
                "macro_vvmd_credit_spread_ratio": [0.01],
                "macro_vvmd_credit_spread_z_63d": [0.2],
                "macro_vvmd_rates_term_ratio": [-0.015],
                "macro_vvmd_rates_term_z_63d": [-0.3],
                "macro_vvmd_vix_term_slope": [5.0],
                "macro_vvmd_vix_term_ratio": [-0.1],
                "macro_vvmd_vix_term_z_126d": [0.1],
                "macro_vvmd_spy_overnight_ret": [0.001],
                "macro_vvmd_spy_intraday_ret": [0.0005],
                "macro_vvmd_fx_usdjpy_ret_1d": [0.001],
                "macro_vvmd_fx_usdjpy_ret_5d": [0.004],
                "macro_vvmd_fx_usdjpy_ret_20d": [0.01],
                "macro_vvmd_fx_usdjpy_z_20d": [0.2],
            }
        )

    def dividends(self, *, start: str, end: str) -> pl.DataFrame:
        key = (start, end)
        if key not in self._dividend_cache:
            self._dividend_cache[key] = pl.DataFrame(
                {
                    "Code": ["1301"],
                    "AnnouncementDate": [date(2023, 12, 20)],
                    "AnnouncementTime": ["13:00:00"],
                    "ExDate": [date(2024, 1, 5)],
                    "GrossDividendRate": [30.0],
                    "CommemorativeSpecialCode": ["0"],
                    "StatusCode": ["1"],
                    "ReferenceNumber": ["REF"],
                }
            )
        return self._dividend_cache[key]

    def fs_details(self, *, start: str, end: str) -> pl.DataFrame:
        key = (start, end)
        if key not in self._fs_cache:
            self._fs_cache[key] = pl.DataFrame(
                {
                    "Code": ["1301"] * 4,
                    "DisclosedDate": [
                        date(2023, 3, 31),
                        date(2023, 6, 30),
                        date(2023, 9, 30),
                        date(2023, 12, 31),
                    ],
                    "DisclosedTime": ["15:00:00"] * 4,
                    "NetSales": [80.0, 90.0, 95.0, 105.0],
                    "OperatingProfit": [8.0, 9.0, 9.5, 10.5],
                    "Profit": [4.5, 5.0, 5.5, 6.0],
                    "TotalAssets": [250.0, 252.0, 255.0, 258.0],
                    "Equity": [130.0, 131.0, 133.0, 135.0],
                    "NetCashProvidedByOperatingActivities": [6.0, 6.5, 7.0, 7.5],
                    "PurchaseOfPropertyPlantAndEquipment": [-2.5, -2.6, -2.7, -2.8],
                }
            )
        return self._fs_cache[key]

    def trading_breakdown(self, *, start: str, end: str) -> pl.DataFrame:
        key = (start, end)
        if key not in self._bd_cache:
            self._bd_cache[key] = pl.DataFrame(
                {
                    "Code": ["1301"],
                    "Date": [date(2023, 12, 28)],
                    "LongBuyValue": [300_000.0],
                    "MarginBuyNewValue": [80_000.0],
                    "LongSellValue": [250_000.0],
                    "MarginSellNewValue": [60_000.0],
                    "ShortSellWithoutMarginValue": [35_000.0],
                }
            )
        return self._bd_cache[key]


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DatasetBuilderSettings:
    monkeypatch.setenv("JQUANTS_AUTH_EMAIL", "user@example.com")
    monkeypatch.setenv("JQUANTS_AUTH_PASSWORD", "secret")
    # BATCH-2B Safety: Prevent test data from overwriting production 'latest' symlinks
    monkeypatch.setenv("NO_LATEST_SYMLINK", "1")
    os.environ["WARMUP_DAYS"] = "0"
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
    dataset_builder.storage = StorageClient(settings=settings)
    yield dataset_builder


def test_dataset_builder_end_to_end(builder: DatasetBuilder, settings: DatasetBuilderSettings) -> None:
    output_path = builder.build(start="2024-01-01", end="2024-01-02")

    if not output_path.exists():
        # When NO_LATEST_SYMLINK=1 is active the builder may return the logical
        # latest symlink path. Fall back to the concrete parquet artifact.
        candidates = sorted(settings.data_output_dir.glob("ml_dataset_*_full.parquet"))
        assert candidates, "Expected dataset parquet artifact to exist"
        output_path = candidates[-1]

    resolved = output_path.resolve(strict=True) if output_path.is_symlink() else output_path
    df = pl.read_parquet(resolved)
    assert df.shape[0] == 2
    assert "dmi_net_adv60" in df.columns
    assert "dmi_imbalance" in df.columns
    assert "dmi_long_short_ratio" in df.columns
    assert "is_margin_daily_valid" in df.columns
    assert "foreign_sentiment" in df.columns
    assert "smart_flow_indicator" in df.columns
    assert "macro_vix_close" in df.columns
    assert {"fs_revenue_ttm", "fs_op_margin", "fs_roe_ttm"}.issubset(set(df.columns))
    assert df.filter(pl.col("Code") == "1301").select("fs_revenue_ttm").drop_nulls().height > 0
    assert "div_dy_12m" in df.columns
    assert df.filter(pl.col("Code") == "1301").select("div_dy_12m").drop_nulls().height > 0
    assert {
        "bd_net_ratio",
        "bd_short_share",
        "bd_activity_ratio",
        "bd_net_z260",
        "bd_credit_new_net",
        "bd_net_ratio_local_max",
    }.issubset(set(df.columns))
    assert df.filter(pl.col("Code") == "1301").select("bd_net_ratio").drop_nulls().height > 0

    # Ensure cache hit on subsequent build
    builder.build(start="2024-01-01", end="2024-01-02")
    data_sources: MockDataSources = builder.data_sources  # type: ignore[assignment]
    assert data_sources.margin_calls == 1
    assert data_sources.trades_calls == 1
    assert data_sources.vix_calls == 1
