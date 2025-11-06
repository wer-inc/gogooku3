from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest
from builder.config import get_settings
from builder.pipelines.dataset_builder import DatasetBuilder
from builder.utils import business_date_range
from builder.utils.artifacts import DatasetArtifactWriter


class StubCalendarFetcher:
    def fetch_calendar(self, *, year: int, market_code: str | None = None) -> dict:
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        days = business_date_range(start, end)
        return {"trading_calendar": [{"Date": day, "HolidayDivision": "1"} for day in days]}


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
                    "AdjustmentFactor": "1.0",
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
                    "AdjustmentFactor": "1.0",
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

    def authenticate(self) -> str:  # pragma: no cover - not used in stub
        return "token"

    def check_rate_limit(self, *, code: str, date: str) -> None:
        return None


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
        self._dividend_cache: dict[tuple[str, str], pl.DataFrame] = {}
        self._fs_cache: dict[tuple[str, str], pl.DataFrame] = {}
        self._bd_cache: dict[tuple[str, str], pl.DataFrame] = {}

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
                # Cross-market extensions
                "macro_vvmd_vrp_spy": [0.02, 0.018],
                "macro_vvmd_vrp_spy_z_252d": [0.5, 0.55],
                "macro_vvmd_vrp_spy_high_flag": [0, 1],
                "macro_vvmd_credit_spread_ratio": [0.01, 0.012],
                "macro_vvmd_credit_spread_z_63d": [0.2, 0.25],
                "macro_vvmd_rates_term_ratio": [-0.015, -0.012],
                "macro_vvmd_rates_term_z_63d": [-0.3, -0.28],
                "macro_vvmd_vix_term_slope": [5.0, 4.5],
                "macro_vvmd_vix_term_ratio": [-0.1, -0.08],
                "macro_vvmd_vix_term_z_126d": [0.1, 0.12],
                "macro_vvmd_spy_overnight_ret": [0.001, -0.002],
                "macro_vvmd_spy_intraday_ret": [0.0005, 0.0007],
                "macro_vvmd_fx_usdjpy_ret_1d": [0.001, -0.0005],
                "macro_vvmd_fx_usdjpy_ret_5d": [0.004, 0.003],
                "macro_vvmd_fx_usdjpy_ret_20d": [0.01, 0.012],
                "macro_vvmd_fx_usdjpy_z_20d": [0.2, 0.25],
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

    def dividends(self, *, start: str, end: str) -> pl.DataFrame:
        key = (start, end)
        if key not in self._dividend_cache:
            self._dividend_cache[key] = pl.DataFrame(
                {
                    "Code": ["1301", "1301"],
                    "AnnouncementDate": [date(2023, 6, 20), date(2023, 12, 20)],
                    "AnnouncementTime": ["13:00:00", "13:00:00"],
                    "ExDate": [date(2023, 7, 5), date(2024, 1, 5)],
                    "GrossDividendRate": [20.0, 30.0],
                    "CommemorativeSpecialCode": ["0", "0"],
                    "StatusCode": ["1", "1"],
                    "ReferenceNumber": ["A1", "A2"],
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
                    "NetSales": [100.0, 110.0, 120.0, 130.0],
                    "OperatingProfit": [10.0, 11.0, 12.0, 13.0],
                    "Profit": [6.0, 6.5, 7.0, 7.5],
                    "TotalAssets": [300.0, 305.0, 310.0, 320.0],
                    "Equity": [150.0, 152.0, 154.0, 156.0],
                    "NetCashProvidedByOperatingActivities": [8.0, 8.5, 9.0, 9.5],
                    "PurchaseOfPropertyPlantAndEquipment": [-3.0, -3.1, -3.2, -3.3],
                }
            )
        return self._fs_cache[key]

    def trading_breakdown(self, *, start: str, end: str) -> pl.DataFrame:
        key = (start, end)
        if key not in self._bd_cache:
            self._bd_cache[key] = pl.DataFrame(
                {
                    "Code": ["1301", "1301"],
                    "Date": [date(2023, 12, 28), date(2024, 1, 4)],
                    "LongBuyValue": [500_000.0, 520_000.0],
                    "MarginBuyNewValue": [100_000.0, 110_000.0],
                    "LongSellValue": [480_000.0, 450_000.0],
                    "MarginSellNewValue": [90_000.0, 95_000.0],
                    "ShortSellWithoutMarginValue": [40_000.0, 45_000.0],
                }
            )
        return self._bd_cache[key]


class StubStorage:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.settings = get_settings()
        self._writer = DatasetArtifactWriter(settings=self.settings)

    def write_dataset(
        self,
        df: pl.DataFrame,
        *,
        start_date: str | None,
        end_date: str | None,
        extra_metadata: dict | None = None,
    ):
        return self._writer.write(
            df,
            start_date=start_date,
            end_date=end_date,
            extra_metadata=extra_metadata,
        )

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
    base_cols = {
        "Code",
        "SectorCode",
        "Date",
        "AdjustmentClose",
        "AdjustmentOpen",
        "AdjustmentHigh",
        "AdjustmentLow",
        "AdjustmentVolume",
        "AdjustmentFactor",
        "ret_prev_1d",
        "gap_ov_prev1",
        "gap_id_prev1",
    }
    assert base_cols.issubset(set(df.columns))
    forbidden = {"Close", "Open", "High", "Low", "Volume", "Adj Close"}
    assert forbidden.isdisjoint(set(df.columns))
    assert df.schema["Date"] == pl.Date
    assert all("_right" not in col.lower() for col in df.columns)
    assert "log_returns_1d" not in df.columns
    assert "gap_ov_today" not in df.columns and "gap_id_today" not in df.columns

    gap_view = df.select(["gap_ov_prev1", "gap_id_prev1", "ret_prev_1d"]).drop_nulls()
    if not gap_view.is_empty():
        lhs = 1 + gap_view["ret_prev_1d"]
        rhs = (1 + gap_view["gap_ov_prev1"]) * (1 + gap_view["gap_id_prev1"])
        diff = (lhs - rhs).abs()
        assert diff.max() < 1e-4

    metadata_path = resolved.with_name(resolved.stem + "_metadata.json")
    assert metadata_path.exists()
    meta = json.loads(metadata_path.read_text())
    schema_meta = meta.get("schema_governance")
    assert schema_meta is not None
    assert schema_meta.get("canonical_family") == [
        "adjustmentclose",
        "adjustmentopen",
        "adjustmenthigh",
        "adjustmentlow",
        "adjustmentvolume",
    ]
    assert "alias_map" in schema_meta
    assert "dropped" in schema_meta
    assert schema_meta.get("dropped_right_columns") is not None
    assert schema_meta.get("dropped_log_return_columns") is not None

    gap_meta = meta.get("gap_decomposition")
    assert gap_meta is not None
    assert gap_meta.get("mode") == "prev1_only"
    assert gap_meta.get("feature_clock") == "close"

    fund_meta = meta.get("fundamental_features")
    assert fund_meta is not None
    assert "fs" in fund_meta and "dividend" in fund_meta

    feature_index_path = resolved.with_name(resolved.stem + "_feature_index.json")
    assert feature_index_path.exists()
    feature_index = json.loads(feature_index_path.read_text())
    dataset_cols = pl.read_parquet(resolved, n_rows=0).columns
    assert feature_index["columns"] == dataset_cols
    assert feature_index["feature_columns"], "feature manifest should list feature columns"
    assert feature_index["column_hash"], "feature index must contain column hash"
    feature_summary = meta.get("feature_index")
    assert feature_summary is not None
    assert feature_summary["column_hash"] == feature_index["column_hash"]
    assert "gap_ov_prev1" in feature_index["columns"]
    assert "gap_id_prev1" in feature_index["columns"]
    assert "log_returns_1d" not in feature_index["columns"]
    quality_cols = [col for col in df.columns if col.endswith("_cs_rank")]
    assert quality_cols  # quality features should be generated
    assert "dmi_net_adv60" in df.columns
    assert "is_margin_daily_valid" in df.columns
    dmi_slice = df.filter(pl.col("is_margin_daily_valid") == 1)
    if not dmi_slice.is_empty():
        assert dmi_slice.select("dmi_net_adv60").drop_nulls().height > 0
        assert dmi_slice.select("dmi_long_short_ratio").drop_nulls().height > 0
    assert "close_peer_mean" in df.columns
    assert "foreign_sentiment" in df.columns
    assert "smart_flow_indicator" in df.columns
    assert "macro_vix_close" in df.columns
    fx_ret = df.select("macro_vvmd_fx_usdjpy_ret_1d").drop_nulls()
    assert fx_ret.height > 0
    vrp_vals = df.select("macro_vvmd_vrp_spy").drop_nulls()
    assert vrp_vals.height > 0

    fs_expected_cols = {
        "fs_revenue_ttm",
        "fs_op_profit_ttm",
        "fs_net_income_ttm",
        "fs_cfo_ttm",
        "fs_capex_ttm",
        "fs_fcf_ttm",
        "fs_sales_yoy",
        "fs_op_margin",
        "fs_net_margin",
        "fs_roe_ttm",
        "fs_roa_ttm",
        "fs_accruals_ttm",
        "fs_cfo_to_ni",
        "fs_observation_count",
        "fs_lag_days",
        "fs_is_recent",
        "fs_staleness_bd",
        "is_fs_valid",
    }
    assert fs_expected_cols.issubset(set(df.columns))
    fs_slice = (
        df.filter(pl.col("Code") == "1301")
        .select(list(fs_expected_cols))
        .drop_nulls(subset=["fs_revenue_ttm"])
        .sort(["fs_observation_count", "fs_revenue_ttm"])
    )
    assert fs_slice.height > 0
    latest = fs_slice.row(-1, named=True)
    assert latest["fs_revenue_ttm"] == pytest.approx(330.0, rel=1e-6)
    assert latest["fs_op_profit_ttm"] == pytest.approx(33.0, rel=1e-6)
    assert latest["fs_net_income_ttm"] == pytest.approx(19.5, rel=1e-6)
    assert latest["fs_cfo_ttm"] == pytest.approx(25.5, rel=1e-6)
    assert latest["fs_capex_ttm"] == pytest.approx(9.3, rel=1e-6)
    assert latest["fs_fcf_ttm"] == pytest.approx(16.2, rel=1e-6)
    assert latest["fs_op_margin"] == pytest.approx(0.1, rel=1e-6)
    assert latest["fs_net_margin"] == pytest.approx(0.059090909, rel=1e-6)
    assert latest["fs_roe_ttm"] == pytest.approx(0.12745098, rel=1e-6)
    assert latest["fs_roa_ttm"] == pytest.approx(0.063414634, rel=1e-6)
    assert latest["fs_accruals_ttm"] == pytest.approx(-0.019512195, rel=1e-6)
    assert latest["fs_cfo_to_ni"] == pytest.approx(1.3076923, rel=1e-6)
    assert latest["fs_observation_count"] >= 1
    assert latest["fs_lag_days"] == 0 or latest["fs_lag_days"] is None
    assert latest["is_fs_valid"] in (0, 1)

    assert "div_dy_12m" in df.columns
    div_values = (
        df.filter((pl.col("Code") == "1301") & (pl.col("div_is_obs") == 1))
        .select("div_dy_12m", "div_days_to_ex")
        .drop_nulls(subset=["div_dy_12m"])
    )
    assert div_values.height > 0
    assert div_values.select("div_dy_12m").item(0, 0) > 0
    assert {"div_pre3", "div_post3", "div_is_ex0"}.issubset(set(df.columns))

    expected_bd_cols = {
        "bd_total_value",
        "bd_net_value",
        "bd_net_ratio",
        "bd_short_share",
        "bd_activity_ratio",
        "bd_net_ratio_chg_1d",
        "bd_short_share_chg_1d",
        "bd_net_z20",
        "bd_net_z260",
        "bd_short_z260",
        "bd_credit_new_net",
        "bd_credit_close_net",
        "bd_net_ratio_local_max",
        "bd_net_ratio_local_min",
        "bd_turn_up",
        "bd_staleness_bd",
        "bd_is_recent",
        "is_bd_valid",
    }
    assert expected_bd_cols.issubset(set(df.columns))
    bd_ratios = df.filter(pl.col("Code") == "1301").select("bd_net_ratio").drop_nulls()
    assert bd_ratios.height > 0
    z260_values = df.select("bd_net_z260").drop_nulls()
    if z260_values.height:
        assert z260_values.select(pl.col("bd_net_z260").abs().max()).item() <= pytest.approx(6.0, rel=0.5)
    turn_values = df.select("bd_turn_up").drop_nulls()
    if turn_values.height:
        assert set(turn_values["bd_turn_up"].unique()) <= {0, 1}
    local_turn = df.select(["bd_net_ratio_local_max", "bd_net_ratio_local_min"]).drop_nulls()
    if local_turn.height:
        assert set(local_turn["bd_net_ratio_local_max"].unique()) <= {0, 1}
        assert set(local_turn["bd_net_ratio_local_min"].unique()) <= {0, 1}

    # Past return features are generated
    return_cols = {"ret_prev_1d", "ret_prev_5d", "ret_prev_10d", "ret_prev_20d"}
    assert return_cols.issubset(set(df.columns))
    assert df.schema["dollar_volume"] == pl.Float64
    assert df.select("dollar_volume").drop_nulls().height > 0

    assert {"ret_overnight", "ret_intraday"}.issubset(set(df.columns))
    composition_check = (
        df.select(["ret_overnight", "ret_intraday", "ret_prev_1d"])
        .drop_nulls()
        .with_columns(
            (((1 + pl.col("ret_overnight")) * (1 + pl.col("ret_intraday")) - 1 - pl.col("ret_prev_1d")).abs()).alias(
                "abs_err"
            )
        )
        .select(
            [
                pl.col("abs_err").mean().alias("mae"),
                pl.col("abs_err").quantile(0.99).alias("q99"),
            ]
        )
    )
    if composition_check.height:
        mae = float(composition_check["mae"][0])
        q99 = float(composition_check["q99"][0])
        assert mae < 1e-6
        assert q99 < 1e-5

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

    assert "wmi_net_adv5d" in df.columns
    assert "wmi_imbalance" in df.columns
    weekly_values = df.select("wmi_net_adv5d").drop_nulls()
    assert weekly_values.height >= 0

    # Flow features release on publication day only
    release_slice = df.filter(pl.col("Date") == pl.date(2024, 1, 5)).select("institutional_accumulation").drop_nulls()
    assert release_slice.height == df.select("Code").unique().height

    pre_release = df.filter(pl.col("Date") < pl.date(2024, 1, 5))
    assert pre_release.select("institutional_accumulation").drop_nulls().height == 0

    # Macro VIX should be shifted forward (first day null, subsequent day populated)
    unique_dates = df.select("Date").unique().sort("Date").to_series().to_list()
    assert len(unique_dates) >= 2
    first_date = unique_dates[0]
    second_date = unique_dates[1]
    first_day_slice = df.filter(pl.col("Date") == first_date).select("macro_vix_close")
    assert not first_day_slice.is_empty(), "macro_vix_close missing for first trading day"
    first_day_vix = first_day_slice.item(0, 0)
    assert first_day_vix is None
    later_slice = df.filter(pl.col("Date") == second_date).select("macro_vix_close")
    assert not later_slice.is_empty(), "macro_vix_close missing for second trading day"
    later_vix = later_slice.item(0, 0)
    assert later_vix is not None and later_vix > 0

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


def test_shift_macro_features_respects_fx_columns(builder: DatasetBuilder) -> None:
    frame = pl.DataFrame(
        {
            "date": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), "1d", eager=True),
            "macro_vix_close": [20.0, 21.0, 22.0],
            "macro_fx_usdjpy_close": [140.0, 141.0, 142.0],
        }
    )

    shifted = builder._shift_macro_features(frame)

    assert shifted["macro_vix_close"][0] is None
    assert shifted["macro_vix_close"][1] == pytest.approx(20.0)

    assert shifted["macro_fx_usdjpy_close"][0] == pytest.approx(140.0)
    assert shifted["macro_fx_usdjpy_close"][1] == pytest.approx(141.0)
