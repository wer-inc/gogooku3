"""Unit tests for daily margin interest feature processing."""

from __future__ import annotations

import datetime as _dt
from datetime import datetime

import polars as pl
import pytest

from src.gogooku3.features.margin_daily import (
    _add_business_days_jp,
    add_daily_core_features,
    add_daily_margin_block,
    add_publish_reason_flags,
    asof_attach_to_daily,
    attach_adv20_and_scale,
    build_daily_effective,
    create_interaction_features,
)


@pytest.fixture
def sample_daily_margin_data():
    """Sample daily margin interest data from API."""
    return pl.DataFrame([
        {
            "Code": "1301",
            "PublishedDate": datetime(2024, 3, 15),
            "ApplicationDate": datetime(2024, 3, 14),
            "PublishReason": {
                "Restricted": "0",
                "DailyPublication": "1",
                "Monitoring": "0",
                "RestrictedByJSF": "0",
                "PrecautionByJSF": "1",
                "UnclearOrSecOnAlert": "0"
            },
            "ShortMarginOutstanding": 1500.0,
            "LongMarginOutstanding": 3000.0,
            "DailyChangeShortMarginOutstanding": 100.0,
            "DailyChangeLongMarginOutstanding": -200.0,
            "ShortMarginOutstandingListedShareRatio": 0.75,
            "LongMarginOutstandingListedShareRatio": 1.50,
            "ShortLongRatio": 50.0,
            "ShortNegotiableMarginOutstanding": 500.0,
            "ShortStandardizedMarginOutstanding": 1000.0,
            "LongNegotiableMarginOutstanding": 800.0,
            "LongStandardizedMarginOutstanding": 2200.0,
            "DailyChangeShortNegotiableMarginOutstanding": 50.0,
            "DailyChangeShortStandardizedMarginOutstanding": 50.0,
            "DailyChangeLongNegotiableMarginOutstanding": -100.0,
            "DailyChangeLongStandardizedMarginOutstanding": -100.0,
            "TSEMarginBorrowingAndLendingRegulationClassification": "002",
        },
        {
            "Code": "1301",
            "PublishedDate": datetime(2024, 3, 18),  # Monday (next business day after Friday)
            "ApplicationDate": datetime(2024, 3, 15),
            "PublishReason": {
                "Restricted": "1",
                "DailyPublication": "1",
                "Monitoring": "0",
                "RestrictedByJSF": "0",
                "PrecautionByJSF": "0",
                "UnclearOrSecOnAlert": "0"
            },
            "ShortMarginOutstanding": 1600.0,
            "LongMarginOutstanding": 2800.0,
            "DailyChangeShortMarginOutstanding": 100.0,
            "DailyChangeLongMarginOutstanding": -200.0,
            "ShortMarginOutstandingListedShareRatio": 0.80,
            "LongMarginOutstandingListedShareRatio": 1.40,
            "ShortLongRatio": 57.14,
            "ShortNegotiableMarginOutstanding": 550.0,
            "ShortStandardizedMarginOutstanding": 1050.0,
            "LongNegotiableMarginOutstanding": 750.0,
            "LongStandardizedMarginOutstanding": 2050.0,
            "DailyChangeShortNegotiableMarginOutstanding": 50.0,
            "DailyChangeShortStandardizedMarginOutstanding": 50.0,
            "DailyChangeLongNegotiableMarginOutstanding": -50.0,
            "DailyChangeLongStandardizedMarginOutstanding": -150.0,
            "TSEMarginBorrowingAndLendingRegulationClassification": "003",
        }
    ])


@pytest.fixture
def sample_quotes_data():
    """Sample daily quotes data."""
    return pl.DataFrame([
        {
            "Code": "1301",
            "Date": datetime(2024, 3, 14),
            "Close": 1000.0,
            "AdjustmentVolume": 50000,
            "returns_1d": 0.01,
            "returns_3d": 0.025,
        },
        {
            "Code": "1301",
            "Date": datetime(2024, 3, 15),
            "Close": 1010.0,
            "AdjustmentVolume": 55000,
            "returns_1d": 0.015,
            "returns_3d": 0.030,
        },
        {
            "Code": "1301",
            "Date": datetime(2024, 3, 18),  # Monday
            "Close": 1025.0,
            "AdjustmentVolume": 60000,
            "returns_1d": 0.010,
            "returns_3d": 0.020,
        },
        {
            "Code": "1301",
            "Date": datetime(2024, 3, 19),  # Tuesday
            "Close": 1035.0,
            "AdjustmentVolume": 52000,
            "returns_1d": 0.005,
            "returns_3d": 0.015,
        }
    ])


@pytest.fixture
def sample_adv20_data():
    """Sample ADV20 data for scaling."""
    return pl.DataFrame([
        {
            "Code": "1301",
            "Date": datetime(2024, 3, 14),
            "ADV20_shares": 50000.0,
        },
        {
            "Code": "1301",
            "Date": datetime(2024, 3, 15),
            "ADV20_shares": 51000.0,
        },
        {
            "Code": "1301",
            "Date": datetime(2024, 3, 18),
            "ADV20_shares": 52000.0,
        },
        {
            "Code": "1301",
            "Date": datetime(2024, 3, 19),
            "ADV20_shares": 52500.0,
        }
    ])


class TestBusinessDayCalculations:
    """Test Japanese business day calculations."""

    def test_add_business_days_jp(self):
        """Test adding business days correctly handles weekends."""
        # Friday + 1 business day = Monday
        friday = _dt.date(2024, 3, 15)
        monday = _add_business_days_jp(friday, 1)
        assert monday == _dt.date(2024, 3, 18)

        # Thursday + 1 business day = Friday
        thursday = _dt.date(2024, 3, 14)
        friday_result = _add_business_days_jp(thursday, 1)
        assert friday_result == _dt.date(2024, 3, 15)

        # Monday + 5 business days = next Monday (skips weekend)
        monday = _dt.date(2024, 3, 18)
        next_monday = _add_business_days_jp(monday, 5)
        assert next_monday == _dt.date(2024, 3, 25)


class TestDailyEffectiveCalculation:
    """Test effective_start date calculation."""

    def test_build_daily_effective(self, sample_daily_margin_data):
        """Test that effective_start is computed as T+1 business day."""
        result = build_daily_effective(sample_daily_margin_data)

        # Check that we have effective_start column
        assert "effective_start" in result.columns

        # Check that effective dates are T+1 business days
        expected_dates = [
            _dt.date(2024, 3, 18),  # March 15 (Fri) + 1 BD = March 18 (Mon)
            _dt.date(2024, 3, 19),  # March 18 (Mon) + 1 BD = March 19 (Tue)
        ]

        actual_dates = result.select("effective_start").to_series().to_list()
        assert actual_dates == expected_dates


class TestCoreFeatureGeneration:
    """Test core daily margin feature generation."""

    def test_add_daily_core_features(self, sample_daily_margin_data):
        """Test core feature calculations."""
        result = add_daily_core_features(sample_daily_margin_data)

        # Check that all expected features exist
        expected_features = [
            "dmi_long", "dmi_short", "dmi_net", "dmi_total",
            "dmi_credit_ratio", "dmi_imbalance", "dmi_short_long_ratio",
            "dmi_std_share_long", "dmi_std_share_short",
            "dmi_neg_share_long", "dmi_neg_share_short",
            "dmi_d_long_1d", "dmi_d_short_1d", "dmi_d_net_1d", "dmi_d_ratio_1d"
        ]

        for feature in expected_features:
            assert feature in result.columns, f"Missing feature: {feature}"

        # Test specific calculations for first row
        first_row = result.filter(pl.col("Code") == "1301").head(1)

        # Basic calculations
        assert first_row.select("dmi_long").item() == 3000.0
        assert first_row.select("dmi_short").item() == 1500.0
        assert first_row.select("dmi_net").item() == 1500.0  # 3000 - 1500
        assert first_row.select("dmi_total").item() == 4500.0  # 3000 + 1500
        assert first_row.select("dmi_credit_ratio").item() == pytest.approx(2.0, rel=1e-3)  # 3000/1500
        assert first_row.select("dmi_imbalance").item() == pytest.approx(0.333, rel=1e-2)  # 1500/4500


class TestPublishReasonFlags:
    """Test PublishReason flag expansion."""

    def test_add_publish_reason_flags(self, sample_daily_margin_data):
        """Test that PublishReason flags are correctly expanded."""
        result = add_publish_reason_flags(sample_daily_margin_data)

        # Check flag columns exist
        flag_cols = [
            "dmi_reason_restricted", "dmi_reason_dailypublication", "dmi_reason_monitoring",
            "dmi_reason_restrictedbyj sf", "dmi_reason_precautionbyjsf", "dmi_reason_unclearorsectonalert"
        ]

        # Check regulation level mapping
        assert "dmi_tse_reg_level" in result.columns

        # Test first row flags (DailyPublication=1, PrecautionByJSF=1)
        first_row = result.filter(pl.col("Code") == "1301").head(1)
        assert first_row.select("dmi_reason_dailypublication").item() == 1
        assert first_row.select("dmi_reason_precautionbyjsf").item() == 1
        assert first_row.select("dmi_reason_restricted").item() == 0

        # Test reason count
        assert first_row.select("dmi_reason_count").item() == 2

        # Test TSE regulation level (002 -> 2)
        assert first_row.select("dmi_tse_reg_level").item() == 2


class TestADVScaling:
    """Test ADV20 scaling functionality."""

    def test_attach_adv20_and_scale(self, sample_adv20_data):
        """Test ADV20 scaling of margin features."""
        # Create sample margin data with effective_start
        margin_data = pl.DataFrame([
            {
                "Code": "1301",
                "effective_start": datetime(2024, 3, 18),
                "dmi_long": 3000.0,
                "dmi_short": 1500.0,
                "dmi_total": 4500.0,
                "dmi_d_long_1d": 100.0,
                "dmi_d_short_1d": 50.0,
                "dmi_d_net_1d": 150.0,
            }
        ])

        result = attach_adv20_and_scale(margin_data, sample_adv20_data)

        # Check scaled features exist
        scaled_features = [
            "dmi_long_to_adv20", "dmi_short_to_adv20", "dmi_total_to_adv20",
            "dmi_d_long_to_adv1d", "dmi_d_short_to_adv1d", "dmi_d_net_to_adv1d"
        ]

        for feature in scaled_features:
            assert feature in result.columns, f"Missing scaled feature: {feature}"

        # Test calculation (effective_start 2024-03-18 should join with 2024-03-18 ADV)
        expected_adv = 52000.0
        first_row = result.head(1)

        assert first_row.select("dmi_long_to_adv20").item() == pytest.approx(3000.0 / expected_adv, rel=1e-3)
        assert first_row.select("dmi_short_to_adv20").item() == pytest.approx(1500.0 / expected_adv, rel=1e-3)


class TestAsOfJoin:
    """Test as-of join functionality for attaching to daily quotes."""

    def test_asof_attach_to_daily(self, sample_quotes_data):
        """Test as-of join with daily quotes."""
        # Sample daily margin data with effective_start
        daily_margin = pl.DataFrame([
            {
                "Code": "1301",
                "effective_start": datetime(2024, 3, 18),  # Monday
                "PublishedDate": datetime(2024, 3, 15),
                "ApplicationDate": datetime(2024, 3, 14),
                "dmi_long": 3000.0,
                "dmi_short": 1500.0,
            }
        ])

        result = asof_attach_to_daily(sample_quotes_data, daily_margin)

        # Check timing features exist
        timing_features = ["dmi_impulse", "dmi_days_since_pub", "dmi_days_since_app", "is_dmi_valid"]
        for feature in timing_features:
            assert feature in result.columns, f"Missing timing feature: {feature}"

        # Check data availability by date
        # March 14, 15: no data (before effective_start)
        # March 18: effective_start, should have impulse=1
        # March 19: should have impulse=0

        march_18_row = result.filter((pl.col("Code") == "1301") & (pl.col("Date") == datetime(2024, 3, 18)))
        march_19_row = result.filter((pl.col("Code") == "1301") & (pl.col("Date") == datetime(2024, 3, 19)))

        assert march_18_row.select("dmi_impulse").item() == 1  # Impulse on effective date
        assert march_18_row.select("is_dmi_valid").item() == 1  # Valid data
        assert march_18_row.select("dmi_days_since_pub").item() == 3  # 18 - 15 = 3 days

        assert march_19_row.select("dmi_impulse").item() == 0  # No impulse day after
        assert march_19_row.select("is_dmi_valid").item() == 1  # Still valid (same data)
        assert march_19_row.select("dmi_days_since_pub").item() == 4  # 19 - 15 = 4 days


class TestFullPipeline:
    """Test the complete daily margin processing pipeline."""

    def test_add_daily_margin_block_complete(self, sample_quotes_data, sample_daily_margin_data, sample_adv20_data):
        """Test complete pipeline integration."""
        result = add_daily_margin_block(
            quotes=sample_quotes_data,
            daily_df=sample_daily_margin_data,
            adv20_df=sample_adv20_data,
            enable_z_scores=True,
        )

        # Check that we have margin features
        margin_features = ["dmi_long", "dmi_short", "is_dmi_valid"]
        for feature in margin_features:
            assert feature in result.columns, f"Missing feature: {feature}"

        # Check that original quotes data is preserved
        original_features = ["Code", "Date", "Close", "AdjustmentVolume"]
        for feature in original_features:
            assert feature in result.columns, f"Missing original feature: {feature}"

        # Check data types
        assert result.schema["is_dmi_valid"] == pl.Int8
        assert result.schema["dmi_impulse"] == pl.Int8

        # Ensure no data leakage: margin data should only appear from effective_start onwards
        march_14_row = result.filter((pl.col("Code") == "1301") & (pl.col("Date") == datetime(2024, 3, 14)))
        assert march_14_row.select("is_dmi_valid").item() == 0  # No data before effective_start

    def test_add_daily_margin_block_empty_data(self, sample_quotes_data):
        """Test pipeline with empty daily margin data."""
        empty_df = pl.DataFrame()

        result = add_daily_margin_block(
            quotes=sample_quotes_data,
            daily_df=empty_df,
            adv20_df=None,
            enable_z_scores=True,
        )

        # Should return original quotes with null margin features
        assert len(result) == len(sample_quotes_data)
        assert "is_dmi_valid" in result.columns
        assert result.select("is_dmi_valid").null_count().item() == 0  # Should be filled with 0s


class TestInteractionFeatures:
    """Test interaction feature creation."""

    def test_create_interaction_features(self):
        """Test interaction feature calculations."""
        # Sample data with required columns
        df = pl.DataFrame([
            {
                "Code": "1301",
                "dmi_short_to_adv20": 0.1,  # Above 80th percentile if others are lower
                "dmi_long_to_adv20": 0.05,
                "dmi_z26_long": 2.0,  # Above 1.5
                "dmi_d_net_1d": 100.0,  # Positive
                "dmi_impulse": 1,
                "returns_1d": 0.01,  # Positive
                "returns_3d": 0.02,  # Positive
            },
            {
                "Code": "1302",
                "dmi_short_to_adv20": 0.02,  # Low
                "dmi_long_to_adv20": 0.01,
                "dmi_z26_long": 0.5,  # Low
                "dmi_d_net_1d": -50.0,  # Negative
                "dmi_impulse": 0,
                "returns_1d": -0.01,  # Negative
                "returns_3d": -0.015,  # Negative
            }
        ])

        result = create_interaction_features(df)

        # Check interaction features exist
        interaction_features = [
            "dmi_short_squeeze_setup_d",
            "dmi_long_unwind_risk_d",
            "dmi_with_trend_d"
        ]

        for feature in interaction_features:
            assert feature in result.columns, f"Missing interaction feature: {feature}"

        # Test logic for first row (should trigger some interactions)
        first_row = result.filter(pl.col("Code") == "1301").head(1)

        # Short squeeze setup: high short position + positive return
        # dmi_long_unwind_risk_d: high z-score + negative return
        # dmi_with_trend_d: trend alignment

        # Note: Actual values depend on quantile calculations across the small dataset
        assert first_row.select("dmi_with_trend_d").item() == 1  # Both net flow and returns are positive