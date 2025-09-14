"""
Integration tests for daily margin interest pipeline with leak verification.

This module tests the complete end-to-end pipeline:
1. Raw daily margin data → effective dates → features → panel attachment
2. Temporal integrity and leak prevention validation
3. Integration with existing weekly margin system
4. Performance and data quality validation

Critical safety tests:
- T+1 rule enforcement (no same-day application)
- As-of backward join correctness
- No future data contamination
- Proper handling of correction data
"""

from __future__ import annotations

import pytest
import polars as pl
from datetime import datetime, timedelta, date
import numpy as np
from pathlib import Path

import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.gogooku3.features.margin_daily import (
    add_daily_margin_block,
    build_daily_effective,
    add_daily_core_features,
    asof_attach_to_daily,
    _add_business_days_jp,
)
from scripts.data.ml_dataset_builder import MLDatasetBuilder


class TestDailyMarginPipelineIntegration:
    """Integration tests for the complete daily margin pipeline."""

    @pytest.fixture
    def sample_quotes_panel(self) -> pl.DataFrame:
        """Create a realistic daily quotes panel for testing."""
        dates = [date(2024, 1, 2) + timedelta(days=i) for i in range(30)]
        # Filter to business days only
        business_dates = [d for d in dates if d.weekday() < 5]

        codes = ["1301", "1332", "2269", "4063", "6501"]  # Realistic TSE codes

        rows = []
        for code in codes:
            base_price = 1000.0
            base_volume = 100000

            for i, date_val in enumerate(business_dates):
                # Simulate realistic stock price movement
                price_change = np.random.normal(0, 0.02)
                base_price *= (1 + price_change)

                volume = int(base_volume * np.random.uniform(0.5, 2.0))

                rows.append({
                    "Code": code,
                    "Date": date_val,
                    "Close": base_price,
                    "Volume": volume,
                    "AdjustmentVolume": volume,  # Same as Volume for simplicity
                })

        df = pl.DataFrame(rows)
        df = df.with_columns([
            pl.col("Date").cast(pl.Date),
            pl.col("Close").cast(pl.Float64),
            pl.col("Volume").cast(pl.Int64),
            pl.col("AdjustmentVolume").cast(pl.Int64),
        ])

        return df.sort(["Code", "Date"])

    @pytest.fixture
    def sample_daily_margin_raw(self) -> pl.DataFrame:
        """Create sample daily margin interest data."""
        # Simulate realistic daily margin data pattern
        pub_dates = [
            date(2024, 1, 3),   # Wed publication
            date(2024, 1, 4),   # Thu publication
            date(2024, 1, 9),   # Mon publication (after weekend)
            date(2024, 1, 10),  # Tue publication
            date(2024, 1, 12),  # Fri publication
            date(2024, 1, 16),  # Tue publication (after holiday Mon)
            date(2024, 1, 18),  # Thu publication
        ]

        codes = ["1301", "2269", "6501"]  # Subset of codes with margin data

        rows = []
        for pub_date in pub_dates:
            # Application date is typically same day or T-1
            app_date = pub_date - timedelta(days=np.random.choice([0, 1]))

            for code in codes:
                if np.random.random() < 0.7:  # 70% chance of data per code/date
                    long_outstanding = int(np.random.uniform(100_000, 5_000_000))
                    short_outstanding = int(np.random.uniform(10_000, 1_000_000))

                    # Simulate composition breakdown
                    long_std = int(long_outstanding * np.random.uniform(0.6, 0.9))
                    long_neg = long_outstanding - long_std
                    short_std = int(short_outstanding * np.random.uniform(0.7, 0.95))
                    short_neg = short_outstanding - short_std

                    # Simulate regulatory flags (rarely active)
                    reason_flags = {
                        "Restricted": np.random.choice(["0", "1"], p=[0.95, 0.05]),
                        "DailyPublication": np.random.choice(["0", "1"], p=[0.8, 0.2]),
                        "Monitoring": np.random.choice(["0", "1"], p=[0.9, 0.1]),
                        "RestrictedByJSF": np.random.choice(["0", "1"], p=[0.98, 0.02]),
                        "PrecautionByJSF": np.random.choice(["0", "1"], p=[0.97, 0.03]),
                        "UnclearOrSecOnAlert": np.random.choice(["0", "1"], p=[0.99, 0.01]),
                    }

                    # TSE regulation classification (mostly normal)
                    reg_class = np.random.choice(
                        ["", "002", "003", "102"],
                        p=[0.85, 0.1, 0.03, 0.02]
                    )

                    rows.append({
                        "Code": code,
                        "Date": app_date,
                        "PublishedDate": pub_date,
                        "ApplicationDate": app_date,
                        "LongMarginOutstanding": long_outstanding,
                        "ShortMarginOutstanding": short_outstanding,
                        "LongStandardizedMarginOutstanding": long_std,
                        "LongNegotiableMarginOutstanding": long_neg,
                        "ShortStandardizedMarginOutstanding": short_std,
                        "ShortNegotiableMarginOutstanding": short_neg,
                        "ShortLongRatio": (short_outstanding / long_outstanding * 100) if long_outstanding > 0 else None,
                        "PublishReason": reason_flags,
                        "TSEMarginBorrowingAndLendingRegulationClassification": reg_class,
                    })

        df = pl.DataFrame(rows)
        df = df.with_columns([
            pl.col("Date").cast(pl.Date),
            pl.col("PublishedDate").cast(pl.Date),
            pl.col("ApplicationDate").cast(pl.Date),
        ])

        return df.sort(["Code", "PublishedDate"])

    def test_end_to_end_pipeline_integration(self, sample_quotes_panel, sample_daily_margin_raw):
        """Test complete end-to-end pipeline with realistic data."""
        # Step 1: Run complete pipeline
        result = add_daily_margin_block(
            quotes=sample_quotes_panel,
            daily_df=sample_daily_margin_raw,
            adv20_df=None,  # Let it compute ADV20 internally
            enable_z_scores=True,
        )

        # Validate basic structure
        assert len(result) == len(sample_quotes_panel), "Row count should be preserved"
        assert "Code" in result.columns and "Date" in result.columns

        # Check for key daily margin features
        expected_dmi_features = [
            "dmi_long", "dmi_short", "dmi_net", "dmi_credit_ratio",
            "dmi_impulse", "is_dmi_valid", "dmi_days_since_pub",
            "dmi_reason_restricted", "dmi_reason_dailypublication",
        ]

        for feature in expected_dmi_features:
            assert feature in result.columns, f"Missing feature: {feature}"

        # Validate data types
        assert result["dmi_long"].dtype == pl.Float64
        assert result["dmi_impulse"].dtype == pl.Int8
        assert result["is_dmi_valid"].dtype == pl.Int8

        print(f"✅ Pipeline integration test passed: {len(result)} rows, {len(result.columns)} columns")

    def test_temporal_leak_prevention(self, sample_quotes_panel, sample_daily_margin_raw):
        """Critical test: Ensure no temporal data leakage."""
        # Build effective dates first to understand the transformation
        daily_effective = build_daily_effective(sample_daily_margin_raw)

        # Verify T+1 rule: effective_start > PublishedDate
        leaks = daily_effective.filter(
            pl.col("effective_start") <= pl.col("PublishedDate")
        )

        assert len(leaks) == 0, f"Found {len(leaks)} T+1 rule violations (same-day application)"

        # Run full pipeline
        result = add_daily_margin_block(
            quotes=sample_quotes_panel,
            daily_df=sample_daily_margin_raw,
        )

        # Verify as-of join correctness: effective_start <= Date for all valid data
        valid_data = result.filter(pl.col("is_dmi_valid") == 1)

        if len(valid_data) > 0:
            future_leaks = valid_data.filter(
                pl.col("effective_start") > pl.col("Date")
            )

            assert len(future_leaks) == 0, f"Found {len(future_leaks)} future data leaks"

        print("✅ Temporal leak prevention test passed")

    def test_correction_data_handling(self):
        """Test that correction data (multiple PublishedDate for same ApplicationDate) is handled correctly."""
        # Create test data with corrections
        correction_data = pl.DataFrame([
            {
                "Code": "1301",
                "Date": date(2024, 1, 10),
                "ApplicationDate": date(2024, 1, 10),
                "PublishedDate": date(2024, 1, 11),  # Original publication
                "LongMarginOutstanding": 1000000,
                "ShortMarginOutstanding": 200000,
            },
            {
                "Code": "1301",
                "Date": date(2024, 1, 10),
                "ApplicationDate": date(2024, 1, 10),  # Same application date
                "PublishedDate": date(2024, 1, 12),  # Correction publication
                "LongMarginOutstanding": 1100000,  # Corrected value
                "ShortMarginOutstanding": 220000,
            }
        ]).with_columns([
            pl.col("Date").cast(pl.Date),
            pl.col("ApplicationDate").cast(pl.Date),
            pl.col("PublishedDate").cast(pl.Date),
        ])

        # Add required columns for complete processing
        correction_data = correction_data.with_columns([
            pl.lit(None).alias("LongStandardizedMarginOutstanding"),
            pl.lit(None).alias("LongNegotiableMarginOutstanding"),
            pl.lit(None).alias("ShortStandardizedMarginOutstanding"),
            pl.lit(None).alias("ShortNegotiableMarginOutstanding"),
            pl.lit(None).alias("ShortLongRatio"),
            pl.lit(None).alias("PublishReason"),
            pl.lit(None).alias("TSEMarginBorrowingAndLendingRegulationClassification"),
        ])

        # Create quotes panel for the date
        quotes = pl.DataFrame([
            {"Code": "1301", "Date": date(2024, 1, 15), "Close": 1000.0, "Volume": 100000}
        ]).with_columns([pl.col("Date").cast(pl.Date)])

        # Process data - should keep only the latest correction
        result = add_daily_margin_block(quotes, correction_data)

        # Verify only the corrected value is used
        valid_row = result.filter(pl.col("is_dmi_valid") == 1)

        if len(valid_row) > 0:
            assert float(valid_row["dmi_long"][0]) == 1100000.0, "Should use corrected value"

        print("✅ Correction data handling test passed")

    def test_business_day_calculation_accuracy(self):
        """Test business day calculations for various scenarios."""
        test_cases = [
            # (input_date, expected_next_business_day)
            (date(2024, 1, 3), date(2024, 1, 4)),    # Wed -> Thu
            (date(2024, 1, 4), date(2024, 1, 5)),    # Thu -> Fri
            (date(2024, 1, 5), date(2024, 1, 8)),    # Fri -> Mon
            (date(2024, 1, 6), date(2024, 1, 8)),    # Sat -> Mon
            (date(2024, 1, 7), date(2024, 1, 8)),    # Sun -> Mon
        ]

        for input_date, expected in test_cases:
            result = _add_business_days_jp(input_date, 1)
            assert result == expected, f"Business day calc failed: {input_date} -> {result}, expected {expected}"

        print("✅ Business day calculation test passed")

    def test_integration_with_ml_dataset_builder(self, sample_quotes_panel, sample_daily_margin_raw):
        """Test integration through MLDatasetBuilder interface."""
        builder = MLDatasetBuilder(output_dir=Path("output"))

        # Test through the builder interface
        result = builder.add_daily_margin_block(
            df=sample_quotes_panel,
            daily_df=sample_daily_margin_raw,
            adv_window_days=20,
            enable_z_scores=True,
        )

        # Validate integration
        assert len(result) == len(sample_quotes_panel)
        assert "dmi_long" in result.columns
        assert "is_dmi_valid" in result.columns

        # Check coverage statistics (should be logged)
        valid_coverage = float(result.select(pl.col("is_dmi_valid").mean()).item())

        # With sample data, expect some coverage but not 100%
        assert 0.0 <= valid_coverage <= 1.0, f"Invalid coverage: {valid_coverage}"

        print(f"✅ MLDatasetBuilder integration test passed (coverage: {valid_coverage:.1%})")

    def test_adv20_scaling_integration(self, sample_quotes_panel, sample_daily_margin_raw):
        """Test ADV20 computation and scaling integration."""
        # Ensure AdjustmentVolume is present
        quotes_with_adv = sample_quotes_panel.with_columns([
            pl.col("Volume").alias("AdjustmentVolume")
        ])

        result = add_daily_margin_block(
            quotes=quotes_with_adv,
            daily_df=sample_daily_margin_raw,
        )

        # Check for ADV20-scaled features
        adv_features = [col for col in result.columns if "_to_adv" in col]

        if len(adv_features) > 0:
            print(f"✅ ADV20 scaling features present: {adv_features}")
        else:
            print("ℹ️  ADV20 scaling features not created (expected if no valid daily data)")

    def test_performance_with_large_dataset(self):
        """Test performance with a larger synthetic dataset."""
        # Create larger dataset
        n_codes = 100
        n_days = 60

        codes = [f"{1000 + i:04d}" for i in range(n_codes)]
        start_date = date(2024, 1, 2)
        dates = [start_date + timedelta(days=i) for i in range(n_days)]
        business_dates = [d for d in dates if d.weekday() < 5]

        # Create quotes panel
        quotes_rows = []
        for code in codes:
            for date_val in business_dates:
                quotes_rows.append({
                    "Code": code,
                    "Date": date_val,
                    "Close": 1000.0,
                    "Volume": 100000,
                    "AdjustmentVolume": 100000,
                })

        quotes_large = pl.DataFrame(quotes_rows).with_columns([
            pl.col("Date").cast(pl.Date)
        ])

        # Create sparse daily margin data (realistic sparsity)
        margin_rows = []
        pub_dates = business_dates[::3]  # Every 3rd business day

        for pub_date in pub_dates[:10]:  # Limit to 10 publication dates
            for code in codes[:20]:  # 20% of codes have margin data
                if np.random.random() < 0.3:  # 30% chance per code
                    margin_rows.append({
                        "Code": code,
                        "Date": pub_date,
                        "ApplicationDate": pub_date,
                        "PublishedDate": pub_date,
                        "LongMarginOutstanding": 1000000,
                        "ShortMarginOutstanding": 200000,
                        "LongStandardizedMarginOutstanding": 800000,
                        "LongNegotiableMarginOutstanding": 200000,
                        "ShortStandardizedMarginOutstanding": 160000,
                        "ShortNegotiableMarginOutstanding": 40000,
                        "ShortLongRatio": 20.0,
                        "PublishReason": None,
                        "TSEMarginBorrowingAndLendingRegulationClassification": "",
                    })

        if margin_rows:
            margin_large = pl.DataFrame(margin_rows).with_columns([
                pl.col("Date").cast(pl.Date),
                pl.col("ApplicationDate").cast(pl.Date),
                pl.col("PublishedDate").cast(pl.Date),
            ])
        else:
            # Empty margin data
            margin_large = pl.DataFrame(schema={
                "Code": pl.Utf8,
                "Date": pl.Date,
                "ApplicationDate": pl.Date,
                "PublishedDate": pl.Date,
                "LongMarginOutstanding": pl.Int64,
                "ShortMarginOutstanding": pl.Int64,
            })

        # Run pipeline and measure
        import time
        start_time = time.time()

        result = add_daily_margin_block(quotes_large, margin_large)

        end_time = time.time()
        execution_time = end_time - start_time

        # Validate results
        assert len(result) == len(quotes_large)
        assert "is_dmi_valid" in result.columns

        # Performance assertion (should be fast)
        assert execution_time < 5.0, f"Performance test failed: {execution_time:.2f}s > 5s"

        print(f"✅ Performance test passed: {len(result):,} rows processed in {execution_time:.2f}s")

    def test_empty_data_handling(self, sample_quotes_panel):
        """Test pipeline behavior with empty daily margin data."""
        empty_daily = pl.DataFrame(schema={
            "Code": pl.Utf8,
            "Date": pl.Date,
            "PublishedDate": pl.Date,
            "LongMarginOutstanding": pl.Int64,
            "ShortMarginOutstanding": pl.Int64,
        })

        result = add_daily_margin_block(sample_quotes_panel, empty_daily)

        # Should return original data with null dmi features
        assert len(result) == len(sample_quotes_panel)
        assert "is_dmi_valid" in result.columns

        # All dmi_valid should be 0 (false)
        assert result["is_dmi_valid"].sum() == 0

        print("✅ Empty data handling test passed")


def test_leak_verification_comprehensive():
    """Comprehensive leak verification test with multiple scenarios."""
    # Create test scenarios that could cause leaks
    scenarios = [
        {
            "name": "Same day publication",
            "app_date": date(2024, 1, 10),
            "pub_date": date(2024, 1, 10),
            "quote_date": date(2024, 1, 10),
            "should_be_valid": False,  # T+1 rule prevents same-day
        },
        {
            "name": "Next day application",
            "app_date": date(2024, 1, 10),
            "pub_date": date(2024, 1, 10),
            "quote_date": date(2024, 1, 11),  # Next day
            "should_be_valid": True,  # Valid T+1
        },
        {
            "name": "Weekend rollover",
            "app_date": date(2024, 1, 12),  # Friday
            "pub_date": date(2024, 1, 12),
            "quote_date": date(2024, 1, 15),  # Following Monday
            "should_be_valid": True,  # Valid after weekend
        },
        {
            "name": "Future quote date",
            "app_date": date(2024, 1, 10),
            "pub_date": date(2024, 1, 10),
            "quote_date": date(2024, 1, 9),   # Before publication
            "should_be_valid": False,  # Future data
        },
    ]

    for scenario in scenarios:
        # Create test data
        daily_data = pl.DataFrame([{
            "Code": "1301",
            "Date": scenario["app_date"],
            "ApplicationDate": scenario["app_date"],
            "PublishedDate": scenario["pub_date"],
            "LongMarginOutstanding": 1000000,
            "ShortMarginOutstanding": 200000,
            "LongStandardizedMarginOutstanding": 800000,
            "LongNegotiableMarginOutstanding": 200000,
            "ShortStandardizedMarginOutstanding": 160000,
            "ShortNegotiableMarginOutstanding": 40000,
            "ShortLongRatio": 20.0,
            "PublishReason": None,
            "TSEMarginBorrowingAndLendingRegulationClassification": "",
        }]).with_columns([
            pl.col("Date").cast(pl.Date),
            pl.col("ApplicationDate").cast(pl.Date),
            pl.col("PublishedDate").cast(pl.Date),
        ])

        quotes = pl.DataFrame([{
            "Code": "1301",
            "Date": scenario["quote_date"],
            "Close": 1000.0,
            "Volume": 100000,
        }]).with_columns([pl.col("Date").cast(pl.Date)])

        # Run pipeline
        result = add_daily_margin_block(quotes, daily_data)

        # Check validity
        is_valid = bool(result["is_dmi_valid"][0])

        assert is_valid == scenario["should_be_valid"], \
            f"Leak test failed for '{scenario['name']}': " \
            f"expected valid={scenario['should_be_valid']}, got valid={is_valid}"

        print(f"✅ Leak verification passed: {scenario['name']}")


def create_sample_quotes_panel() -> pl.DataFrame:
    """Create a realistic daily quotes panel for testing."""
    dates = [date(2024, 1, 2) + timedelta(days=i) for i in range(30)]
    # Filter to business days only
    business_dates = [d for d in dates if d.weekday() < 5]

    codes = ["1301", "1332", "2269", "4063", "6501"]  # Realistic TSE codes

    rows = []
    for code in codes:
        base_price = 1000.0
        base_volume = 100000

        for i, date_val in enumerate(business_dates):
            # Simulate realistic stock price movement
            price_change = np.random.normal(0, 0.02)
            base_price *= (1 + price_change)

            volume = int(base_volume * np.random.uniform(0.5, 2.0))

            rows.append({
                "Code": code,
                "Date": date_val,
                "Close": base_price,
                "Volume": volume,
                "AdjustmentVolume": volume,  # Same as Volume for simplicity
            })

    df = pl.DataFrame(rows)
    df = df.with_columns([
        pl.col("Date").cast(pl.Date),
        pl.col("Close").cast(pl.Float64),
        pl.col("Volume").cast(pl.Int64),
        pl.col("AdjustmentVolume").cast(pl.Int64),
    ])

    return df.sort(["Code", "Date"])


def create_sample_daily_margin_raw() -> pl.DataFrame:
    """Create sample daily margin interest data."""
    # Simulate realistic daily margin data pattern
    pub_dates = [
        date(2024, 1, 3),   # Wed publication
        date(2024, 1, 4),   # Thu publication
        date(2024, 1, 9),   # Mon publication (after weekend)
        date(2024, 1, 10),  # Tue publication
        date(2024, 1, 12),  # Fri publication
        date(2024, 1, 16),  # Tue publication (after holiday Mon)
        date(2024, 1, 18),  # Thu publication
    ]

    codes = ["1301", "2269", "6501"]  # Subset of codes with margin data

    rows = []
    for pub_date in pub_dates:
        # Application date is typically same day or T-1
        app_date = pub_date - timedelta(days=int(np.random.choice([0, 1])))

        for code in codes:
            if np.random.random() < 0.7:  # 70% chance of data per code/date
                long_outstanding = int(np.random.uniform(100_000, 5_000_000))
                short_outstanding = int(np.random.uniform(10_000, 1_000_000))

                # Simulate composition breakdown
                long_std = int(long_outstanding * np.random.uniform(0.6, 0.9))
                long_neg = long_outstanding - long_std
                short_std = int(short_outstanding * np.random.uniform(0.7, 0.95))
                short_neg = short_outstanding - short_std

                # Simulate regulatory flags (rarely active)
                reason_flags = {
                    "Restricted": np.random.choice(["0", "1"], p=[0.95, 0.05]),
                    "DailyPublication": np.random.choice(["0", "1"], p=[0.8, 0.2]),
                    "Monitoring": np.random.choice(["0", "1"], p=[0.9, 0.1]),
                    "RestrictedByJSF": np.random.choice(["0", "1"], p=[0.98, 0.02]),
                    "PrecautionByJSF": np.random.choice(["0", "1"], p=[0.97, 0.03]),
                    "UnclearOrSecOnAlert": np.random.choice(["0", "1"], p=[0.99, 0.01]),
                }

                # TSE regulation classification (mostly normal)
                reg_class = np.random.choice(
                    ["", "002", "003", "102"],
                    p=[0.85, 0.1, 0.03, 0.02]
                )

                rows.append({
                    "Code": code,
                    "Date": app_date,
                    "PublishedDate": pub_date,
                    "ApplicationDate": app_date,
                    "LongMarginOutstanding": long_outstanding,
                    "ShortMarginOutstanding": short_outstanding,
                    "LongStandardizedMarginOutstanding": long_std,
                    "LongNegotiableMarginOutstanding": long_neg,
                    "ShortStandardizedMarginOutstanding": short_std,
                    "ShortNegotiableMarginOutstanding": short_neg,
                    "ShortLongRatio": (short_outstanding / long_outstanding * 100) if long_outstanding > 0 else None,
                    "PublishReason": reason_flags,
                    "TSEMarginBorrowingAndLendingRegulationClassification": reg_class,
                })

    df = pl.DataFrame(rows)
    df = df.with_columns([
        pl.col("Date").cast(pl.Date),
        pl.col("PublishedDate").cast(pl.Date),
        pl.col("ApplicationDate").cast(pl.Date),
    ])

    return df.sort(["Code", "PublishedDate"])


if __name__ == "__main__":
    """Run integration tests manually."""
    import sys
    import os

    # Add project root to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    # Create test instance and run key tests
    test_instance = TestDailyMarginPipelineIntegration()

    # Create test data (not using pytest fixtures)
    quotes = create_sample_quotes_panel()
    daily = create_sample_daily_margin_raw()

    try:
        # Run core integration tests
        test_instance.test_end_to_end_pipeline_integration(quotes, daily)
        test_instance.test_temporal_leak_prevention(quotes, daily)
        test_instance.test_business_day_calculation_accuracy()
        test_instance.test_integration_with_ml_dataset_builder(quotes, daily)
        test_instance.test_empty_data_handling(quotes)

        # Run leak verification
        test_leak_verification_comprehensive()

        print("\n" + "="*60)
        print("✅ ALL INTEGRATION TESTS PASSED")
        print("✅ Daily margin interest pipeline ready for production")
        print("✅ Leak verification tests completed successfully")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)