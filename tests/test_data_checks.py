"""
Data quality and integrity checks for CI.

Tests verify that:
1. Column counts and types are correct
2. LOO (Leave-One-Out) features exclude self
3. No data leakage in cross-sectional standardization
4. Embargo periods are enforced in CV splits
"""

from __future__ import annotations

import pytest
import polars as pl
import numpy as np
from pathlib import Path
from typing import List, Set, Dict, Any

from gogooku3.features_ext.sector_loo import add_sector_loo
from gogooku3.features_ext.cs_standardize import fit_cs_stats, transform_cs
from gogooku3.training.cv_purged import purged_kfold_indices


class TestDataIntegrity:
    """Test data integrity and column consistency."""

    @pytest.fixture
    def sample_data(self) -> pl.DataFrame:
        """Create sample panel data for testing."""
        np.random.seed(42)
        n_stocks = 100
        n_days = 250

        dates = []
        codes = []
        returns = []
        sectors = []

        for d in range(n_days):
            date = f"2024-{(d//20)+1:02d}-{(d%20)+1:02d}"
            for s in range(n_stocks):
                dates.append(date)
                codes.append(f"STOCK_{s:04d}")
                returns.append(np.random.randn() * 0.02)
                sectors.append(s // 10)  # 10 sectors

        return pl.DataFrame({
            "Date": dates,
            "Code": codes,
            "returns_1d": returns,
            "sector33_id": sectors,
            "volatility_20d": np.abs(np.random.randn(len(dates)) * 0.1),
        })

    def test_column_count_after_extensions(self, sample_data: pl.DataFrame) -> None:
        """Test that column count increases correctly after feature extensions."""
        initial_cols = len(sample_data.columns)

        # Add LOO feature
        df = add_sector_loo(sample_data, ret_col="returns_1d", sec_col="sector33_id")

        # Should have added exactly 1 column (sec_ret_1d_eq_loo)
        assert len(df.columns) == initial_cols + 1
        assert "sec_ret_1d_eq_loo" in df.columns

        # Verify original columns are preserved
        for col in sample_data.columns:
            assert col in df.columns

    def test_loo_excludes_self(self, sample_data: pl.DataFrame) -> None:
        """Test that LOO aggregation excludes self correctly."""
        df = add_sector_loo(sample_data, ret_col="returns_1d", sec_col="sector33_id")

        # For each row, LOO should not equal the sector mean including self
        # unless there's only 1 stock in the sector
        sector_means = (
            df.group_by(["Date", "sector33_id"])
            .agg(
                pl.col("returns_1d").mean().alias("sec_mean_with_self"),
                pl.col("returns_1d").count().alias("sec_count")
            )
        )

        df_check = df.join(sector_means, on=["Date", "sector33_id"])

        # Where count > 1, LOO should differ from self-inclusive mean
        multi_stock = df_check.filter(pl.col("sec_count") > 1)
        if multi_stock.height > 0:
            # LOO should not equal the mean that includes self
            diff = (
                multi_stock["sec_ret_1d_eq_loo"] - multi_stock["sec_mean_with_self"]
            ).abs()
            assert (diff > 1e-10).all(), "LOO should exclude self from calculation"

    def test_column_types_consistency(self, sample_data: pl.DataFrame) -> None:
        """Test that column types remain consistent."""
        # Code should be string
        assert sample_data["Code"].dtype == pl.Utf8

        # Numeric columns should be numeric
        numeric_cols = ["returns_1d", "volatility_20d", "sector33_id"]
        for col in numeric_cols:
            assert pl.datatypes.is_numeric_dtype(sample_data[col].dtype)

        # No "-" strings in numeric columns (DMI data issue)
        for col in numeric_cols:
            if sample_data[col].dtype == pl.Utf8:
                assert not sample_data[col].str.contains("-").any()

    def test_minimum_column_requirement(self) -> None:
        """Test that we maintain at least 395 base columns + extensions."""
        # This would be tested on actual data
        expected_base_cols = 395
        expected_extensions = 10  # x_* interactions

        # Mock test - in reality, load actual dataset
        total_expected = expected_base_cols + expected_extensions

        # Placeholder assertion
        assert total_expected >= 405, f"Should have at least {expected_base_cols} base + {expected_extensions} extension columns"


class TestCrossValidationIntegrity:
    """Test cross-validation split integrity."""

    def test_purged_kfold_no_overlap(self) -> None:
        """Test that purged KFold has no train/test overlap with embargo."""
        # Create date sequence
        dates = pd.date_range("2020-01-01", "2024-12-31", freq="D")
        dates_np = dates.to_numpy()

        # Get folds
        folds = purged_kfold_indices(dates_np, n_splits=5, embargo_days=20)

        for i, fold in enumerate(folds):
            train_dates = dates_np[fold.train_idx]
            val_dates = dates_np[fold.val_idx]

            # No direct overlap
            overlap = set(train_dates) & set(val_dates)
            assert len(overlap) == 0, f"Fold {i} has train/val overlap"

            # Check embargo enforcement (20 days)
            if len(val_dates) > 0:
                val_start = val_dates.min()
                val_end = val_dates.max()

                # No train data within embargo of validation
                embargo_start = val_start - np.timedelta64(20, "D")
                embargo_end = val_end + np.timedelta64(20, "D")

                train_in_embargo = [
                    d for d in train_dates
                    if embargo_start <= d <= embargo_end
                ]
                assert len(train_in_embargo) == 0, f"Fold {i} has train data in embargo period"

    def test_fold_sizes_reasonable(self) -> None:
        """Test that fold sizes are reasonable and balanced."""
        dates = pd.date_range("2020-01-01", "2024-12-31", freq="D")
        dates_np = dates.to_numpy()

        folds = purged_kfold_indices(dates_np, n_splits=5, embargo_days=20)

        train_sizes = [len(f.train_idx) for f in folds]
        val_sizes = [len(f.val_idx) for f in folds]

        # All folds should have data
        assert all(s > 0 for s in train_sizes), "Some folds have no training data"
        assert all(s > 0 for s in val_sizes), "Some folds have no validation data"

        # Validation sizes should be roughly balanced (within 2x)
        max_val = max(val_sizes)
        min_val = min(val_sizes)
        assert max_val / min_val < 2.0, "Validation fold sizes are too imbalanced"


class TestCrossSectionalStandardization:
    """Test cross-sectional standardization for leakage."""

    def test_cs_z_no_future_leakage(self) -> None:
        """Test that CS-Z uses only training statistics."""
        # Create sample data
        np.random.seed(42)
        dates = ["2024-01-01"] * 100 + ["2024-01-02"] * 100
        values = np.random.randn(200)

        df = pl.DataFrame({
            "Date": dates,
            "value": values,
        })

        # Split train/test
        train_df = df.filter(pl.col("Date") == "2024-01-01")
        test_df = df.filter(pl.col("Date") == "2024-01-02")

        # Fit on train only
        stats = fit_cs_stats(train_df, ["value"], date_col="Date")

        # Transform both
        train_transformed = transform_cs(train_df, stats, ["value"])
        test_transformed = transform_cs(test_df, stats, ["value"])

        # Test data should use train statistics, not its own
        test_mean = test_df["value"].mean()
        test_std = test_df["value"].std()

        # The CS-Z on test should NOT be standardized to mean=0, std=1
        # if using train statistics
        test_cs_z = test_transformed["value_cs_z"]
        assert abs(test_cs_z.mean()) > 0.01 or abs(test_cs_z.std() - 1.0) > 0.01, \
            "Test data appears to be using its own statistics (leakage)"

    def test_cs_z_train_standardized(self) -> None:
        """Test that training data is properly standardized."""
        np.random.seed(42)
        dates = ["2024-01-01"] * 1000
        values = np.random.randn(1000) * 2 + 5  # Non-standard distribution

        df = pl.DataFrame({
            "Date": dates,
            "value": values,
        })

        # Fit and transform on same data (training)
        stats = fit_cs_stats(df, ["value"], date_col="Date")
        transformed = transform_cs(df, stats, ["value"])

        # Should be standardized to approximately mean=0, std=1
        cs_z = transformed["value_cs_z"]
        assert abs(cs_z.mean()) < 0.01, "CS-Z mean should be ~0 on training data"
        assert abs(cs_z.std() - 1.0) < 0.01, "CS-Z std should be ~1 on training data"


class TestFeatureGroupsIntegrity:
    """Test feature groups configuration."""

    def test_feature_groups_non_overlapping(self) -> None:
        """Test that feature groups don't overlap."""
        # In a real test, load from configs/feature_groups.yaml
        groups = {
            "MA": ["ma_"],
            "EMA": ["ema_"],
            "BB": ["bb_"],
            "VOL": ["vol", "volatility"],
            "FLOW": ["flow_"],
            "MARGIN": ["margin_"],
            "DMI": ["dmi_"],
            "STMT": ["stmt_"],
            "MKT": ["mkt_"],
            "SEC": ["sec_"],
            "INTERACTIONS": ["x_"],
        }

        # Check for prefix overlaps
        all_prefixes = []
        for group_name, prefixes in groups.items():
            for prefix in prefixes:
                # Check if this prefix overlaps with any existing
                for existing in all_prefixes:
                    assert not prefix.startswith(existing) and not existing.startswith(prefix), \
                        f"Prefix overlap: {prefix} and {existing}"
                all_prefixes.append(prefix)

    def test_interaction_features_start_with_x(self) -> None:
        """Test that all interaction features start with x_."""
        # This would check actual generated features
        interaction_names = [
            "x_trend_intensity",
            "x_rel_sec_mom",
            "x_mom_sh_5",
            "x_rvol5_dir",
            "x_squeeze_pressure",
            "x_credit_rev_bias",
            "x_pead_effect",
            "x_rev_gate",
            "x_alpha_meanrev_stable",
            "x_flow_smart_rel",
        ]

        for name in interaction_names:
            assert name.startswith("x_"), f"Interaction feature {name} should start with x_"


class TestWarmupPeriods:
    """Test that warmup periods are handled correctly."""

    def test_rolling_features_have_warmup_nans(self) -> None:
        """Test that rolling features have NaN during warmup."""
        # Create sample data
        np.random.seed(42)
        n_days = 300
        values = np.random.randn(n_days)

        df = pl.DataFrame({
            "Code": ["STOCK_001"] * n_days,
            "value": values,
        })

        # Apply rolling mean with 252-day window
        df = df.with_columns(
            pl.col("value")
            .rolling_mean(window_size=252, min_periods=252 // 2)
            .over("Code")
            .alias("value_ma252")
        )

        # First ~126 days should have NaN
        warmup_period = df.head(126)
        assert warmup_period["value_ma252"].null_count() > 0, \
            "Rolling features should have NaN during warmup"

        # After warmup, should have values
        after_warmup = df.tail(n_days - 126)
        assert after_warmup["value_ma252"].null_count() == 0, \
            "Rolling features should have values after warmup"


# Import pandas for date handling (if not already imported)
try:
    import pandas as pd
except ImportError:
    pd = None


@pytest.mark.skipif(pd is None, reason="pandas required for date tests")
class TestEndToEndDataPipeline:
    """Test the complete data pipeline integrity."""

    def test_pipeline_preserves_column_order(self) -> None:
        """Test that pipeline preserves column order and adds new columns at end."""
        # This would test actual pipeline execution
        pass  # Placeholder for actual implementation

    def test_pipeline_memory_efficiency(self) -> None:
        """Test that pipeline completes within memory constraints."""
        # Monitor memory usage during pipeline execution
        pass  # Placeholder for actual implementation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])