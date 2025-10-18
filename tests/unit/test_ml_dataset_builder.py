"""
Unit tests for ML Dataset Builder
"""

import pytest

pytest.skip(
    "core.ml_dataset_builder is not available in current repository structure - temporarily skipped pending migration",
    allow_module_level=True,
)

import polars as pl
from core.ml_dataset_builder import MLDatasetBuilder


class TestMLDatasetBuilder:
    def test_init(self):
        """Test MLDatasetBuilder initialization"""
        builder = MLDatasetBuilder()
        assert builder is not None
        assert builder.feature_cols is not None

    def test_calculate_returns(self, sample_price_data):
        """Test return calculations"""
        # Calculate returns
        result = sample_price_data.with_columns(
            [
                pl.col("Close").pct_change().over("Code").alias("returns_1d"),
            ]
        )

        # Check returns exist
        assert "returns_1d" in result.columns

        # Check first value is null for each stock
        for code in ["7203", "9984"]:
            stock_data = result.filter(pl.col("Code") == code)
            assert stock_data["returns_1d"][0] is None
            assert stock_data["returns_1d"][1] is not None

    def test_build_features(self, sample_price_data, sample_topix_data):
        """Test feature building"""
        builder = MLDatasetBuilder()

        # Build features
        result = builder.build_features(sample_price_data, sample_topix_data)

        # Check result is not empty
        assert len(result) > 0

        # Check essential columns exist
        assert "Code" in result.columns
        assert "Date" in result.columns
        assert "returns_1d" in result.columns

        # Check TOPIX features exist
        assert (
            "alpha_1d_topix" in result.columns
            or "relative_returns_1d_topix" in result.columns
        )

    def test_no_data_leakage(self, sample_price_data):
        """Test that there's no data leakage between stocks"""
        # Modify one stock's data
        modified_data = sample_price_data.with_columns(
            [
                pl.when(pl.col("Code") == "7203")
                .then(pl.lit(9999999.0))
                .otherwise(pl.col("Close"))
                .alias("Close")
            ]
        )

        # Calculate features
        result = modified_data.with_columns(
            [
                pl.col("Close").pct_change().over("Code").alias("returns_1d"),
            ]
        )

        # Check that 9984's returns are not affected
        stock_9984 = result.filter(pl.col("Code") == "9984")
        returns_9984 = stock_9984["returns_1d"].drop_nulls().to_list()

        # Returns should be reasonable (not affected by 7203's extreme value)
        assert all(abs(r) < 1.0 for r in returns_9984)  # Less than 100% daily return
