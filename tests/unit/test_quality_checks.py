"""
Unit tests for Quality Checks
"""

import pandas as pd
import polars as pl
import pytest

from quality.price_checks import PolarsValidator, PriceDataValidator


class TestPolarsValidator:
    def test_validate_schema(self, sample_price_data):
        """Test schema validation"""
        validator = PolarsValidator()

        # Should pass with correct schema
        assert validator.validate_schema(sample_price_data) is True

    def test_check_ohlc_consistency(self):
        """Test OHLC consistency check"""
        validator = PolarsValidator()

        # Valid OHLC data
        valid_data = pl.DataFrame(
            {
                "Code": ["7203"],
                "Date": [pl.datetime(2024, 1, 4)],
                "Open": [100.0],
                "High": [110.0],  # High is highest
                "Low": [90.0],  # Low is lowest
                "Close": [105.0],
                "Volume": [1000],
            }
        )

        assert validator.check_ohlc_consistency(valid_data) is True

        # Invalid OHLC data (High < Close)
        invalid_data = pl.DataFrame(
            {
                "Code": ["7203"],
                "Date": [pl.datetime(2024, 1, 4)],
                "Open": [100.0],
                "High": [103.0],  # High is less than Close - invalid!
                "Low": [90.0],
                "Close": [105.0],
                "Volume": [1000],
            }
        )

        assert validator.check_ohlc_consistency(invalid_data) is False

    def test_check_null_values(self, sample_price_data):
        """Test null value checking"""
        validator = PolarsValidator()

        # No nulls in sample data
        null_counts = validator.check_null_values(sample_price_data)
        assert len(null_counts) == 0

        # Add null values
        data_with_nulls = sample_price_data.with_columns(
            [
                pl.when(pl.col("Code") == "7203")
                .then(None)
                .otherwise(pl.col("Volume"))
                .alias("Volume")
            ]
        )

        null_counts = validator.check_null_values(data_with_nulls)
        assert "Volume" in null_counts
        assert null_counts["Volume"] > 0


class TestPriceDataValidator:
    def test_check_duplicates(self):
        """Test duplicate checking"""
        # Create data with duplicates
        df = pd.DataFrame(
            {
                "ticker": ["7203", "7203", "9984"],
                "date": pd.to_datetime(["2024-01-04", "2024-01-04", "2024-01-04"]),
                "open": [100.0, 100.0, 200.0],
                "high": [110.0, 110.0, 220.0],
                "low": [90.0, 90.0, 180.0],
                "close": [105.0, 105.0, 210.0],
                "volume": [1000, 1000, 2000],
            }
        )

        # Should raise ValueError for duplicates
        with pytest.raises(ValueError, match="Duplicate records found"):
            PriceDataValidator.check_duplicates(df)

        # No duplicates
        df_unique = df.drop_duplicates(subset=["ticker", "date"])
        assert PriceDataValidator.check_duplicates(df_unique) is True

    def test_check_price_limits(self):
        """Test price limit checking"""
        # Normal price changes
        df_normal = pd.DataFrame(
            {
                "ticker": ["7203", "7203"],
                "date": pd.to_datetime(["2024-01-04", "2024-01-05"]),
                "open": [100.0, 102.0],
                "high": [110.0, 112.0],
                "low": [90.0, 92.0],
                "close": [105.0, 107.0],  # ~2% change
                "volume": [1000, 1100],
            }
        )

        assert PriceDataValidator.check_price_limits(df_normal, limit_pct=30.0) is True

        # Extreme price change
        df_extreme = pd.DataFrame(
            {
                "ticker": ["7203", "7203"],
                "date": pd.to_datetime(["2024-01-04", "2024-01-05"]),
                "open": [100.0, 200.0],
                "high": [110.0, 220.0],
                "low": [90.0, 180.0],
                "close": [105.0, 210.0],  # 100% change!
                "volume": [1000, 1100],
            }
        )

        # Should still return True but log warning
        assert PriceDataValidator.check_price_limits(df_extreme, limit_pct=30.0) is True
