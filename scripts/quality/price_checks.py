#!/usr/bin/env python3
"""
Price Data Quality Checks
価格データの品質チェック実装（pandera + カスタムルール）
"""

import pandas as pd
import polars as pl
import pandera as pa
from pandera import Check, Column, DataFrameSchema
from typing import Dict, Any
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PriceDataValidator:
    """価格データバリデーター（仕様書§4.2準拠）"""

    # Pandera schema definition
    schema = DataFrameSchema(
        columns={
            "ticker": Column(
                str,
                checks=[
                    Check.str_matches(r"^\d{4}$", error="Ticker must be 4 digits"),
                    Check(lambda x: x.notna().all(), error="Ticker cannot be null"),
                ],
                description="Stock ticker code (4 digits)",
            ),
            "date": Column(
                pd.DatetimeTZDtype(tz="Asia/Tokyo"),
                checks=[Check(lambda x: x.notna().all(), error="Date cannot be null")],
                description="Trading date",
            ),
            "open": Column(
                float,
                checks=[
                    Check.gt(0, error="Open price must be positive"),
                    Check(lambda x: x.notna().all(), error="Open cannot be null"),
                ],
                description="Opening price",
            ),
            "high": Column(
                float,
                checks=[
                    Check.gt(0, error="High price must be positive"),
                    Check(lambda x: x.notna().all(), error="High cannot be null"),
                ],
                description="High price",
            ),
            "low": Column(
                float,
                checks=[
                    Check.gt(0, error="Low price must be positive"),
                    Check(lambda x: x.notna().all(), error="Low cannot be null"),
                ],
                description="Low price",
            ),
            "close": Column(
                float,
                checks=[
                    Check.gt(0, error="Close price must be positive"),
                    Check(lambda x: x.notna().all(), error="Close cannot be null"),
                ],
                description="Closing price",
            ),
            "volume": Column(
                int,
                checks=[
                    Check.ge(0, error="Volume must be non-negative"),
                    Check(lambda x: x.notna().all(), error="Volume cannot be null"),
                ],
                description="Trading volume",
            ),
        },
        coerce=True,
        strict=False,
        description="Price data validation schema",
    )

    @staticmethod
    def ohlc_consistency(df: pd.DataFrame) -> bool:
        """
        OHLCの整合性チェック
        High >= max(Open, Low, Close)
        Low <= min(Open, High, Close)

        Args:
            df: 価格データ

        Returns:
            整合性がある場合True

        Raises:
            AssertionError: 整合性違反時
        """
        hi_ok = (df["high"] >= df[["open", "low", "close"]].max(axis=1)).all()
        lo_ok = (df["low"] <= df[["open", "high", "close"]].min(axis=1)).all()

        if not hi_ok:
            violations = df[df["high"] < df[["open", "low", "close"]].max(axis=1)]
            logger.error(f"High price violations found: {len(violations)} rows")
            logger.error(f"Sample violations:\n{violations.head()}")

        if not lo_ok:
            violations = df[df["low"] > df[["open", "high", "close"]].min(axis=1)]
            logger.error(f"Low price violations found: {len(violations)} rows")
            logger.error(f"Sample violations:\n{violations.head()}")

        assert hi_ok and lo_ok, "OHLC consistency failed"
        return True

    @staticmethod
    def check_price_limits(df: pd.DataFrame, limit_pct: float = 30.0) -> bool:
        """
        日次変動幅チェック（ストップ高・ストップ安考慮）

        Args:
            df: 価格データ
            limit_pct: 最大変動率（%）

        Returns:
            問題ない場合True
        """
        # Calculate daily returns
        df_sorted = df.sort_values(["ticker", "date"])
        df_sorted["prev_close"] = df_sorted.groupby("ticker")["close"].shift(1)
        df_sorted["daily_change"] = (
            df_sorted["close"] / df_sorted["prev_close"] - 1
        ) * 100

        # Check for extreme changes
        extreme_changes = df_sorted[abs(df_sorted["daily_change"]) > limit_pct].dropna()

        if not extreme_changes.empty:
            logger.warning(
                f"Found {len(extreme_changes)} extreme price changes (>{limit_pct}%)"
            )
            logger.warning(
                f"Sample:\n{extreme_changes[['ticker', 'date', 'daily_change']].head()}"
            )

        return True

    @staticmethod
    def check_duplicates(df: pd.DataFrame) -> bool:
        """
        重複チェック（ticker + date）

        Args:
            df: 価格データ

        Returns:
            重複がない場合True

        Raises:
            ValueError: 重複が見つかった場合
        """
        duplicates = df.duplicated(subset=["ticker", "date"], keep=False)

        if duplicates.any():
            dup_data = df[duplicates].sort_values(["ticker", "date"])
            raise ValueError(f"Duplicate records found:\n{dup_data}")

        return True

    @staticmethod
    def validate_business_days(df: pd.DataFrame, calendar=None) -> bool:
        """
        営業日チェック

        Args:
            df: 価格データ
            calendar: TSECalendarインスタンス

        Returns:
            全て営業日の場合True
        """
        if calendar is None:
            logger.warning("No calendar provided, skipping business day validation")
            return True

        invalid_dates = []
        for date in df["date"].dt.date.unique():
            if not calendar.is_business_day(date):
                invalid_dates.append(date)

        if invalid_dates:
            logger.error(
                f"Found {len(invalid_dates)} non-business days: {invalid_dates[:5]}"
            )
            return False

        return True

    @classmethod
    def validate_all(cls, df: pd.DataFrame, calendar=None) -> Dict[str, Any]:
        """
        全ての検証を実行

        Args:
            df: 価格データ
            calendar: TSECalendarインスタンス

        Returns:
            検証結果の辞書
        """
        results = {
            "schema_valid": False,
            "ohlc_valid": False,
            "duplicates_valid": False,
            "price_limits_valid": False,
            "business_days_valid": False,
            "errors": [],
        }

        # Schema validation
        try:
            cls.schema.validate(df)
            results["schema_valid"] = True
            logger.info("✓ Schema validation passed")
        except pa.errors.SchemaError as e:
            results["errors"].append(f"Schema error: {e}")
            logger.error(f"✗ Schema validation failed: {e}")

        # OHLC consistency
        try:
            cls.ohlc_consistency(df)
            results["ohlc_valid"] = True
            logger.info("✓ OHLC consistency check passed")
        except AssertionError as e:
            results["errors"].append(f"OHLC error: {e}")
            logger.error(f"✗ OHLC consistency check failed: {e}")

        # Duplicates check
        try:
            cls.check_duplicates(df)
            results["duplicates_valid"] = True
            logger.info("✓ No duplicates found")
        except ValueError as e:
            results["errors"].append(f"Duplicate error: {e}")
            logger.error(f"✗ Duplicate check failed: {e}")

        # Price limits check
        try:
            cls.check_price_limits(df)
            results["price_limits_valid"] = True
            logger.info("✓ Price limits check passed")
        except Exception as e:
            results["errors"].append(f"Price limit error: {e}")
            logger.error(f"✗ Price limits check failed: {e}")

        # Business days check
        if calendar:
            try:
                cls.validate_business_days(df, calendar)
                results["business_days_valid"] = True
                logger.info("✓ Business days check passed")
            except Exception as e:
                results["errors"].append(f"Business day error: {e}")
                logger.error(f"✗ Business days check failed: {e}")

        # Overall result
        results["all_valid"] = all(
            [
                results["schema_valid"],
                results["ohlc_valid"],
                results["duplicates_valid"],
                results["price_limits_valid"],
                results.get("business_days_valid", True),
            ]
        )

        return results


class PolarsValidator:
    """Polars DataFrame用のバリデーター"""

    @staticmethod
    def validate_schema(df: pl.DataFrame) -> bool:
        """スキーマ検証"""
        required_columns = ["Code", "Date", "Open", "High", "Low", "Close", "Volume"]

        # Check columns exist
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        # Check data types
        expected_types = {
            "Code": pl.Utf8,
            "Date": [pl.Date, pl.Datetime],
            "Open": [pl.Float32, pl.Float64],
            "High": [pl.Float32, pl.Float64],
            "Low": [pl.Float32, pl.Float64],
            "Close": [pl.Float32, pl.Float64],
            "Volume": [pl.Int32, pl.Int64, pl.Float32, pl.Float64],
        }

        for col, expected_type in expected_types.items():
            actual_type = df[col].dtype
            if isinstance(expected_type, list):
                if actual_type not in expected_type:
                    logger.warning(
                        f"Column {col} has type {actual_type}, expected one of {expected_type}"
                    )
            elif actual_type != expected_type:
                logger.warning(
                    f"Column {col} has type {actual_type}, expected {expected_type}"
                )

        return True

    @staticmethod
    def check_ohlc_consistency(df: pl.DataFrame) -> bool:
        """OHLC整合性チェック（Polars版）"""
        # High should be >= all other prices
        high_violations = df.filter(
            (pl.col("High") < pl.col("Open"))
            | (pl.col("High") < pl.col("Low"))
            | (pl.col("High") < pl.col("Close"))
        )

        # Low should be <= all other prices
        low_violations = df.filter(
            (pl.col("Low") > pl.col("Open"))
            | (pl.col("Low") > pl.col("High"))
            | (pl.col("Low") > pl.col("Close"))
        )

        if not high_violations.is_empty():
            logger.error(f"High price violations: {len(high_violations)} rows")
            logger.error(f"Sample:\n{high_violations.head()}")

        if not low_violations.is_empty():
            logger.error(f"Low price violations: {len(low_violations)} rows")
            logger.error(f"Sample:\n{low_violations.head()}")

        return high_violations.is_empty() and low_violations.is_empty()

    @staticmethod
    def check_null_values(df: pl.DataFrame) -> Dict[str, int]:
        """NULL値のチェック"""
        null_counts = {}

        for col in df.columns:
            null_count = df[col].null_count()
            if null_count > 0:
                null_counts[col] = null_count
                null_pct = (null_count / len(df)) * 100
                logger.warning(
                    f"Column {col} has {null_count} ({null_pct:.2f}%) null values"
                )

        return null_counts


# Test functions
def test_validators():
    """バリデーターのテスト"""

    # Create sample data with some issues
    data = {
        "ticker": ["7203", "7203", "7203", "9999"],
        "date": pd.to_datetime(
            [
                "2024-01-04 09:00:00+09:00",
                "2024-01-05 09:00:00+09:00",
                "2024-01-09 09:00:00+09:00",
                "2024-01-10 09:00:00+09:00",
            ]
        ),
        "open": [7000.0, 7050.0, 7100.0, 100.0],
        "high": [7100.0, 7150.0, 6900.0, 110.0],  # 6900 is invalid (< low)
        "low": [6950.0, 7000.0, 7050.0, 95.0],
        "close": [7050.0, 7100.0, 7150.0, 105.0],
        "volume": [1000000, 1100000, 1200000, 500000],
    }

    df_pandas = pd.DataFrame(data)

    print("Testing Pandas validator...")
    print("-" * 50)

    # Run validation
    results = PriceDataValidator.validate_all(df_pandas)

    print("\nValidation Results:")
    for key, value in results.items():
        if key != "errors":
            print(f"  {key}: {value}")

    if results["errors"]:
        print("\nErrors found:")
        for error in results["errors"]:
            print(f"  - {error}")

    # Test Polars validator
    print("\n" + "=" * 50)
    print("Testing Polars validator...")
    print("-" * 50)

    df_polars = pl.DataFrame(
        {
            "Code": ["7203", "7203", "7203"],
            "Date": pl.date_range(
                datetime(2024, 1, 4).date(),
                datetime(2024, 1, 6).date(),
                interval="1d",
                eager=True,
            ),
            "Open": [7000.0, 7050.0, 7100.0],
            "High": [7100.0, 7150.0, 7200.0],
            "Low": [6950.0, 7000.0, 7050.0],
            "Close": [7050.0, 7100.0, 7150.0],
            "Volume": [1000000, 1100000, 1200000],
        }
    )

    print("Schema validation:", PolarsValidator.validate_schema(df_polars))
    print("OHLC consistency:", PolarsValidator.check_ohlc_consistency(df_polars))
    print("Null values:", PolarsValidator.check_null_values(df_polars))


if __name__ == "__main__":
    test_validators()
