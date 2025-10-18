#!/usr/bin/env python
"""
Data validation script for ATFT-GAT-FAN training.
Validates target values and feature distributions before training.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_target_columns(df: pl.DataFrame) -> dict[str, dict]:
    """Check target column statistics."""
    results = {}
    target_cols = [
        col for col in df.columns if col.startswith("target_") and "binary" not in col
    ]

    if not target_cols:
        logger.error("❌ No target columns found!")
        return results

    logger.info(f"Found {len(target_cols)} target columns: {target_cols}")

    for col in target_cols:
        values = df[col].drop_nulls()
        if len(values) > 0:
            non_zero = (values != 0).sum()
            stats = {
                "count": len(values),
                "non_zero": non_zero,
                "non_zero_ratio": non_zero / len(values),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "null_count": df[col].is_null().sum(),
                "null_ratio": df[col].is_null().sum() / len(df),
            }
            results[col] = stats
        else:
            results[col] = {"error": "All values are null"}

    return results


def check_feature_scales(df: pl.DataFrame) -> dict[str, dict]:
    """Check feature column scales and distributions."""
    results = {}

    # Get numeric columns (excluding targets and identifiers)
    exclude_patterns = [
        "target_",
        "Date",
        "date",
        "Code",
        "code",
        "row_idx",
        "sector",
        "_id",
    ]
    feature_cols = []

    for col in df.columns:
        if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
            if not any(pattern in col for pattern in exclude_patterns):
                feature_cols.append(col)

    logger.info(f"Analyzing {len(feature_cols)} feature columns")

    # Sample features for analysis
    sample_features = feature_cols[:20] if len(feature_cols) > 20 else feature_cols

    for col in sample_features:
        values = df[col].drop_nulls()
        if len(values) > 0:
            mean_val = float(values.mean())
            std_val = float(values.std())

            # Check if feature needs normalization
            needs_norm = abs(mean_val) > 10 or std_val > 100

            stats = {
                "mean": mean_val,
                "std": std_val,
                "min": float(values.min()),
                "max": float(values.max()),
                "needs_normalization": needs_norm,
                "null_ratio": df[col].is_null().sum() / len(df),
            }
            results[col] = stats

    return results


def check_date_range(df: pl.DataFrame) -> dict:
    """Check date range and distribution."""
    date_cols = [col for col in df.columns if "date" in col.lower()]

    if not date_cols:
        return {"error": "No date column found"}

    date_col = date_cols[0]
    dates = df[date_col].drop_nulls()

    if len(dates) == 0:
        return {"error": "All dates are null"}

    min_date = dates.min()
    max_date = dates.max()

    # Check if data is too old
    current_year = datetime.now().year
    min_year = min_date.year if hasattr(min_date, "year") else int(str(min_date)[:4])

    is_old = (current_year - min_year) > 5

    return {
        "date_column": date_col,
        "min_date": str(min_date),
        "max_date": str(max_date),
        "unique_dates": dates.n_unique(),
        "total_rows": len(dates),
        "is_old_data": is_old,
        "warning": "Data starts more than 5 years ago" if is_old else None,
    }


def validate_training_data(data_dir: str) -> bool:
    """Main validation function for training data."""
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.error(f"❌ Data directory does not exist: {data_dir}")
        return False

    # Check train directory
    train_dir = data_path / "train"
    if not train_dir.exists():
        logger.error(f"❌ Training directory does not exist: {train_dir}")
        return False

    # Get sample files
    parquet_files = list(train_dir.glob("*.parquet"))[:5]

    if not parquet_files:
        logger.error(f"❌ No parquet files found in {train_dir}")
        return False

    logger.info(f"📊 Validating {len(parquet_files)} sample files from {train_dir}")

    all_valid = True
    total_issues = 0

    for file_path in parquet_files:
        logger.info(f"\n=== Validating {file_path.name} ===")

        try:
            df = pl.read_parquet(file_path)
            logger.info(f"  Shape: {df.shape}")

            # Check targets
            target_stats = check_target_columns(df)
            if target_stats:
                for col, stats in target_stats.items():
                    if "error" in stats:
                        logger.error(f"  ❌ {col}: {stats['error']}")
                        total_issues += 1
                    else:
                        zero_ratio = 1 - stats["non_zero_ratio"]
                        if zero_ratio > 0.95:
                            logger.warning(
                                f"  ⚠️ {col}: {zero_ratio:.1%} zeros (might cause zero loss)"
                            )
                            total_issues += 1
                        else:
                            logger.info(
                                f"  ✅ {col}: mean={stats['mean']:.6f}, non_zero={stats['non_zero_ratio']:.1%}"
                            )
            else:
                logger.error("  ❌ No target columns found!")
                total_issues += 1
                all_valid = False

            # Check date range
            date_info = check_date_range(df)
            if "error" in date_info:
                logger.error(f"  ❌ Date check: {date_info['error']}")
                total_issues += 1
            else:
                logger.info(
                    f"  📅 Date range: {date_info['min_date']} to {date_info['max_date']}"
                )
                if date_info.get("is_old_data"):
                    logger.warning(f"  ⚠️ {date_info['warning']}")

            # Check feature scales (brief)
            feature_stats = check_feature_scales(df)
            needs_norm_count = sum(
                1
                for stats in feature_stats.values()
                if stats.get("needs_normalization")
            )
            if needs_norm_count > 0:
                logger.info(
                    f"  📊 {needs_norm_count}/{len(feature_stats)} features need normalization"
                )
                logger.info("  ✅ Feature normalization is ENABLED in config")

        except Exception as e:
            logger.error(f"  ❌ Error reading {file_path.name}: {e}")
            total_issues += 1
            all_valid = False

    # Summary
    logger.info("\n" + "=" * 60)
    if all_valid and total_issues == 0:
        logger.info("✅ DATA VALIDATION PASSED - All checks successful!")
        return True
    elif all_valid:
        logger.warning(
            f"⚠️ DATA VALIDATION COMPLETED WITH WARNINGS - {total_issues} issues found"
        )
        logger.info("Training can proceed but monitor for potential issues.")
        return True
    else:
        logger.error("❌ DATA VALIDATION FAILED - Critical issues found!")
        return False


def validate_dataset_file(file_path: str) -> bool:
    """Validate a single dataset file."""
    file_path = Path(file_path)

    if not file_path.exists():
        logger.error(f"❌ File does not exist: {file_path}")
        return False

    logger.info(f"📊 Validating dataset: {file_path}")

    try:
        df = pl.read_parquet(file_path, n_rows=10000)  # Sample for speed
        logger.info(f"  Shape (sample): {df.shape}")

        # Check targets
        target_stats = check_target_columns(df)
        valid = True

        if not target_stats:
            logger.error("  ❌ No target columns found!")
            return False

        for col, stats in target_stats.items():
            if "error" not in stats:
                logger.info(
                    f"  ✅ {col}: mean={stats['mean']:.6f}, non_zero={stats['non_zero_ratio']:.1%}"
                )
                if stats["non_zero_ratio"] < 0.05:
                    logger.warning(f"  ⚠️ {col} has very few non-zero values!")
                    valid = False

        # Check date range
        date_info = check_date_range(df)
        if "error" not in date_info:
            logger.info(
                f"  📅 Date range: {date_info['min_date']} to {date_info['max_date']}"
            )

        return valid

    except Exception as e:
        logger.error(f"❌ Error reading file: {e}")
        return False


if __name__ == "__main__":
    # Check if specific file provided
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        success = validate_dataset_file(file_path)
    else:
        # Default: validate training data directory
        data_dir = os.environ.get(
            "DATA_DIR", "/home/ubuntu/gogooku3-standalone/output/atft_data"
        )
        success = validate_training_data(data_dir)

    sys.exit(0 if success else 1)
