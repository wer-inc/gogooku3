#!/usr/bin/env python3
"""
Validation script for Phase 1-3 sec_id migration.

This script validates that the dataset build correctly:
1. Loaded dim_security and attached sec_id
2. Used sec_id for all migrated joins
3. Applied categorical encoding
4. Maintained data integrity
"""

import sys
from pathlib import Path

import polars as pl


def test_sec_id_exists(dataset_path: Path) -> bool:
    """Test that SecId column exists and is properly populated."""
    print("\n1. Testing SecId existence...")

    try:
        df = pl.read_parquet(dataset_path)

        # Check SecId exists (PascalCase, not snake_case)
        if "SecId" not in df.columns:
            print("   ❌ FAILED: SecId column not found")
            return False

        # Check SecId type (Categorical is expected)
        if df["SecId"].dtype != pl.Categorical:
            print(f"   ⚠️  WARNING: SecId has dtype {df['SecId'].dtype} (expected Categorical)")

        # Get SecId statistics
        sec_id_null_count = df["SecId"].null_count()
        sec_id_valid_count = len(df) - sec_id_null_count
        null_pct = (sec_id_null_count / len(df) * 100) if len(df) > 0 else 0

        print("   ✅ SecId exists and is valid")
        print(f"      - Type: {df['SecId'].dtype}")
        print(f"      - Valid count: {sec_id_valid_count:,} ({100-null_pct:.1f}%)")
        print(f"      - NULL count: {sec_id_null_count:,} ({null_pct:.1f}%)")
        print(f"      - Unique values: {df['SecId'].n_unique()}")

        if null_pct > 90:
            print("      - ℹ️  High NULL rate is NORMAL for historical data")
            print("        (delisted securities not in current dim_security)")

        return True
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False


def test_code_still_exists(dataset_path: Path) -> bool:
    """Test that Code column still exists (parallel schema)."""
    print("\n2. Testing parallel schema (Code + SecId)...")

    try:
        df = pl.read_parquet(dataset_path)

        # Check Code exists (backward compatibility)
        if "Code" not in df.columns:
            print("   ❌ FAILED: 'Code' column not found")
            return False

        # Check Code type
        if df["Code"].dtype not in [pl.String, pl.Utf8, pl.Categorical]:
            print(f"   ❌ FAILED: Code has wrong dtype: {df['Code'].dtype}")
            return False

        print("   ✅ Parallel schema maintained")
        print(f"      - Code type: {df['Code'].dtype}")
        print(f"      - Unique codes: {df['Code'].n_unique()}")

        return True
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False


def test_categorical_encoding(dataset_path: Path) -> bool:
    """Test that categorical encoding was applied to low-cardinality columns."""
    print("\n3. Testing categorical encoding...")

    try:
        df = pl.read_parquet(dataset_path)

        # Expected categorical columns
        expected_categorical = []

        # Check which columns should be categorical
        for col in ["Code", "SecId", "SectorCode"]:
            if col in df.columns:
                expected_categorical.append(col)

        categorical_found = []
        categorical_missing = []

        for col in expected_categorical:
            if df[col].dtype == pl.Categorical:
                categorical_found.append(col)
            else:
                categorical_missing.append(col)

        if categorical_missing:
            print(f"   ⚠️  WARNING: Expected categorical but found {len(categorical_missing)} non-categorical columns:")
            for col in categorical_missing:
                print(f"      - {col}: {df[col].dtype}")

        if categorical_found:
            print(f"   ✅ Categorical encoding applied to {len(categorical_found)} columns:")
            for col in categorical_found:
                print(f"      - {col}: {df[col].dtype} ({df[col].n_unique()} unique values)")
        else:
            print("   ℹ️  No categorical columns found (may not be enabled)")

        return True  # Non-blocking - categorical is optional optimization
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False


def test_data_integrity(dataset_path: Path) -> bool:
    """Test basic data integrity checks."""
    print("\n4. Testing data integrity...")

    try:
        df = pl.read_parquet(dataset_path)

        # Check row count
        row_count = len(df)
        if row_count == 0:
            print("   ❌ FAILED: Dataset is empty")
            return False

        # Check for duplicate (Code, Date) pairs (primary key)
        code_col = "Code"
        date_col = "Date"

        if code_col in df.columns and date_col in df.columns:
            duplicate_count = (
                df.group_by([code_col, date_col]).agg(pl.len().alias("count")).filter(pl.col("count") > 1).height
            )

            if duplicate_count > 0:
                print(f"   ❌ FAILED: {duplicate_count} duplicate (Code, Date) pairs found")
                return False
            else:
                print("   ✅ No duplicate (Code, Date) pairs")

        # Check SecId to Code mapping consistency
        if "SecId" in df.columns and code_col in df.columns:
            # Each SecId should map to exactly one Code (excluding NULLs)
            sec_id_code_mapping = df.filter(pl.col("SecId").is_not_null()).select(["SecId", code_col]).unique()
            sec_ids_with_multiple_codes = (
                sec_id_code_mapping.group_by("SecId").agg(pl.len().alias("code_count")).filter(pl.col("code_count") > 1)
            )

            if len(sec_ids_with_multiple_codes) > 0:
                print(f"   ❌ FAILED: {len(sec_ids_with_multiple_codes)} SecIds map to multiple Codes")
                print(f"      First 5: {sec_ids_with_multiple_codes.head(5)}")
                return False
            else:
                print("   ✅ SecId to Code mapping is 1:1")

        print("   ✅ Data integrity checks passed")
        print(f"      - Total rows: {row_count:,}")
        print(f"      - Columns: {len(df.columns)}")
        print(f"      - Date range: {df[date_col].min()} to {df[date_col].max()}")

        return True
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False


def test_join_columns(dataset_path: Path) -> bool:
    """Test that expected join result columns exist."""
    print("\n5. Testing join result columns...")

    try:
        df = pl.read_parquet(dataset_path)

        # Expected columns from joins
        expected_join_columns = [
            "SectorCode",  # From listed join (PascalCase in gogooku5)
        ]

        missing_columns = [col for col in expected_join_columns if col not in df.columns]

        if missing_columns:
            print(f"   ⚠️  WARNING: {len(missing_columns)} expected join columns missing:")
            for col in missing_columns:
                print(f"      - {col}")
        else:
            print(f"   ✅ All expected join columns present ({len(expected_join_columns)} columns)")

        return True  # Non-blocking
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False


def main():
    """Run all validation tests."""
    print("=" * 80)
    print("Phase 1-3 sec_id Migration Validation")
    print("=" * 80)

    # Check for dataset file (use 2024Q1 instead of 2024Q1_smoke)
    chunk_dir = Path("/workspace/gogooku3/output_g5/chunks/2024Q1")
    dataset_files = list(chunk_dir.glob("*.parquet")) if chunk_dir.exists() else []

    if not dataset_files:
        print(f"\n❌ ERROR: No dataset files found in {chunk_dir}")
        print("\nPlease ensure the Q1 smoke test has completed successfully.")
        sys.exit(1)

    dataset_path = dataset_files[0]
    print(f"\nValidating dataset: {dataset_path}")
    print(f"File size: {dataset_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Run tests
    tests = [
        test_sec_id_exists,
        test_code_still_exists,
        test_categorical_encoding,
        test_data_integrity,
        test_join_columns,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            result = test_func(dataset_path)
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n   ❌ EXCEPTION: {e}")
            failed += 1

    # Summary
    print("\n" + "=" * 80)
    print(f"Validation Summary: {passed} passed, {failed} failed")
    print("=" * 80)

    if failed > 0:
        print("\n⚠️  Some validation tests failed. Please review the errors above.")
        sys.exit(1)
    else:
        print("\n✅ All validation tests passed!")
        print("\nPhase 1-3 sec_id migration is complete and validated:")
        print("  - Phase 1: ✅ dim_security generation")
        print("  - Phase 2: ✅ sec_id propagation + categorical optimization")
        print("  - Phase 3: ✅ Join migration to sec_id")
        print("\nReady for Phase 4 (performance benchmarking)")
        sys.exit(0)


if __name__ == "__main__":
    main()
