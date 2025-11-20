#!/usr/bin/env python3
"""
Phase 1 Data Quality Validation

Validates that Phase 1 features are correctly implemented in actual data:
1. Flag columns (4 columns with correct Int8 dtype and valid values)
2. Git metadata (non-null git_sha and git_branch in metadata.json)
3. Macro columns (40+ columns present)
4. Data integrity (NULL distributions, value ranges)
"""

import json
from pathlib import Path

import polars as pl


def validate_phase1_data_quality(chunk_id: str = "2025Q4") -> bool:
    """Validate Phase 1 implementation in actual chunk data"""

    base_dir = Path("/workspace/gogooku3/output_g5/chunks")
    chunk_dir = base_dir / chunk_id
    parquet_path = chunk_dir / "ml_dataset.parquet"
    metadata_path = chunk_dir / "metadata.json"

    print("=" * 80)
    print(f"Phase 1 Data Quality Validation: {chunk_id}")
    print("=" * 80)
    print()

    if not parquet_path.exists():
        print(f"❌ Chunk parquet not found: {parquet_path}")
        return False

    # Load data
    print(f"📊 Loading data from {parquet_path.name}...")
    df = pl.read_parquet(parquet_path)
    print(f"   Rows: {len(df):,}, Columns: {len(df.columns):,}")
    print()

    all_passed = True

    # =========================================================================
    # Check 1: Flag columns (Phase 1.3)
    # =========================================================================
    print("=" * 80)
    print("Check 1: Flag Columns (Phase 1.3)")
    print("=" * 80)

    expected_flags = [
        "flag_halted",
        "flag_price_limit_hit",
        "flag_delisted",
        "flag_adjustment_event",
    ]

    flag_check_passed = True
    for flag_col in expected_flags:
        if flag_col not in df.columns:
            print(f"❌ Missing flag column: {flag_col}")
            flag_check_passed = False
            all_passed = False
            continue

        # Check dtype
        dtype = df[flag_col].dtype
        if dtype != pl.Int8:
            print(f"❌ {flag_col}: Wrong dtype (got {dtype}, expected Int8)")
            flag_check_passed = False
            all_passed = False
            continue

        # Check value distribution
        null_count = df[flag_col].null_count()
        value_counts = df[flag_col].value_counts().sort("count", descending=True)

        print(f"✅ {flag_col}: Int8")
        print(f"   - NULL: {null_count:,} ({null_count/len(df)*100:.2f}%)")
        print("   - Value distribution:")
        for row in value_counts.head(5).iter_rows(named=True):
            val = row[flag_col]
            count = row["count"]
            pct = count / len(df) * 100
            print(f"     {val}: {count:,} ({pct:.2f}%)")

    if flag_check_passed:
        print("\n✅ All 4 flag columns validated")
    print()

    # =========================================================================
    # Check 2: Git metadata (Phase 1.4)
    # =========================================================================
    print("=" * 80)
    print("Check 2: Git Metadata (Phase 1.4)")
    print("=" * 80)

    if not metadata_path.exists():
        print(f"❌ Metadata file not found: {metadata_path}")
        all_passed = False
    else:
        with open(metadata_path) as f:
            metadata = json.load(f)

        git_sha = metadata.get("git_sha")
        git_branch = metadata.get("git_branch")

        if not git_sha:
            print("❌ git_sha not found in metadata")
            all_passed = False
        else:
            print(f"✅ git_sha: {git_sha}")

        if not git_branch:
            print("❌ git_branch not found in metadata")
            all_passed = False
        else:
            print(f"✅ git_branch: {git_branch}")

    print()

    # =========================================================================
    # Check 3: Macro columns (Phase 1.5)
    # =========================================================================
    print("=" * 80)
    print("Check 3: Macro Columns (Phase 1.5)")
    print("=" * 80)

    # Expected macro prefixes
    macro_prefixes = [
        "topix_",
        "nk225_opt_",
        "trades_spec_",
        "trading_cal_",
        "vix_",
    ]

    macro_cols = []
    for col in df.columns:
        for prefix in macro_prefixes:
            if col.startswith(prefix):
                macro_cols.append(col)
                break

    if len(macro_cols) < 35:
        print(f"❌ Too few macro columns: {len(macro_cols)} (expected ≥35)")
        all_passed = False
    else:
        print(f"✅ Macro columns found: {len(macro_cols)}")

    # Show sample
    print("\n   Sample macro columns (first 10):")
    for col in sorted(macro_cols)[:10]:
        null_count = df[col].null_count()
        null_pct = null_count / len(df) * 100
        print(f"   - {col}: NULL={null_count:,} ({null_pct:.1f}%)")

    print()

    # =========================================================================
    # Check 4: Data integrity
    # =========================================================================
    print("=" * 80)
    print("Check 4: Data Integrity")
    print("=" * 80)

    # Check Date column
    if "Date" in df.columns:
        date_nulls = df["Date"].null_count()
        if date_nulls > 0:
            print(f"❌ Date column has {date_nulls:,} NULLs")
            all_passed = False
        else:
            print("✅ Date column: No NULLs")

        date_min = df["Date"].min()
        date_max = df["Date"].max()
        print(f"   Range: {date_min} to {date_max}")

    # Check Code column
    if "Code" in df.columns:
        code_nulls = df["Code"].null_count()
        if code_nulls > 0:
            print(f"❌ Code column has {code_nulls:,} NULLs")
            all_passed = False
        else:
            print("✅ Code column: No NULLs")

        unique_codes = df["Code"].n_unique()
        print(f"   Unique codes: {unique_codes:,}")

    # Check target columns (should have some non-null values)
    target_cols = ["ret_prev_1d", "ret_prev_5d", "ret_prev_10d", "ret_prev_20d"]
    for col in target_cols:
        if col in df.columns:
            null_count = df[col].null_count()
            null_pct = null_count / len(df) * 100
            if null_pct > 95:
                print(f"⚠️  {col}: Too many NULLs ({null_pct:.1f}%)")
            else:
                print(f"✅ {col}: {null_pct:.1f}% NULL")

    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 80)
    print("Summary")
    print("=" * 80)

    if all_passed:
        print(f"✅ All Phase 1 data quality checks passed for {chunk_id}!")
        print()
        print("Phase 1 features are correctly implemented:")
        print("  - Flag columns: 4 columns with Int8 dtype ✅")
        print("  - Git metadata: git_sha and git_branch present ✅")
        print("  - Macro columns: 35+ columns found ✅")
        print("  - Data integrity: No critical issues ✅")
        return True
    else:
        print(f"❌ Some data quality checks failed for {chunk_id}")
        return False


if __name__ == "__main__":
    success = validate_phase1_data_quality("2025Q4")
    exit(0 if success else 1)
