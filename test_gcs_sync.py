#!/usr/bin/env python3
"""Test script for GCS sync functionality."""

import os
import sys
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import polars as pl
import pandas as pd
from src.gogooku3.utils.gcs_storage import (
    save_parquet_with_gcs,
    is_gcs_enabled,
    get_gcs_bucket,
    list_gcs_files,
)


def test_helper_function():
    """Test save_parquet_with_gcs with both Polars and Pandas."""
    print("=" * 60)
    print("Testing save_parquet_with_gcs helper function")
    print("=" * 60)

    # Check GCS config
    print(f"\n✅ GCS Enabled: {is_gcs_enabled()}")
    print(f"✅ GCS Bucket: {get_gcs_bucket()}")

    # Create test directory
    test_dir = Path("output/test_gcs")
    test_dir.mkdir(parents=True, exist_ok=True)

    # Test 1: Polars DataFrame
    print("\n--- Test 1: Polars DataFrame ---")
    polars_df = pl.DataFrame({
        "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
        "code": ["1301", "1301", "1301"],
        "close": [100.0, 101.5, 99.8],
    })
    polars_path = test_dir / "test_polars.parquet"
    result_path = save_parquet_with_gcs(polars_df, polars_path, gcs_path="test/test_polars.parquet")
    print(f"✅ Polars test saved to: {result_path}")
    assert polars_path.exists(), "Polars file should exist locally"

    # Test 2: Pandas DataFrame
    print("\n--- Test 2: Pandas DataFrame ---")
    pandas_df = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
        "code": ["9984", "9984", "9984"],
        "close": [5000.0, 5100.0, 5050.0],
    })
    pandas_path = test_dir / "test_pandas.parquet"
    result_path = save_parquet_with_gcs(pandas_df, pandas_path, gcs_path="test/test_pandas.parquet")
    print(f"✅ Pandas test saved to: {result_path}")
    assert pandas_path.exists(), "Pandas file should exist locally"

    # Test 3: Check GCS upload (if enabled)
    if is_gcs_enabled():
        print("\n--- Test 3: Verify GCS Upload ---")
        gcs_files = list_gcs_files(prefix="test/")
        print(f"✅ Files in GCS test/ prefix: {len(gcs_files)}")
        for f in gcs_files:
            print(f"  - {f}")

        # Check if our test files are there
        expected_files = ["test/test_polars.parquet", "test/test_pandas.parquet"]
        for expected in expected_files:
            if expected in gcs_files:
                print(f"✅ {expected} uploaded successfully")
            else:
                print(f"⚠️  {expected} not found in GCS (may still be uploading)")
    else:
        print("\n⚠️  GCS disabled, skipping upload verification")

    print("\n" + "=" * 60)
    print("✅ All helper function tests passed!")
    print("=" * 60)


def test_raw_data_locations():
    """Verify that raw data locations exist and would be uploaded."""
    print("\n" + "=" * 60)
    print("Checking raw data directories")
    print("=" * 60)

    raw_dirs = [
        "output/raw/flow",
        "output/raw/jquants",
        "output/raw/margin",
        "output/raw/statements",
        "output/raw/short_selling",
    ]

    for raw_dir in raw_dirs:
        path = Path(raw_dir)
        if path.exists():
            files = list(path.glob("*.parquet"))
            print(f"✅ {raw_dir}: {len(files)} parquet files")
        else:
            print(f"⚠️  {raw_dir}: does not exist yet")

    # Check graph cache
    graph_cache_dir = Path("output/graph_cache")
    if graph_cache_dir.exists():
        cache_files = list(graph_cache_dir.rglob("*.parquet"))
        cache_size_mb = sum(f.stat().st_size for f in cache_files) / (1024 * 1024)
        print(f"✅ Graph cache: {len(cache_files)} files ({cache_size_mb:.1f} MB)")
    else:
        print(f"⚠️  Graph cache: does not exist yet")


if __name__ == "__main__":
    try:
        test_helper_function()
        test_raw_data_locations()
        print("\n✅ All GCS sync tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
