#!/usr/bin/env python3
"""
Test script for API parallelization (futures + FS details).

Usage:
    python test_parallelization.py
"""
import logging
import sys
import time
from pathlib import Path

# Configure logging to show parallel progress
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.builder.api.advanced_fetcher import AdvancedJQuantsFetcher
from src.builder.config.settings import get_settings


def test_parallel_futures():
    """Test parallel futures API fetch."""
    print("\n" + "=" * 70)
    print("TEST 1: Futures API Parallelization")
    print("=" * 70)

    settings = get_settings()
    fetcher = AdvancedJQuantsFetcher(settings=settings)

    # Test with 1 week of data (2024-01-01 to 2024-01-05, 5 business days)
    print("\n📊 Fetching futures data (2024-01-01 → 2024-01-05)")
    print(f"Parallel mode: {settings.index_option_parallel_fetch}")
    print(f"Concurrency: {settings.index_option_parallel_concurrency}")

    start_time = time.time()

    try:
        df = fetcher.fetch_futures(start="2024-01-01", end="2024-01-05")
        elapsed = time.time() - start_time

        print(f"\n✅ Futures fetch completed in {elapsed:.2f}s")
        print(f"Records: {len(df):,}")
        if len(df) > 0:
            print(f"Columns: {df.columns[:5]}...")
            print(
                f"Date range: {df['Date'].min() if 'Date' in df.columns else 'N/A'} → {df['Date'].max() if 'Date' in df.columns else 'N/A'}"
            )
    except Exception as e:
        print(f"\n❌ Futures fetch failed: {e}")
        import traceback

        traceback.print_exc()


def test_parallel_fs_details():
    """Test parallel FS details API fetch."""
    print("\n" + "=" * 70)
    print("TEST 2: FS Details API Parallelization")
    print("=" * 70)

    settings = get_settings()
    fetcher = AdvancedJQuantsFetcher(settings=settings)

    # Test with 1 week of data (2024-01-01 to 2024-01-07, 7 days)
    print("\n📊 Fetching FS details (2024-01-01 → 2024-01-07)")
    print(f"Parallel mode: {settings.index_option_parallel_fetch}")
    print(f"Concurrency: {settings.index_option_parallel_concurrency}")

    start_time = time.time()

    try:
        df = fetcher.fetch_fs_details(start="2024-01-01", end="2024-01-07")
        elapsed = time.time() - start_time

        print(f"\n✅ FS details fetch completed in {elapsed:.2f}s")
        print(f"Records: {len(df):,}")
        if len(df) > 0:
            print(f"Columns: {df.columns[:5]}...")
            print(f"Unique codes: {df['Code'].n_unique() if 'Code' in df.columns else 'N/A'}")
    except Exception as e:
        print(f"\n❌ FS details fetch failed: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Run all parallelization tests."""
    print("\n" + "=" * 70)
    print("API PARALLELIZATION TEST SUITE")
    print("=" * 70)

    # Test both endpoints
    test_parallel_futures()
    test_parallel_fs_details()

    print("\n" + "=" * 70)
    print("TEST SUITE COMPLETE")
    print("=" * 70)
    print("\nLook for these log messages to confirm parallelization:")
    print("  - 'Parallel fetch enabled: futures data for N business days'")
    print("  - 'Futures parallel progress: X/Y days processed'")
    print("  - 'Parallel fetch enabled: FS details for N days'")
    print("  - 'FS details parallel progress: X/Y days processed'")
    print("\nIf you see sequential progress messages instead, parallelization is disabled.")


if __name__ == "__main__":
    main()
