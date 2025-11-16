#!/usr/bin/env python3
"""Test J-Quants short_selling API availability for different periods."""

import os
import sys
from pathlib import Path

# Add gogooku5 data sources to path
sys.path.insert(0, str(Path(__file__).parent / "gogooku5" / "data" / "src"))

from builder.api.advanced_fetcher import AdvancedJQuantsFetcher
from builder.config.settings import DatasetBuilderSettings


def test_short_selling_api():
    """Test short_selling API for 2024Q1, 2025Q1, and recent periods."""

    # Create fetcher (uses .env automatically)
    fetcher = AdvancedJQuantsFetcher()

    # Test periods
    test_periods = [
        ("2024Q1", "2024-01-01", "2024-03-31"),
        ("2025Q1", "2025-01-01", "2025-03-31"),
        ("Recent", "2025-11-01", "2025-11-15"),
        ("Sample Week 2024", "2024-01-15", "2024-01-19"),  # One week in 2024
    ]

    print("=" * 80)
    print("Testing J-Quants short_selling API")
    print("=" * 80)

    for period_name, start_date, end_date in test_periods:
        print(f"\n{'=' * 80}")
        print(f"Period: {period_name} ({start_date} to {end_date})")
        print("=" * 80)

        try:
            # Fetch short_selling data
            df = fetcher.fetch_short_selling(start=start_date, end=end_date)

            if df is None or len(df) == 0:
                print(f"❌ NO DATA RETURNED")
                continue

            print(f"✅ Data returned: {len(df):,} rows")
            print(f"\nColumns ({len(df.columns)}): {', '.join(df.columns)}")

            # Show sample data
            print(f"\nFirst 5 rows:")
            print(df.head(5))

            # Check for NULL values
            null_counts = {}
            for col in df.columns:
                null_count = df[col].null_count()
                if null_count > 0:
                    null_counts[col] = null_count

            if null_counts:
                print(f"\nColumns with NULL values:")
                for col, count in sorted(null_counts.items(), key=lambda x: x[1], reverse=True):
                    pct = (count / len(df)) * 100
                    print(f"  - {col}: {count:,} ({pct:.1f}%)")
            else:
                print(f"\n✅ No NULL values in any column")

            # Show unique dates
            if "Date" in df.columns:
                unique_dates = df["Date"].n_unique()
                print(f"\nUnique dates: {unique_dates}")
                print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

            # Show unique sectors if present
            if "Sector33Code" in df.columns:
                unique_sectors = df["Sector33Code"].n_unique()
                print(f"Unique sectors: {unique_sectors}")

        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("Test completed")
    print("=" * 80)


if __name__ == "__main__":
    test_short_selling_api()
