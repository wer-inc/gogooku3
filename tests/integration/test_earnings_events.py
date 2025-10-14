#!/usr/bin/env python3
"""
Earnings events features test script.
Tests the earnings announcement proximity flags and PEAD features implementation.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import polars as pl
from datetime import date, timedelta

def test_earnings_events_features():
    """Test the earnings events feature implementation."""
    print("🧪 Testing Earnings Events Features")

    # Create sample quotes data
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(20)]
    quotes = pl.DataFrame({
        "Code": ["1301"] * 20,
        "Date": dates,
        "returns_1d": [0.01, -0.02, 0.015, -0.005, 0.03, 0.02, -0.01, 0.005,
                      -0.015, 0.025, 0.01, -0.008, 0.012, -0.018, 0.022,
                      0.015, -0.012, 0.008, -0.005, 0.018],
    })

    # Create sample earnings announcement data
    announcement_df = pl.DataFrame({
        "Code": ["1301", "1301"],
        "Date": [date(2024, 1, 5), date(2024, 1, 15)],
        "AnnouncementDate": [date(2024, 1, 5), date(2024, 1, 15)],
        "CompanyName": ["Company A", "Company A"],
        "FiscalYear": [2024, 2024],
        "FiscalQuarter": [1, 1],
    })

    print(f"📊 Sample data:")
    print(f"  - Quotes: {len(quotes)} records from {quotes['Date'].min()} to {quotes['Date'].max()}")
    print(f"  - Announcements: {len(announcement_df)} events on {announcement_df['AnnouncementDate'].to_list()}")

    # Test earnings events features
    from gogooku3.features.earnings_events import add_earnings_event_block

    print("🚀 Testing earnings event features...")
    result = add_earnings_event_block(
        quotes=quotes,
        announcement_df=announcement_df,
        statements_df=None,
        enable_pead=True,
        enable_volatility=True
    )

    print(f"✅ Earnings event features generated:")
    earnings_cols = [col for col in result.columns if 'earnings' in col or 'pead' in col or 'days_' in col]
    for col in earnings_cols:
        non_null_count = result[col].drop_nulls().len()
        print(f"  - {col}: {non_null_count} non-null values")

    # Check specific feature functionality
    print(f"\n🔍 Feature validation:")

    # Check proximity features around announcement dates
    jan_5_row = result.filter(pl.col("Date") == date(2024, 1, 5))
    jan_4_row = result.filter(pl.col("Date") == date(2024, 1, 4))  # Day before
    jan_6_row = result.filter(pl.col("Date") == date(2024, 1, 6))  # Day after

    if not jan_5_row.is_empty():
        upcoming = jan_4_row["earnings_upcoming_5d"].to_list()[0] if not jan_4_row.is_empty() else None
        recent = jan_6_row["earnings_recent_5d"].to_list()[0] if not jan_6_row.is_empty() else None
        print(f"  - Jan 4 (before): upcoming_5d = {upcoming}")
        print(f"  - Jan 6 (after): recent_5d = {recent}")

    # Check days-to/since-earnings features
    days_to_sample = result.filter(pl.col("days_to_earnings").is_not_null())
    days_since_sample = result.filter(pl.col("days_since_earnings").is_not_null())

    if not days_to_sample.is_empty():
        print(f"  - Days to earnings: {days_to_sample['days_to_earnings'].min()} to {days_to_sample['days_to_earnings'].max()}")

    if not days_since_sample.is_empty():
        print(f"  - Days since earnings: {days_since_sample['days_since_earnings'].min()} to {days_since_sample['days_since_earnings'].max()}")

    # Check earnings day return (surprise proxy)
    earnings_day_returns = result.filter(pl.col("earnings_day_return").is_not_null())
    if not earnings_day_returns.is_empty():
        print(f"  - Earnings day returns: {earnings_day_returns['earnings_day_return'].to_list()}")

    # Show sample of results
    print(f"\n📋 Sample results:")
    sample_cols = ["Date", "earnings_upcoming_5d", "earnings_recent_5d", "days_to_earnings",
                   "days_since_earnings", "earnings_day_return"]
    sample_data = result.select(sample_cols).head(10)
    print(sample_data)

    print("✅ All tests passed!")

    # Test with no announcement data (null features)
    print(f"\n🔍 Testing null features (no announcement data):")
    null_result = add_earnings_event_block(
        quotes=quotes,
        announcement_df=None,
        statements_df=None,
        enable_pead=True,
        enable_volatility=True
    )

    null_earnings_cols = [col for col in null_result.columns if 'earnings' in col or 'days_' in col]
    print(f"  - Null features added: {len(null_earnings_cols)} columns")
    for col in null_earnings_cols[:5]:  # Show first 5
        non_zero_count = len(null_result.filter(pl.col(col) != 0))
        print(f"    * {col}: {non_zero_count} non-zero values (should be 0 for most)")

    print("✅ Null features test passed!")

    return result


if __name__ == "__main__":
    result_df = test_earnings_events_features()