#!/usr/bin/env python3
"""
Extract specific date range from a merged dataset.

Usage:
    python extract_date_range.py \
        --input data/output/datasets/ml_dataset_2022_2024_merged.parquet \
        --output data/output/datasets/ml_dataset_2023_2024_extracted.parquet \
        --start-date 2023-01-01 \
        --end-date 2024-12-31
"""

import argparse
import sys
from pathlib import Path

import polars as pl


def extract_date_range(
    input_path: str,
    output_path: str,
    start_date: str,
    end_date: str,
    date_col: str = "date",
) -> None:
    """Extract rows within specified date range."""

    print(f"\n{'='*80}")
    print(f"📅 Extracting Date Range: {start_date} → {end_date}")
    print(f"{'='*80}\n")

    # Read input dataset
    print(f"📂 Reading input: {input_path}")
    df = pl.read_parquet(input_path)
    print(f"   Total rows: {len(df):,}")
    print(f"   Total cols: {len(df.columns):,}")

    # Check if date column exists
    if date_col not in df.columns:
        print(f"\n❌ ERROR: Column '{date_col}' not found in dataset")
        print(f"   Available date-like columns: {[c for c in df.columns if 'date' in c.lower()]}")
        sys.exit(1)

    # Get date range of input
    input_start = df[date_col].min()
    input_end = df[date_col].max()
    print(f"   Input date range: {input_start} → {input_end}")

    # Filter by date range
    print("\n🔍 Filtering rows...")
    df_filtered = df.filter(
        (pl.col(date_col) >= pl.lit(start_date).str.to_date()) & (pl.col(date_col) <= pl.lit(end_date).str.to_date())
    )

    filtered_rows = len(df_filtered)
    print(f"   Filtered rows: {filtered_rows:,} ({filtered_rows / len(df) * 100:.2f}% of input)")

    if filtered_rows == 0:
        print(f"\n⚠️  WARNING: No rows found in date range {start_date} → {end_date}")
        print(f"   Input range: {input_start} → {input_end}")
        sys.exit(1)

    # Get actual date range of filtered data
    actual_start = df_filtered[date_col].min()
    actual_end = df_filtered[date_col].max()
    print(f"   Actual range: {actual_start} → {actual_end}")

    # Write output
    print(f"\n💾 Writing output: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_filtered.write_parquet(output_path)

    # Verify output
    output_size = Path(output_path).stat().st_size / (1024**3)
    print(f"   Output size: {output_size:.2f} GB")

    print(f"\n{'='*80}")
    print("✅ Date range extraction complete")
    print(f"{'='*80}\n")
    print("Summary:")
    print(f"  Input:  {len(df):,} rows × {len(df.columns):,} cols")
    print(f"  Output: {filtered_rows:,} rows × {len(df_filtered.columns):,} cols")
    print(f"  Date range: {actual_start} → {actual_end}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Extract date range from merged dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract 2023-2024 from 2022-2024 merged dataset
    python extract_date_range.py \\
        --input data/output/datasets/ml_dataset_2022_2024_merged.parquet \\
        --output data/output/datasets/ml_dataset_2023_2024_extracted.parquet \\
        --start-date 2023-01-01 \\
        --end-date 2024-12-31
        """,
    )

    parser.add_argument("--input", "-i", required=True, help="Input parquet file path")
    parser.add_argument("--output", "-o", required=True, help="Output parquet file path")
    parser.add_argument("--start-date", "-s", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", "-e", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--date-col", default="date", help="Name of date column (default: 'date')")

    args = parser.parse_args()

    extract_date_range(
        input_path=args.input,
        output_path=args.output,
        start_date=args.start_date,
        end_date=args.end_date,
        date_col=args.date_col,
    )


if __name__ == "__main__":
    main()
