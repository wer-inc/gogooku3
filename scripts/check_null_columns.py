#!/usr/bin/env python3
"""
Check for all-null columns in gogooku5 datasets.
Analyzes 2024 and 2025 data to identify columns with 100% null values.
"""

import polars as pl
import sys
from pathlib import Path

def analyze_null_columns(parquet_path: str, dataset_name: str):
    """Analyze a parquet file for all-null columns."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {dataset_name}")
    print(f"File: {parquet_path}")
    print(f"{'='*80}")

    try:
        # Read parquet file
        df = pl.read_parquet(parquet_path)
        total_rows = len(df)

        print(f"Total rows: {total_rows:,}")
        print(f"Total columns: {len(df.columns):,}")

        # Calculate null counts for each column
        null_counts = df.null_count()

        # Find columns that are 100% null
        all_null_cols = []
        for col in df.columns:
            null_count = null_counts[col][0]
            if null_count == total_rows:
                all_null_cols.append(col)

        # Find columns with high null percentage (>95%)
        high_null_cols = []
        for col in df.columns:
            null_count = null_counts[col][0]
            null_pct = (null_count / total_rows) * 100
            if 95 < null_pct < 100:
                high_null_cols.append((col, null_pct))

        # Print results
        print(f"\n📊 Null Analysis Results:")
        print(f"  All-null columns (100%): {len(all_null_cols)}")
        print(f"  High-null columns (>95%): {len(high_null_cols)}")

        if all_null_cols:
            print(f"\n❌ ALL-NULL COLUMNS ({len(all_null_cols)}):")
            for i, col in enumerate(sorted(all_null_cols), 1):
                print(f"  {i:3d}. {col}")
        else:
            print(f"\n✅ No all-null columns found!")

        if high_null_cols:
            print(f"\n⚠️  HIGH-NULL COLUMNS (>95% but <100%):")
            for col, pct in sorted(high_null_cols, key=lambda x: x[1], reverse=True):
                print(f"  {col:50s} {pct:6.2f}%")

        return all_null_cols, high_null_cols

    except Exception as e:
        print(f"❌ Error analyzing {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return [], []

def main():
    base_path = Path("/workspace/gogooku3/gogooku5/data/output/datasets")

    datasets = [
        ("ml_dataset_2024_full.parquet", "2024 Dataset"),
        ("ml_dataset_2025_full.parquet", "2025 Dataset"),
        ("ml_dataset_2024_2025_full_for_apex.parquet", "2024-2025 Combined (APEX)"),
    ]

    all_results = {}

    for filename, name in datasets:
        filepath = base_path / filename
        if filepath.exists():
            all_null, high_null = analyze_null_columns(str(filepath), name)
            all_results[name] = {
                'all_null': all_null,
                'high_null': high_null
            }
        else:
            print(f"\n⚠️  File not found: {filepath}")

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    # Find columns that are all-null in ALL datasets
    if all_results:
        all_datasets_null = set(all_results[list(all_results.keys())[0]]['all_null'])
        for dataset_name, results in all_results.items():
            all_datasets_null &= set(results['all_null'])

        if all_datasets_null:
            print(f"\n⚠️  Columns that are ALL-NULL in ALL datasets ({len(all_datasets_null)}):")
            for col in sorted(all_datasets_null):
                print(f"  - {col}")
        else:
            print(f"\n✅ No columns are all-null across all datasets")

        # Show dataset-specific all-null columns
        print(f"\n📋 Dataset-specific all-null columns:")
        for dataset_name, results in all_results.items():
            print(f"\n  {dataset_name}: {len(results['all_null'])} all-null columns")

    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()
