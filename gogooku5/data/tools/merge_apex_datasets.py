#!/usr/bin/env python3
"""
Merge multiple year datasets for APEX-Ranker training.

This script combines 2023, 2024, and 2025 datasets into a single unified dataset,
handling schema differences and ensuring data quality.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import polars as pl


def merge_datasets(input_paths: list[str], output_path: str, verbose: bool = True):
    """
    Merge multiple parquet datasets with schema alignment.

    Args:
        input_paths: List of input parquet file paths
        output_path: Output parquet file path
        verbose: Print progress messages
    """
    if verbose:
        print("=" * 80)
        print("APEX-Ranker Dataset Merger")
        print("=" * 80)

    # Step 1: Load all datasets
    datasets = []
    all_columns = set()

    for i, input_path in enumerate(input_paths, 1):
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if verbose:
            print(f"\n[{i}/{len(input_paths)}] Loading: {path.name}")

        df = pl.read_parquet(input_path)

        if verbose:
            print(f"  Rows: {len(df):,}")
            print(f"  Columns: {len(df.columns):,}")
            print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
            print(f"  Unique dates: {df['Date'].n_unique()}")
            print(f"  Unique stocks: {df['Code'].n_unique()}")

        datasets.append((path.name, df))
        all_columns.update(df.columns)

    # Step 2: Identify schema differences
    if verbose:
        print("\n" + "=" * 80)
        print("Schema Analysis")
        print("=" * 80)

        # Find columns present in all datasets
        common_columns = set(datasets[0][1].columns)
        for name, df in datasets[1:]:
            common_columns &= set(df.columns)

        print(f"Common columns: {len(common_columns):,}")
        print(f"Total unique columns: {len(all_columns):,}")

        # Find dataset-specific columns
        for name, df in datasets:
            unique_cols = set(df.columns) - common_columns
            if unique_cols:
                print(f"\n{name} unique columns ({len(unique_cols)}):")
                for col in sorted(unique_cols):
                    print(f"  - {col}")

    # Step 3: Align schemas (use common columns only for safety)
    if verbose:
        print("\n" + "=" * 80)
        print("Schema Alignment")
        print("=" * 80)
        print(f"Using {len(common_columns):,} common columns")

    # Sort common columns for consistent ordering
    common_columns_sorted = sorted(common_columns)

    aligned_datasets = []
    for name, df in datasets:
        # Select only common columns in consistent order
        df_aligned = df.select(common_columns_sorted)
        aligned_datasets.append(df_aligned)

        if verbose:
            print(f"  {name}: {len(df_aligned):,} rows x {len(df_aligned.columns)} cols")

    # Step 4: Concatenate datasets
    if verbose:
        print("\n" + "=" * 80)
        print("Concatenating Datasets")
        print("=" * 80)

    merged_df = pl.concat(aligned_datasets, how="vertical")

    if verbose:
        print(f"Merged dataset: {len(merged_df):,} rows")

    # Step 5: Remove duplicates (if any)
    if verbose:
        print("\n" + "=" * 80)
        print("Duplicate Removal")
        print("=" * 80)

    # Check for duplicates on (Date, Code)
    initial_rows = len(merged_df)
    merged_df = merged_df.unique(subset=["Date", "Code"], maintain_order=True)
    duplicates_removed = initial_rows - len(merged_df)

    if verbose:
        if duplicates_removed > 0:
            print(f"⚠️  Removed {duplicates_removed:,} duplicate rows")
        else:
            print("✅ No duplicates found")

    # Step 6: Sort by Date, Code
    if verbose:
        print("\n" + "=" * 80)
        print("Sorting Dataset")
        print("=" * 80)

    merged_df = merged_df.sort(["Date", "Code"])

    # Step 7: Save merged dataset
    if verbose:
        print("\n" + "=" * 80)
        print("Saving Merged Dataset")
        print("=" * 80)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged_df.write_parquet(output_path, compression="zstd")

    if verbose:
        file_size = output_path.stat().st_size / (1024**3)  # GB
        print(f"✅ Saved to: {output_path}")
        print(f"   Size: {file_size:.2f} GB")
        print(f"   Rows: {len(merged_df):,}")
        print(f"   Columns: {len(merged_df.columns):,}")

    # Step 8: Generate metadata
    metadata = {
        "created_at": datetime.now().isoformat(),
        "script": "merge_apex_datasets.py",
        "input_files": [str(Path(p).name) for p in input_paths],
        "output_file": str(output_path.name),
        "total_rows": len(merged_df),
        "total_columns": len(merged_df.columns),
        "date_range": {"start": str(merged_df["Date"].min()), "end": str(merged_df["Date"].max())},
        "unique_dates": merged_df["Date"].n_unique(),
        "unique_stocks": merged_df["Code"].n_unique(),
        "duplicates_removed": duplicates_removed,
        "common_columns": len(common_columns),
        "total_unique_columns": len(all_columns),
        "schema_differences": {name: len(set(df.columns) - common_columns) for name, df in datasets},
    }

    metadata_path = output_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"✅ Metadata saved to: {metadata_path}")

    # Step 9: Print final summary
    if verbose:
        print("\n" + "=" * 80)
        print("Merge Complete")
        print("=" * 80)
        print("📊 Final Dataset:")
        print(f"   Rows: {len(merged_df):,}")
        print(f"   Columns: {len(merged_df.columns):,}")
        print(f"   Date range: {merged_df['Date'].min()} to {merged_df['Date'].max()}")
        print(f"   Trading days: {merged_df['Date'].n_unique()}")
        print(f"   Unique stocks: {merged_df['Code'].n_unique()}")
        print(f"   Avg stocks/day: {len(merged_df) / merged_df['Date'].n_unique():.0f}")
        print("\n📁 Output files:")
        print(f"   Dataset: {output_path}")
        print(f"   Metadata: {metadata_path}")

    return merged_df


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple year datasets for APEX-Ranker training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge 2023+2024+2025
  python merge_apex_datasets.py \\
    --inputs ml_dataset_2023_full.parquet \\
             ml_dataset_2024_full.parquet \\
             ml_dataset_2025_full.parquet \\
    --output ml_dataset_2023_2024_2025_full.parquet

  # Using absolute paths
  python merge_apex_datasets.py \\
    --inputs gogooku5/data/output/datasets/ml_dataset_2023_full.parquet \\
             gogooku5/data/output/datasets/ml_dataset_2024_full.parquet \\
             gogooku5/data/output/datasets/ml_dataset_2025_full.parquet \\
    --output gogooku5/data/output/datasets/ml_dataset_2023_2024_2025_full.parquet
        """,
    )
    parser.add_argument("--inputs", nargs="+", required=True, help="Input parquet files to merge")
    parser.add_argument("--output", required=True, help="Output parquet file path")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages")

    args = parser.parse_args()

    try:
        merge_datasets(input_paths=args.inputs, output_path=args.output, verbose=not args.quiet)
        print("\n✅ Success!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
