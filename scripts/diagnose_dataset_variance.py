#!/usr/bin/env python3
"""
Diagnose per-day target variance in ML dataset.

This script checks if cross-sectional normalization in the dataset
is eliminating the signal we're trying to predict.
"""

import polars as pl
import numpy as np
from pathlib import Path


def analyze_per_day_variance(parquet_path: str):
    """Analyze per-day variance of targets in the dataset."""

    print(f"\n{'='*80}")
    print(f"Dataset Variance Diagnostic")
    print(f"{'='*80}")
    print(f"Dataset: {parquet_path}")

    # Load dataset
    print("\n📂 Loading dataset...")
    df = pl.read_parquet(parquet_path)
    print(f"   Total rows: {len(df):,}")
    print(f"   Total columns: {len(df.columns)}")

    # Identify target columns
    target_cols = [col for col in df.columns if col.startswith('horizon_')]
    print(f"   Target columns: {target_cols}")

    if 'Date' not in df.columns:
        print("\n❌ ERROR: 'Date' column not found in dataset")
        return

    # Analyze overall statistics
    print(f"\n{'='*80}")
    print(f"Overall Statistics (Before Per-Day Grouping)")
    print(f"{'='*80}")

    for col in target_cols:
        if col in df.columns:
            stats = df.select(pl.col(col)).describe()
            mean_val = df[col].mean()
            std_val = df[col].std()
            print(f"\n{col}:")
            print(f"   Mean: {mean_val:.6f}")
            print(f"   Std:  {std_val:.6f}")
            print(f"   Min:  {df[col].min():.6f}")
            print(f"   Max:  {df[col].max():.6f}")

    # Analyze per-day statistics
    print(f"\n{'='*80}")
    print(f"Per-Day Statistics (Cross-Sectional Analysis)")
    print(f"{'='*80}")

    # Group by date and compute per-day statistics
    per_day_stats = []

    for col in target_cols:
        if col not in df.columns:
            continue

        # Compute per-day mean and std
        daily_stats = df.group_by('Date').agg([
            pl.col(col).mean().alias('mean'),
            pl.col(col).std().alias('std'),
            pl.col(col).count().alias('n_stocks')
        ])

        # Overall per-day statistics
        mean_of_daily_means = daily_stats['mean'].mean()
        std_of_daily_means = daily_stats['mean'].std()
        mean_of_daily_stds = daily_stats['std'].mean()
        std_of_daily_stds = daily_stats['std'].std()

        print(f"\n{col}:")
        print(f"   Per-day mean: μ={mean_of_daily_means:.6f}, σ={std_of_daily_means:.6f}")
        print(f"   Per-day std:  μ={mean_of_daily_stds:.6f}, σ={std_of_daily_stds:.6f}")

        # Sample a few days
        print(f"\n   Sample days (first 10):")
        sample = daily_stats.head(10)
        for row in sample.iter_rows(named=True):
            date = row['Date']
            mean = row['mean']
            std = row['std']
            n = row['n_stocks']
            print(f"      {date}: mean={mean:+.6f}, std={std:.6f}, n={n}")

        # Check for zero-variance days
        zero_var_days = daily_stats.filter(pl.col('std') < 1e-8)
        if len(zero_var_days) > 0:
            print(f"\n   ⚠️  WARNING: {len(zero_var_days)} days with near-zero variance!")
            print(f"      Sample zero-variance days:")
            for row in zero_var_days.head(5).iter_rows(named=True):
                date = row['Date']
                mean = row['mean']
                std = row['std']
                n = row['n_stocks']
                print(f"         {date}: mean={mean:+.6f}, std={std:.8f}, n={n}")

        # Check if per-day stds are suspiciously uniform
        if mean_of_daily_stds < 0.01:
            print(f"\n   🚨 CRITICAL: Per-day std is very low ({mean_of_daily_stds:.6f})")
            print(f"      This suggests cross-sectional normalization is killing variance!")

        per_day_stats.append({
            'target': col,
            'overall_mean': df[col].mean(),
            'overall_std': df[col].std(),
            'mean_of_daily_means': mean_of_daily_means,
            'std_of_daily_means': std_of_daily_means,
            'mean_of_daily_stds': mean_of_daily_stds,
            'std_of_daily_stds': std_of_daily_stds,
            'zero_var_days': len(zero_var_days),
            'total_days': len(daily_stats)
        })

    # Summary
    print(f"\n{'='*80}")
    print(f"Summary")
    print(f"{'='*80}")

    for stats in per_day_stats:
        print(f"\n{stats['target']}:")
        print(f"   Overall variance:     {stats['overall_std']:.6f}")
        print(f"   Avg per-day variance: {stats['mean_of_daily_stds']:.6f}")
        print(f"   Ratio (should be ~1): {stats['mean_of_daily_stds'] / stats['overall_std']:.4f}")
        print(f"   Zero-variance days:   {stats['zero_var_days']} / {stats['total_days']}")

        # Diagnosis
        ratio = stats['mean_of_daily_stds'] / stats['overall_std'] if stats['overall_std'] > 0 else 0
        if ratio < 0.1:
            print(f"   🚨 DIAGNOSIS: Cross-sectional variance is {ratio*100:.1f}% of overall variance")
            print(f"                 Likely cause: Dataset is already cross-sectionally normalized!")
        elif ratio > 0.8:
            print(f"   ✅ DIAGNOSIS: Cross-sectional variance is preserved ({ratio*100:.1f}%)")
        else:
            print(f"   ⚠️  DIAGNOSIS: Partial variance loss ({ratio*100:.1f}% preserved)")

    # Check for normalization artifacts
    print(f"\n{'='*80}")
    print(f"Normalization Artifact Detection")
    print(f"{'='*80}")

    for col in target_cols:
        if col not in df.columns:
            continue

        # Check if values cluster around 0 with std ~1 (Z-score signature)
        mean_val = df[col].mean()
        std_val = df[col].std()

        print(f"\n{col}:")
        if abs(mean_val) < 0.01 and 0.95 < std_val < 1.05:
            print(f"   🚨 LIKELY Z-SCORED: mean={mean_val:.6f}, std={std_val:.6f}")
            print(f"      Dataset appears to be globally normalized (mean~0, std~1)")
        else:
            print(f"   Mean: {mean_val:.6f}, Std: {std_val:.6f}")
            print(f"   Not standard-normalized (different scale)")


def main():
    """Main entry point."""

    # Default dataset path
    default_path = "output/datasets/ml_dataset_latest_full.parquet"

    if not Path(default_path).exists():
        print(f"❌ Dataset not found: {default_path}")

        # Try to find any ML dataset
        dataset_dir = Path("output/datasets")
        if dataset_dir.exists():
            parquet_files = list(dataset_dir.glob("ml_dataset_*.parquet"))
            if parquet_files:
                print(f"\nFound {len(parquet_files)} dataset(s):")
                for i, f in enumerate(parquet_files, 1):
                    print(f"   {i}. {f}")
                default_path = str(parquet_files[0])
                print(f"\nUsing: {default_path}")
            else:
                print("\n❌ No ML datasets found in output/datasets/")
                return
        else:
            print("\n❌ output/datasets/ directory not found")
            return

    analyze_per_day_variance(default_path)

    print(f"\n{'='*80}")
    print(f"Diagnostic Complete")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
