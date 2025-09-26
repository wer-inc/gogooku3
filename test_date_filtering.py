#!/usr/bin/env python3
"""Test data loading with date filtering to verify we get valid targets"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Test parameters
MIN_DATE = "2018-01-01"
DATASET_PATH = "output/ml_dataset_latest_full.parquet"

print("=" * 60)
print("TESTING DATE FILTERING FOR VALID TARGETS")
print("=" * 60)

# Load dataset
print(f"\nLoading dataset: {DATASET_PATH}")
df = pd.read_parquet(DATASET_PATH)
print(f"Full dataset shape: {df.shape}")
print(f"Full date range: {df['date'].min()} to {df['date'].max()}")

# Apply date filter
df_filtered = df[df['date'] >= MIN_DATE]
print(f"\nAfter filtering (>= {MIN_DATE}):")
print(f"  Shape: {df_filtered.shape}")
print(f"  Date range: {df_filtered['date'].min()} to {df_filtered['date'].max()}")
print(f"  Data retained: {100*df_filtered.shape[0]/df.shape[0]:.1f}%")

# Analyze target columns
print("\n" + "=" * 60)
print("TARGET COLUMN ANALYSIS")
print("=" * 60)

target_cols = [col for col in df.columns if col.startswith('feat_ret_') and col.endswith('d')]
print(f"\nFound {len(target_cols)} target columns: {target_cols}")

for col in target_cols:
    print(f"\n{col} Analysis:")

    # Full dataset
    full_values = df[col].values
    full_valid = np.isfinite(full_values)
    full_nonzero = (full_values != 0) & full_valid

    print(f"  Full dataset:")
    print(f"    Valid: {full_valid.sum()}/{len(full_values)} ({100*full_valid.sum()/len(full_values):.1f}%)")
    print(f"    Non-zero: {full_nonzero.sum()}/{len(full_values)} ({100*full_nonzero.sum()/len(full_values):.1f}%)")

    if full_valid.any():
        valid_full = full_values[full_valid]
        print(f"    Mean: {valid_full.mean():.6f}")
        print(f"    Std: {valid_full.std():.6f}")

    # Filtered dataset
    filt_values = df_filtered[col].values
    filt_valid = np.isfinite(filt_values)
    filt_nonzero = (filt_values != 0) & filt_valid

    print(f"  After date filter (>= {MIN_DATE}):")
    print(f"    Valid: {filt_valid.sum()}/{len(filt_values)} ({100*filt_valid.sum()/len(filt_values):.1f}%)")
    print(f"    Non-zero: {filt_nonzero.sum()}/{len(filt_values)} ({100*filt_nonzero.sum()/len(filt_values):.1f}%)")

    if filt_valid.any():
        valid_filt = filt_values[filt_valid]
        print(f"    Mean: {valid_filt.mean():.6f}")
        print(f"    Std: {valid_filt.std():.6f}")
        print(f"    Sample values: {valid_filt[:5].tolist()}")

# Check early vs late data quality
print("\n" + "=" * 60)
print("DATA QUALITY BY TIME PERIOD")
print("=" * 60)

# Group by year
df['year'] = pd.to_datetime(df['date']).dt.year

for year in sorted(df['year'].unique())[-5:]:  # Last 5 years
    year_data = df[df['year'] == year]
    print(f"\nYear {year} ({len(year_data)} samples):")

    for col in target_cols[:2]:  # Check first 2 horizons
        values = year_data[col].values
        valid = np.isfinite(values)
        nonzero = (values != 0) & valid

        print(f"  {col}:")
        print(f"    Valid: {100*valid.sum()/len(values):.1f}%")
        print(f"    Non-zero: {100*nonzero.sum()/len(values):.1f}%")

print("\n" + "=" * 60)
print("RECOMMENDATION")
print("=" * 60)

# Check if 2018+ data has good quality
recent_data = df[df['date'] >= "2018-01-01"]
for col in target_cols:
    values = recent_data[col].values
    valid = np.isfinite(values)
    nonzero = (values != 0) & valid

    validity_rate = 100 * valid.sum() / len(values)
    nonzero_rate = 100 * nonzero.sum() / len(values)

    if validity_rate < 50:
        print(f"⚠️  {col}: Only {validity_rate:.1f}% valid in 2018+ data")
    elif nonzero_rate < 10:
        print(f"⚠️  {col}: Only {nonzero_rate:.1f}% non-zero in 2018+ data")
    else:
        print(f"✅ {col}: {validity_rate:.1f}% valid, {nonzero_rate:.1f}% non-zero")

print(f"\n💡 Recommendation: Use MIN_TRAINING_DATE='2018-01-01' or later")
print(f"   This filters out early data with invalid/missing targets")