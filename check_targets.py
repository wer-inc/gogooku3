#!/usr/bin/env python3
"""Check if target values in the dataset are valid"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Check a sample of the dataset
data_dir = Path("/home/ubuntu/gogooku3-standalone/output/atft_data")
parquet_files = sorted(data_dir.glob("*.parquet"))[:10]  # Check first 10 files

print(f"Checking {len(parquet_files)} files for target values...")
print("-" * 60)

for i, file in enumerate(parquet_files):
    df = pd.read_parquet(file)
    
    # Check for target columns (feat_ret_*d)
    target_cols = [col for col in df.columns if col.startswith('feat_ret_') and col.endswith('d')]
    
    if not target_cols:
        print(f"❌ File {i}: {file.name} - No target columns found!")
        continue
    
    print(f"\nFile {i}: {file.name}")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    
    for col in target_cols:
        if col in df.columns:
            values = df[col].values
            finite = np.isfinite(values)
            nonzero = values != 0
            
            print(f"  {col}:")
            print(f"    Valid: {finite.sum()}/{len(values)} ({100*finite.sum()/len(values):.1f}%)")
            print(f"    Non-zero: {nonzero.sum()}/{len(values)} ({100*nonzero.sum()/len(values):.1f}%)")
            
            if finite.any():
                valid_values = values[finite]
                print(f"    Mean: {valid_values.mean():.6f}")
                print(f"    Std: {valid_values.std():.6f}")
                print(f"    Min: {valid_values.min():.6f}")
                print(f"    Max: {valid_values.max():.6f}")
                # Show sample values
                sample = valid_values[:5]
                print(f"    Sample: {sample.tolist()}")
            else:
                print(f"    ⚠️  All values are invalid!")
    
    if i >= 4:  # Check first 5 files in detail
        break

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

# Check overall statistics
all_targets = []
for file in parquet_files[:50]:  # Check more files for summary
    df = pd.read_parquet(file)
    for col in ['feat_ret_1d', 'feat_ret_5d', 'feat_ret_10d', 'feat_ret_20d']:
        if col in df.columns:
            all_targets.extend(df[col].values[np.isfinite(df[col].values)])

if all_targets:
    all_targets = np.array(all_targets)
    print(f"Total valid target values checked: {len(all_targets)}")
    print(f"Non-zero values: {(all_targets != 0).sum()} ({100*(all_targets != 0).sum()/len(all_targets):.1f}%)")
    print(f"Overall mean: {all_targets.mean():.6f}")
    print(f"Overall std: {all_targets.std():.6f}")
else:
    print("❌ No valid target values found!")
