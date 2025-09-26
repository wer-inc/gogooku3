#!/usr/bin/env python3
"""Check if target values in the dataset are valid"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Check the actual data location
data_dir = Path("/home/ubuntu/gogooku3-standalone/output/atft_data/train")
parquet_files = sorted(data_dir.glob("*.parquet"))[:10]  # Check first 10 files

print(f"Checking {len(parquet_files)} files from {data_dir}")
print("-" * 60)

for i, file in enumerate(parquet_files):
    df = pd.read_parquet(file)
    
    # Check for target columns (feat_ret_*d)
    target_cols = [col for col in df.columns if col.startswith('feat_ret_') and col.endswith('d')]
    
    print(f"\nFile {i}: {file.name}")
    print(f"  Shape: {df.shape}")
    if 'date' in df.columns:
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    
    if not target_cols:
        print(f"  ❌ No feat_ret_*d columns found!")
        print(f"  Available columns starting with 'feat': {[c for c in df.columns if c.startswith('feat')][:10]}")
        # Check for alternative target names
        alt_targets = [col for col in df.columns if 'ret' in col.lower() or 'return' in col.lower()][:10]
        print(f"  Columns with 'ret'/'return': {alt_targets}")
    else:
        for col in target_cols:
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
                print(f"    Range: [{valid_values.min():.6f}, {valid_values.max():.6f}]")
                # Show sample values
                sample = valid_values[:5]
                print(f"    Sample: {[f'{v:.6f}' for v in sample]}")
    
    if i >= 2:  # Check first 3 files
        break

# Also check column structure
print("\n" + "=" * 60)
print("COLUMN STRUCTURE ANALYSIS")
print("=" * 60)

if parquet_files:
    df = pd.read_parquet(parquet_files[0])
    print(f"Total columns: {len(df.columns)}")
    print(f"Columns starting with 'feat_ret_': {[c for c in df.columns if c.startswith('feat_ret_')]}")
    
    # Group columns by prefix
    prefixes = {}
    for col in df.columns:
        prefix = col.split('_')[0] if '_' in col else col
        if prefix not in prefixes:
            prefixes[prefix] = []
        prefixes[prefix].append(col)
    
    print("\nColumn prefixes:")
    for prefix, cols in sorted(prefixes.items())[:20]:
        print(f"  {prefix}: {len(cols)} columns")
        if len(cols) <= 5:
            print(f"    -> {cols}")
