#!/usr/bin/env python3
"""Check production dataset for valid targets"""


import numpy as np
import pandas as pd


def main():
    dataset_path = "output/ml_dataset_latest_full.parquet"

    print(f"Loading: {dataset_path}")
    df = pd.read_parquet(dataset_path)

    print(f"Dataset shape: {df.shape}")

    # Check for date column
    if "Date" in df.columns:
        date_col = "Date"
    elif "date" in df.columns:
        date_col = "date"
    else:
        print("❌ No date column found!")
        print(f"Available columns: {df.columns[:10].tolist()}")
        return

    print(f"Date column: {date_col}")
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")

    # Find target columns
    target_cols = [c for c in df.columns if 'returns_' in c and c.endswith('d')]
    print(f"\nTarget columns found: {len(target_cols)}")
    if target_cols:
        print(f"  {target_cols[:4]}")
    else:
        print("❌ No returns_*d columns found!")
        # Show what return-like columns exist
        return_like = [c for c in df.columns if 'return' in c.lower()]
        print(f"Return-like columns: {return_like[:10]}")
        return

    # Check data quality over time
    print("\n" + "=" * 60)
    print("DATA QUALITY BY YEAR")
    print("=" * 60)

    df['year'] = pd.to_datetime(df[date_col]).dt.year

    for year in [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]:
        year_data = df[df['year'] == year]
        if len(year_data) == 0:
            continue

        print(f"\nYear {year}: {len(year_data):,} samples")

        for col in target_cols[:2]:  # Check first 2 horizons
            values = year_data[col].values
            valid = np.isfinite(values)
            nonzero = (values != 0) & valid

            valid_pct = 100 * valid.sum() / len(values) if len(values) > 0 else 0
            nonzero_pct = 100 * nonzero.sum() / len(values) if len(values) > 0 else 0

            print(f"  {col}: {valid_pct:5.1f}% valid, {nonzero_pct:5.1f}% non-zero")

    # Check filtered data
    print("\n" + "=" * 60)
    print("FILTERED DATA ANALYSIS (2018+)")
    print("=" * 60)

    recent = df[pd.to_datetime(df[date_col]) >= '2018-01-01']
    print("\nData from 2018-01-01:")
    print(f"  Samples: {len(recent):,} ({100*len(recent)/len(df):.1f}% of total)")
    print(f"  Date range: {recent[date_col].min()} to {recent[date_col].max()}")

    print("\nTarget quality in 2018+ data:")
    for col in target_cols[:4]:  # Check all 4 horizons
        values = recent[col].values
        valid = np.isfinite(values)
        nonzero = (values != 0) & valid

        valid_count = valid.sum()
        nonzero_count = nonzero.sum()
        valid_pct = 100 * valid_count / len(values)
        nonzero_pct = 100 * nonzero_count / len(values)

        if valid_count > 0:
            valid_values = values[valid]
            mean_val = valid_values.mean()
            std_val = valid_values.std()
            status = "✅" if nonzero_pct > 10 else "⚠️"
            print(f"  {status} {col}: {nonzero_count:,} non-zero ({nonzero_pct:.1f}%), mean={mean_val:.6f}, std={std_val:.6f}")
        else:
            print(f"  ❌ {col}: No valid values!")

    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    if target_cols:
        # Check if any horizon has good data
        good_horizons = []
        for col in target_cols:
            values = recent[col].values
            valid = np.isfinite(values)
            nonzero = (values != 0) & valid
            if 100 * nonzero.sum() / len(values) > 10:
                good_horizons.append(col)

        if good_horizons:
            print("✅ Ready for training with:")
            print("   - Date filter: 2018-01-01 or later")
            print(f"   - Target columns: {good_horizons[:4]}")
            print("   - Use: make -f Makefile.production train-production")
        else:
            print("⚠️ Low quality targets detected")
            print("   - Try filtering to 2020+ data")
            print("   - Or check data pipeline for issues")
    else:
        print("❌ No valid target columns found")
        print("   - Expected: returns_1d, returns_5d, returns_10d, returns_20d")
        print("   - Check dataset building pipeline")

if __name__ == "__main__":
    main()
