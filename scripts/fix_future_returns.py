#!/usr/bin/env python3
"""
Fix returns to be FUTURE returns instead of PAST returns
目的: returns_*dを正しい未来リターンに修正

- デフォルト入力は Step1 の成果物に合わせて `output/ml_dataset_latest_full.parquet`
- Ultra‑clean を使う場合は `--input output/ml_dataset_ultra_clean.parquet` を指定
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def fix_future_returns(input_path: Path, output_path: Path) -> Path:
    """過去リターンを未来リターンに修正"""

    print("=" * 60)
    print("🔧 FIXING RETURNS TO BE FUTURE RETURNS")
    print("=" * 60)

    print(f"\n📂 Loading data from: {input_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"✅ Original data shape: {df.shape}")

    # Sort by Code and Date
    df = df.sort_values(['Code', 'Date']).reset_index(drop=True)

    print("\n🔧 Calculating FUTURE returns...")

    # Group by Code (stock) and calculate future returns
    def calculate_future_returns(group):
        """Calculate future returns for a single stock"""
        group = group.sort_values('Date').copy()

        # Calculate future returns using Close price
        # shift(-n) means we look n periods into the future
        group['future_returns_1d'] = group['Close'].pct_change(1).shift(-1)  # Tomorrow's return
        group['future_returns_5d'] = group['Close'].pct_change(5).shift(-5)  # 5 days ahead
        group['future_returns_10d'] = group['Close'].pct_change(10).shift(-10)  # 10 days ahead
        group['future_returns_20d'] = group['Close'].pct_change(20).shift(-20)  # 20 days ahead
        group['future_returns_60d'] = group['Close'].pct_change(60).shift(-60)  # 60 days ahead
        group['future_returns_120d'] = group['Close'].pct_change(120).shift(-120)  # 120 days ahead

        return group

    # Apply to each stock
    print("📊 Processing each stock...")
    df = df.groupby('Code', group_keys=False).apply(calculate_future_returns)

    # Replace old returns columns with new future returns
    print("\n🔄 Replacing old returns with future returns...")

    # Drop old returns columns if they exist
    old_returns_cols = ['returns_1d', 'returns_5d', 'returns_10d', 'returns_20d', 'returns_60d', 'returns_120d']
    existing_old_cols = [col for col in old_returns_cols if col in df.columns]
    if existing_old_cols:
        df = df.drop(columns=existing_old_cols)
        print(f"   Dropped {len(existing_old_cols)} old returns columns")

    # Rename future returns to standard names
    rename_map = {
        'future_returns_1d': 'returns_1d',
        'future_returns_5d': 'returns_5d',
        'future_returns_10d': 'returns_10d',
        'future_returns_20d': 'returns_20d',
        'future_returns_60d': 'returns_60d',
        'future_returns_120d': 'returns_120d'
    }
    df = df.rename(columns=rename_map)
    print(f"   Renamed {len(rename_map)} future returns columns")

    # Remove rows with NaN future returns (last rows of each stock)
    print("\n🧹 Removing rows without future returns...")
    before_count = len(df)

    # Only keep rows where at least returns_1d is not NaN
    df = df[df['returns_1d'].notna()].copy()

    after_count = len(df)
    print(f"   Removed {before_count - after_count:,} rows with NaN future returns")
    print(f"   Final dataset: {after_count:,} rows")

    # Validate the fix
    print("\n🔍 Validating the fix...")

    # Take a sample stock for validation
    sample_code = df['Code'].iloc[0]
    sample_df = df[df['Code'] == sample_code].head(20)

    # Check that returns are now future returns
    validation_passed = True
    for i in range(5, 10):  # Check a few rows
        current_close = sample_df.iloc[i]['Close']
        next_close = sample_df.iloc[i+1]['Close'] if i+1 < len(sample_df) else np.nan

        if not pd.isna(next_close):
            expected_return = (next_close - current_close) / current_close
            actual_return = sample_df.iloc[i]['returns_1d']

            if not pd.isna(actual_return):
                diff = abs(expected_return - actual_return)
                if diff > 0.0001:  # Allow small floating point differences
                    print(f"   ⚠️ Row {i}: Expected {expected_return:.4f}, Got {actual_return:.4f}")
                    validation_passed = False

    if validation_passed:
        print("   ✅ Validation passed: returns_1d is now FUTURE returns!")
    else:
        print("   ⚠️ Some discrepancies found, but continuing...")

    # Save the corrected dataset
    print(f"\n💾 Saving corrected dataset to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print("✅ Dataset with future returns saved successfully!")

    # Print summary statistics
    print("\n📊 Summary of future returns:")
    for col in ['returns_1d', 'returns_5d', 'returns_10d', 'returns_20d']:
        if col in df.columns:
            mean_ret = df[col].mean()
            std_ret = df[col].std()
            print(f"   {col}: mean={mean_ret:.6f}, std={std_ret:.6f}")

    print("\n" + "=" * 60)
    print("🚀 Next Steps")
    print("=" * 60)
    print("\n1. Test with baseline model (expect MUCH lower RankIC):")
    print("   python scripts/test_baseline_rankic.py output/ml_dataset_future_returns.parquet")
    print("\n2. Convert to ATFT format:")
    print("   python scripts/integrated_ml_training_pipeline.py \\")
    print("     --data-path output/ml_dataset_future_returns.parquet \\")
    print("     --only-convert")
    print("\n3. Train ATFT with correct future returns:")
    print("   make train-future-returns")

    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert returns_*d to FUTURE returns")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/ml_dataset_latest_full.parquet"),
        help="Input ML dataset parquet (default: output/ml_dataset_latest_full.parquet)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/ml_dataset_future_returns.parquet"),
        help="Output parquet path (default: output/ml_dataset_future_returns.parquet)",
    )
    args = parser.parse_args()

    out = fix_future_returns(args.input, args.output)
    print(f"\n✨ Dataset with future returns ready at: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
