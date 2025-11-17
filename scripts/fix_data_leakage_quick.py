#!/usr/bin/env python3
"""
Quick Fix: Remove leaked features from dataset

This script removes features that have data leakage (high correlation with targets)
and creates a clean dataset for immediate training.

Usage:
    python scripts/fix_data_leakage_quick.py \\
        --input output/ml_dataset_latest_full.parquet \\
        --output output/ml_dataset_no_leak.parquet
"""

import argparse
from pathlib import Path

import polars as pl

# Features identified with data leakage (correlation > 0.7 with targets)
LEAKED_FEATURES = [
    # Perfect leakage (correlation = 1.0)
    "rank_ret_1d",

    # Severe leakage (correlation > 0.9)
    "ret_1d_rank_in_sec",

    # High leakage (correlation > 0.7)
    "ret_1d_in_sec_z",
    "ret_1d_vs_sec",
    "log_returns_1d",
    "ret_5d_in_sec_z",
    "ret_5d_vs_sec",
    "log_returns_5d",
    "z_in_sec_returns_5d",
    "ret_10d_in_sec_z",
    "ret_10d_vs_sec",
    "log_returns_10d",

    # Other suspicious features (containing 'ret' or 'return')
    "feat_ret_1d",
    "feat_ret_5d",
    "feat_ret_10d",
    "feat_ret_20d",
    "log_returns_20d",
    "mkt_ret_1d",
    "mkt_ret_5d",
    "mkt_ret_10d",
    "mkt_ret_20d",
    "mkt_ret_1d_z",
    "sec_ret_1d_eq",
    "sec_ret_1d_mcap",
    "sec_ret_5d_eq",
    "sec_ret_5d_mcap",
    "sec_ret_20d_eq",
    "sec_ret_20d_mcap",
]


def remove_leaked_features(input_path: str, output_path: str, dry_run: bool = False):
    """
    Remove leaked features from dataset

    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file
        dry_run: If True, only show what would be removed (don't write)
    """
    print("=" * 80)
    print("🔧 DATA LEAKAGE QUICK FIX")
    print("=" * 80)

    # Load dataset
    print(f"\n📂 Loading dataset: {input_path}")
    df = pl.read_parquet(input_path)
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {len(df.columns)}")

    # Identify which leaked features exist
    existing_leaked = [col for col in LEAKED_FEATURES if col in df.columns]
    missing_leaked = [col for col in LEAKED_FEATURES if col not in df.columns]

    print(f"\n🎯 Leaked features to remove: {len(existing_leaked)}/{len(LEAKED_FEATURES)}")
    if existing_leaked:
        print("\n   Features found in dataset:")
        for feat in existing_leaked:
            print(f"      - {feat}")

    if missing_leaked:
        print(f"\n   ℹ️  {len(missing_leaked)} features not in dataset (already removed or never generated)")

    if dry_run:
        print("\n🔍 DRY RUN MODE - No changes will be made")
        print(f"\n   New shape would be: ({df.shape[0]}, {df.shape[1] - len(existing_leaked)})")
        return

    if not existing_leaked:
        print("\n✅ No leaked features found in dataset")
        print("   Dataset appears clean - no action needed")
        return

    # Remove leaked features
    print(f"\n🗑️  Removing {len(existing_leaked)} leaked features...")
    df_clean = df.drop(existing_leaked)

    print(f"   New shape: {df_clean.shape}")
    print(f"   Removed columns: {len(existing_leaked)}")
    print(f"   Remaining columns: {len(df_clean.columns)}")

    # Verify no returns_* columns in features (should only be in targets)
    feature_cols = [col for col in df_clean.columns if col not in ['Date', 'Code', 'returns_1d', 'returns_5d', 'returns_10d', 'returns_20d', 'returns_60d', 'returns_120d']]
    suspicious_remaining = [col for col in feature_cols if 'ret' in col.lower() or 'return' in col.lower()]

    if suspicious_remaining:
        print(f"\n⚠️  WARNING: {len(suspicious_remaining)} suspicious features still remain:")
        for feat in suspicious_remaining[:10]:
            print(f"      - {feat}")
        if len(suspicious_remaining) > 10:
            print(f"      ... and {len(suspicious_remaining) - 10} more")
        print("\n   💡 Consider adding these to LEAKED_FEATURES if they show high correlation")
    else:
        print("\n✅ No suspicious feature names remain")

    # Save clean dataset
    print(f"\n💾 Saving clean dataset: {output_path}")
    df_clean.write_parquet(output_path)

    # Create symlink for easy access
    symlink_path = Path(output_path).parent / "ml_dataset_latest_no_leak.parquet"
    if symlink_path.exists() or symlink_path.is_symlink():
        symlink_path.unlink()
    # Use relative path for symlink
    import os
    output_path_obj = Path(output_path)
    if symlink_path.parent == output_path_obj.parent:
        symlink_path.symlink_to(output_path_obj.name)
    else:
        symlink_path.symlink_to(os.path.relpath(output_path_obj, start=symlink_path.parent))
    print(f"   Symlink created: {symlink_path}")

    print("\n" + "=" * 80)
    print("✅ QUICK FIX COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Re-run leakage detection:")
    print(f"   python scripts/detect_data_leakage.py --data {output_path}")
    print("\n2. If clean, start Phase 1 short test:")
    print("   python scripts/integrated_ml_training_pipeline.py \\")
    print(f"       --data-path {output_path} \\")
    print("       --max-epochs 5 \\")
    print("       --batch-size 1024")
    print("\n3. Expected: Val Sharpe will be MUCH LOWER than before (0.01-0.05)")
    print("   This is NORMAL - previous metrics were inflated by leakage")


def main():
    parser = argparse.ArgumentParser(
        description="Quick fix: Remove leaked features from dataset"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="output/ml_dataset_latest_full.parquet",
        help="Input dataset path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output dataset path (default: auto-generated with timestamp)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without making changes"
    )

    args = parser.parse_args()

    # Auto-generate output path if not provided
    if args.output is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"ml_dataset_no_leak_{timestamp}.parquet")

    remove_leaked_features(args.input, args.output, args.dry_run)


if __name__ == "__main__":
    main()
