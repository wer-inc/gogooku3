#!/usr/bin/env python3
"""
Debug script to identify actual 182 features from training data
"""

import sys
from pathlib import Path

import polars as pl


def main():
    # Find training data
    data_paths = [
        "output/ml_dataset_20251001_154821_full.parquet",
        "output/ml_dataset_latest_full.parquet",
        "output/atft_data/train/part_0000.parquet",
        "output/atft_data/train/part_0001.parquet",
    ]

    data_file = None
    for path in data_paths:
        if Path(path).exists() and not Path(path).is_symlink():
            data_file = path
            break

    if not data_file:
        print("❌ No training data found", file=sys.stderr)
        sys.exit(1)

    print(f"📂 Loading data from: {data_file}")
    df = pl.read_parquet(data_file)

    # Exclude metadata and target columns
    exclude = {
        'Date', 'date',
        'Code', 'code',
        'target',
        'target_1d', 'target_5d', 'target_10d', 'target_20d',
        'feat_ret_1d', 'feat_ret_5d', 'feat_ret_10d', 'feat_ret_20d',  # May be targets
    }

    all_cols = df.columns
    features = [c for c in all_cols if c not in exclude]

    print("\n📊 FEATURE ANALYSIS")
    print("=" * 80)
    print(f"Total columns in parquet: {len(all_cols)}")
    print(f"Excluded columns: {len(exclude & set(all_cols))}")
    print(f"Feature columns: {len(features)}")
    print()

    # Categorize by prefix/pattern
    categories = {
        'price_volume': [],
        'returns': [],
        'sma': [],
        'ema': [],
        'volatility': [],
        'rsi': [],
        'macd': [],
        'bb': [],
        'adx': [],
        'mkt': [],
        'flow': [],
        'margin': [],
        'stmt': [],
        'others': [],
    }

    for feat in sorted(features):
        feat_lower = feat.lower()

        if feat in ['Open', 'High', 'Low', 'Close', 'Volume', 'TurnoverValue', 'dollar_volume']:
            categories['price_volume'].append(feat)
        elif feat.startswith('returns_') or feat.startswith('log_returns_'):
            categories['returns'].append(feat)
        elif feat.startswith('sma_'):
            categories['sma'].append(feat)
        elif feat.startswith('ema_'):
            categories['ema'].append(feat)
        elif 'volatility' in feat_lower or 'vol_' in feat_lower:
            categories['volatility'].append(feat)
        elif feat.startswith('rsi_'):
            categories['rsi'].append(feat)
        elif feat.startswith('macd'):
            categories['macd'].append(feat)
        elif feat.startswith('bb_'):
            categories['bb'].append(feat)
        elif feat.startswith('adx'):
            categories['adx'].append(feat)
        elif feat.startswith('mkt_'):
            categories['mkt'].append(feat)
        elif feat.startswith('flow_'):
            categories['flow'].append(feat)
        elif feat.startswith('margin_') or 'Margin' in feat:
            categories['margin'].append(feat)
        elif feat.startswith('stmt_'):
            categories['stmt'].append(feat)
        else:
            categories['others'].append(feat)

    print("📂 Features by Category:")
    print("=" * 80)
    total_categorized = 0
    for cat, feats in categories.items():
        if feats:
            print(f"\n{cat.upper():20s} ({len(feats):3d} features):")
            for i, f in enumerate(feats, 1):
                print(f"  {i:3d}. {f}")
            total_categorized += len(feats)

    print()
    print("=" * 80)
    print(f"Total categorized: {total_categorized}")
    print()

    # Save feature list
    output_file = "configs/atft/actual_182_features.txt"
    with open(output_file, 'w') as f:
        f.write("# Actual 182 features from training data\n")
        f.write(f"# Generated from: {data_file}\n\n")
        for feat in sorted(features):
            f.write(feat + '\n')

    print(f"✅ Feature list saved to: {output_file}")
    print()

    # Check if count matches
    if len(features) == 182:
        print("✅ MATCH: 182 features confirmed")
    else:
        print(f"⚠️  MISMATCH: Expected 182, got {len(features)}")

        # Show excluded columns
        print("\nExcluded columns found in data:")
        for col in sorted(exclude & set(all_cols)):
            print(f"  - {col}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
