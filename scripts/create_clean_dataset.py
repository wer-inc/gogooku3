#!/usr/bin/env python3
"""
Create clean dataset without data leakage
目的: データリークを除去したクリーンなデータセットを作成
"""

from pathlib import Path

import numpy as np
import pandas as pd


def create_clean_dataset():
    """データリークを除去したクリーンなデータセットを作成"""

    print("=" * 60)
    print("🧹 CREATING CLEAN DATASET WITHOUT DATA LEAKAGE")
    print("=" * 60)

    # 1. データ読み込み
    input_path = Path("/home/ubuntu/gogooku3-standalone/output/ml_dataset_latest_full.parquet")
    output_path = Path("/home/ubuntu/gogooku3-standalone/output/ml_dataset_clean.parquet")

    print(f"\n📂 Loading data from: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"✅ Original data shape: {df.shape}")

    # 2. 除外すべき列の特定
    print("\n🔍 Identifying columns to exclude...")

    # ターゲット列（保持する）
    target_cols = ['returns_1d', 'returns_5d', 'returns_10d', 'returns_20d', 'returns_60d', 'returns_120d']

    # データリークを引き起こす列（除外する）
    leakage_columns = [
        # 対数リターン（ターゲットとほぼ同じ）
        'log_returns_1d', 'log_returns_5d', 'log_returns_10d', 'log_returns_20d',

        # target_*という名前の列（明らかにターゲット）
        'target_1d', 'target_5d', 'target_10d', 'target_20d',
        'target_1d_binary', 'target_5d_binary', 'target_10d_binary',

        # feat_ret_*（特徴量リターンだが、ターゲットと重複の可能性）
        'feat_ret_1d', 'feat_ret_5d', 'feat_ret_10d', 'feat_ret_20d',

        # 未来の情報を含む可能性のある列
        'target', 'label', 'y', 'Y'
    ]

    # 実際に存在する列のみを除外リストに含める
    columns_to_exclude = [col for col in leakage_columns if col in df.columns]

    print(f"\n🗑️ Columns to exclude ({len(columns_to_exclude)}):")
    for col in columns_to_exclude:
        print(f"   - {col}")

    # 3. クリーンなデータセット作成
    print("\n🧹 Creating clean dataset...")

    # メタデータ列
    meta_cols = ['Code', 'Date']

    # 保持する列を決定
    all_cols = df.columns.tolist()
    keep_cols = []

    for col in all_cols:
        # メタデータ列は保持
        if col in meta_cols:
            keep_cols.append(col)
        # ターゲット列は保持
        elif col in target_cols:
            keep_cols.append(col)
        # 除外リストにない列は保持
        elif col not in columns_to_exclude:
            keep_cols.append(col)

    # クリーンなデータセット作成
    clean_df = df[keep_cols].copy()

    print(f"\n✅ Clean data shape: {clean_df.shape}")
    print(f"   Original features: {len([c for c in all_cols if c not in meta_cols + target_cols])}")
    print(f"   Clean features: {len([c for c in keep_cols if c not in meta_cols + target_cols])}")
    print(f"   Removed features: {len(columns_to_exclude)}")

    # 4. データ品質チェック
    print("\n🔍 Verifying data quality...")

    # ターゲット列の確認
    for target in target_cols:
        if target in clean_df.columns:
            non_null = clean_df[target].notna().sum()
            print(f"   {target}: {non_null:,} non-null values")

    # 特徴量の型を確認
    numeric_features = [col for col in clean_df.columns
                       if col not in meta_cols + target_cols
                       and clean_df[col].dtype in ['float64', 'float32', 'int64', 'int32']]

    print("\n📊 Feature types:")
    print(f"   Numeric features: {len(numeric_features)}")
    print(f"   Non-numeric features: {len([c for c in clean_df.columns if c not in meta_cols + target_cols]) - len(numeric_features)}")

    # 5. 保存
    print(f"\n💾 Saving clean dataset to: {output_path}")
    clean_df.to_parquet(output_path, index=False)
    print("✅ Clean dataset saved successfully!")

    # 6. 簡単な検証
    print("\n" + "=" * 60)
    print("📊 Quick Validation")
    print("=" * 60)

    # サンプルデータで相関を再チェック
    sample_df = clean_df.sample(min(10000, len(clean_df)))
    numeric_features_sample = [col for col in numeric_features[:50] if col in sample_df.columns]

    print("\n🔍 Checking for remaining high correlations with targets...")
    high_corr_found = False

    for target in ['returns_1d', 'returns_5d']:
        if target not in sample_df.columns:
            continue

        target_vals = sample_df[target].fillna(0)
        max_corr = 0
        max_corr_feature = None

        for feat in numeric_features_sample:
            feat_vals = sample_df[feat].fillna(0)
            try:
                corr = np.corrcoef(target_vals, feat_vals)[0, 1]
                if abs(corr) > abs(max_corr):
                    max_corr = corr
                    max_corr_feature = feat
                if abs(corr) > 0.9:
                    print(f"   ⚠️ High correlation: {target} vs {feat} = {corr:.4f}")
                    high_corr_found = True
            except:
                pass

        if max_corr_feature:
            print(f"   {target}: max correlation = {max_corr:.4f} (with {max_corr_feature})")

    if not high_corr_found:
        print("\n✅ No obvious data leakage remaining!")
    else:
        print("\n⚠️ Some high correlations remain - manual review recommended")

    # 7. 次のステップ
    print("\n" + "=" * 60)
    print("🚀 Next Steps")
    print("=" * 60)
    print("\n1. Convert clean dataset to ATFT format:")
    print("   python scripts/convert_to_atft_data.py --input output/ml_dataset_clean.parquet")
    print("\n2. Re-run baseline test to verify:")
    print("   python scripts/test_baseline_rankic.py --use-clean")
    print("\n3. Train ATFT with clean data:")
    print("   make train-clean")

    return output_path

if __name__ == "__main__":
    output_path = create_clean_dataset()
    print(f"\n✨ Clean dataset ready at: {output_path}")
