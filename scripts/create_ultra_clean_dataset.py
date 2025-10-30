#!/usr/bin/env python3
"""
Create ultra-clean dataset with correlation-based leakage detection
目的: 相関ベースでデータリークを完全に除去
"""

from pathlib import Path

import numpy as np
import pandas as pd


def create_ultra_clean_dataset():
    """相関ベースでデータリークを完全に除去したデータセットを作成"""

    print("=" * 60)
    print("🧹 CREATING ULTRA-CLEAN DATASET")
    print("=" * 60)

    # 1. データ読み込み
    input_path = Path("/home/ubuntu/gogooku3-standalone/output/ml_dataset_latest_full.parquet")
    output_path = Path("/home/ubuntu/gogooku3-standalone/output/ml_dataset_ultra_clean.parquet")

    print(f"\n📂 Loading data from: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"✅ Original data shape: {df.shape}")

    # 2. ターゲット列とメタデータ列の特定
    target_cols = ['returns_1d', 'returns_5d', 'returns_10d', 'returns_20d', 'returns_60d', 'returns_120d']
    meta_cols = ['Code', 'Date']

    # 3. 初期の特徴量列
    feature_cols = [col for col in df.columns if col not in meta_cols + target_cols]
    print(f"\n📊 Initial features: {len(feature_cols)}")

    # 4. 明示的に除外すべき列（名前ベース）
    print("\n🔍 Phase 1: Name-based exclusion...")

    name_based_exclusions = []
    for col in feature_cols:
        col_lower = col.lower()
        # リターン関連
        if any(x in col_lower for x in ['return', 'ret_', '_ret', 'log_ret', 'feat_ret']):
            name_based_exclusions.append(col)
        # ターゲット関連
        elif any(x in col_lower for x in ['target', 'label', 'y_', '_y']):
            name_based_exclusions.append(col)
        # アルファ（超過リターン）
        elif 'alpha' in col_lower and any(x in col_lower for x in ['1d', '5d', '10d', '20d']):
            name_based_exclusions.append(col)
        # 相対強度（これもリターンの変形）
        elif 'rel_strength' in col_lower and any(x in col_lower for x in ['1d', '5d', '10d', '20d']):
            name_based_exclusions.append(col)

    print(f"   Excluded by name: {len(name_based_exclusions)} features")

    # 5. 相関ベースの除外（厳格な基準）
    print("\n🔍 Phase 2: Correlation-based exclusion...")

    # サンプリングして相関を計算
    sample_size = min(100000, len(df))
    sample_df = df.sample(sample_size, random_state=42)

    correlation_exclusions = []
    CORR_THRESHOLD = 0.8  # 80%以上の相関は除外

    for target_col in target_cols:
        if target_col not in sample_df.columns:
            continue

        target_vals = sample_df[target_col].fillna(0).values

        for col in feature_cols:
            if col in name_based_exclusions:  # 既に除外済みはスキップ
                continue

            if sample_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                try:
                    col_vals = sample_df[col].fillna(0).values
                    corr = np.corrcoef(target_vals, col_vals)[0, 1]

                    if abs(corr) > CORR_THRESHOLD:
                        if col not in correlation_exclusions:
                            correlation_exclusions.append(col)
                            print(f"   🔴 High correlation: {col} with {target_col} = {corr:.4f}")
                except:
                    pass

    print(f"   Excluded by correlation: {len(correlation_exclusions)} additional features")

    # 6. 全除外リストを作成
    all_exclusions = list(set(name_based_exclusions + correlation_exclusions))
    print(f"\n🗑️ Total exclusions: {len(all_exclusions)} features")

    # 7. クリーンなデータセット作成
    print("\n🧹 Creating ultra-clean dataset...")

    keep_cols = meta_cols + target_cols
    for col in feature_cols:
        if col not in all_exclusions:
            keep_cols.append(col)

    clean_df = df[keep_cols].copy()

    print(f"\n✅ Ultra-clean data shape: {clean_df.shape}")
    print(f"   Original features: {len(feature_cols)}")
    print(f"   Clean features: {len(keep_cols) - len(meta_cols) - len(target_cols)}")
    print(f"   Removed features: {len(all_exclusions)}")

    # 8. 最終検証（相関チェック）
    print("\n🔍 Final validation...")

    # クリーンデータで再度相関をチェック
    sample_clean = clean_df.sample(min(50000, len(clean_df)), random_state=42)
    clean_feature_cols = [col for col in clean_df.columns if col not in meta_cols + target_cols]

    max_corr_found = 0
    max_corr_feature = None
    max_corr_target = None

    for target_col in target_cols[:2]:  # 最初の2つのターゲットで検証
        if target_col not in sample_clean.columns:
            continue

        target_vals = sample_clean[target_col].fillna(0).values

        for col in clean_feature_cols[:50]:  # 最初の50特徴量で検証
            if sample_clean[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                try:
                    col_vals = sample_clean[col].fillna(0).values
                    corr = np.corrcoef(target_vals, col_vals)[0, 1]

                    if abs(corr) > abs(max_corr_found):
                        max_corr_found = corr
                        max_corr_feature = col
                        max_corr_target = target_col
                except:
                    pass

    print("\n📊 Maximum correlation found:")
    print(f"   {max_corr_feature} with {max_corr_target}: {max_corr_found:.4f}")

    if abs(max_corr_found) > 0.8:
        print("   ⚠️ WARNING: Still some high correlation remaining!")
    elif abs(max_corr_found) > 0.5:
        print("   🟡 Moderate correlation - acceptable for most ML models")
    else:
        print("   ✅ Low correlation - excellent for training!")

    # 9. 保存
    print(f"\n💾 Saving ultra-clean dataset to: {output_path}")
    clean_df.to_parquet(output_path, index=False)
    print("✅ Ultra-clean dataset saved successfully!")

    # 10. 使用可能な特徴量のサンプル表示
    print("\n📋 Sample of remaining features:")
    remaining_features = clean_feature_cols[:20]
    for i, feat in enumerate(remaining_features, 1):
        print(f"   {i:2}. {feat}")

    print("\n" + "=" * 60)
    print("🚀 Next Steps")
    print("=" * 60)
    print("\n1. Test with baseline model:")
    print("   python scripts/test_baseline_rankic.py --data output/ml_dataset_ultra_clean.parquet")
    print("\n2. Convert to ATFT format:")
    print("   python scripts/convert_to_atft_data.py --input output/ml_dataset_ultra_clean.parquet")
    print("\n3. Train ATFT with ultra-clean data:")
    print("   make train-ultra-clean")

    return output_path

if __name__ == "__main__":
    output_path = create_ultra_clean_dataset()
    print(f"\n✨ Ultra-clean dataset ready at: {output_path}")
