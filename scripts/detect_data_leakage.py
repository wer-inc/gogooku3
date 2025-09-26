#!/usr/bin/env python3
"""
Critical: Detect and fix data leakage
目的: ターゲット変数が特徴量に含まれていないか確認
"""

import pandas as pd
import numpy as np

def detect_data_leakage():
    """データリークを検出"""

    print("=" * 60)
    print("🔍 DATA LEAKAGE DETECTION")
    print("=" * 60)

    # データ読み込み
    print("\n📂 Loading data...")
    df = pd.read_parquet('/home/ubuntu/gogooku3-standalone/output/ml_dataset_latest_full.parquet')
    print(f"✅ Data shape: {df.shape}")

    # ターゲット変数
    target_cols = ['returns_1d', 'returns_5d', 'returns_10d', 'returns_20d', 'returns_60d', 'returns_120d']
    print(f"\n🎯 Target columns: {target_cols}")

    # 特徴量列
    feature_cols = [col for col in df.columns if col not in ['Date', 'Code'] + target_cols]
    print(f"📊 Feature columns: {len(feature_cols)}")

    # 1. ターゲットと同じ名前のパターンを持つ特徴量を探す
    print("\n" + "=" * 60)
    print("1️⃣ Checking for suspicious feature names...")
    print("=" * 60)

    suspicious_features = []
    for col in feature_cols:
        # returnやretなど疑わしいパターン
        if any(pattern in col.lower() for pattern in ['return', 'ret', 'target']):
            suspicious_features.append(col)

    if suspicious_features:
        print(f"\n⚠️ Found {len(suspicious_features)} suspicious features:")
        for col in suspicious_features[:20]:  # 最初の20個を表示
            print(f"   - {col}")
    else:
        print("✅ No suspicious feature names found")

    # 2. ターゲットと完全相関する特徴量を探す
    print("\n" + "=" * 60)
    print("2️⃣ Checking for perfect correlation with targets...")
    print("=" * 60)

    # サンプルデータで相関を計算（メモリ節約）
    sample_df = df.sample(min(100000, len(df)))
    numeric_features = [col for col in feature_cols if sample_df[col].dtype in ['float64', 'float32', 'int64', 'int32']]

    leaked_features = {}
    for target_col in target_cols:
        if target_col not in sample_df.columns:
            continue

        print(f"\n🎯 Checking {target_col}...")

        high_corr_features = []
        target_values = sample_df[target_col].fillna(0)

        for feature_col in numeric_features:
            try:
                feature_values = sample_df[feature_col].fillna(0)
                # 相関係数を計算
                corr = np.corrcoef(target_values, feature_values)[0, 1]

                # 相関が異常に高い（0.99以上）場合
                if abs(corr) > 0.99:
                    high_corr_features.append((feature_col, corr))
            except:
                continue

        if high_corr_features:
            leaked_features[target_col] = high_corr_features
            print(f"   🔴 CRITICAL: Found {len(high_corr_features)} features with correlation > 0.99!")
            for feat, corr in high_corr_features[:5]:  # 最初の5個を表示
                print(f"      - {feat}: correlation = {corr:.4f}")

    # 3. 同じ値を持つ列のペアを探す
    print("\n" + "=" * 60)
    print("3️⃣ Checking for duplicate/identical columns...")
    print("=" * 60)

    # 各ターゲットと特徴量の値が完全一致するか確認
    for target_col in target_cols[:2]:  # 最初の2つのターゲットのみチェック（時間短縮）
        if target_col not in sample_df.columns:
            continue

        target_vals = sample_df[target_col].values
        for feature_col in numeric_features[:50]:  # 最初の50特徴量をチェック
            feature_vals = sample_df[feature_col].values

            # NaNを除いて比較
            mask = ~(np.isnan(target_vals) | np.isnan(feature_vals))
            if mask.sum() > 0:
                if np.array_equal(target_vals[mask], feature_vals[mask]):
                    print(f"   🔴 IDENTICAL: {target_col} == {feature_col}")

    # 4. 特徴量間の相関行列を確認
    print("\n" + "=" * 60)
    print("4️⃣ Feature correlation matrix check...")
    print("=" * 60)

    # returnやretを含む特徴量の相関を確認
    ret_features = [col for col in numeric_features if 'ret' in col.lower() or 'return' in col.lower()]

    if len(ret_features) > 0:
        print(f"\nFound {len(ret_features)} features with 'ret' or 'return' in name")
        print("Sample features:")
        for feat in ret_features[:10]:
            print(f"   - {feat}")

        # これらとターゲットの相関を確認
        if len(ret_features) > 0 and len(target_cols) > 0:
            sample_size = min(10000, len(df))
            sample_df = df.sample(sample_size)

            for target_col in target_cols[:2]:  # 最初の2つのターゲット
                if target_col in sample_df.columns:
                    target_vals = sample_df[target_col].fillna(0)

                    print(f"\n🎯 Correlation with {target_col}:")
                    high_corr_count = 0

                    for feat in ret_features[:20]:  # 最初の20個
                        if feat in sample_df.columns:
                            feat_vals = sample_df[feat].fillna(0)
                            try:
                                corr = np.corrcoef(target_vals, feat_vals)[0, 1]
                                if abs(corr) > 0.5:  # 相関0.5以上を表示
                                    print(f"   {feat}: {corr:.4f}")
                                    if abs(corr) > 0.9:
                                        high_corr_count += 1
                            except:
                                pass

                    if high_corr_count > 0:
                        print(f"   🔴 WARNING: {high_corr_count} features have correlation > 0.9!")

    # 5. 最終診断
    print("\n" + "=" * 60)
    print("🏁 DIAGNOSIS")
    print("=" * 60)

    if leaked_features:
        print("\n🔴 CRITICAL DATA LEAKAGE DETECTED!")
        print("\n原因:")
        print("  1. ターゲット変数と同じ値を持つ特徴量が存在")
        print("  2. returns_*d が特徴量として使用されている可能性")
        print("\n対策:")
        print("  1. データセット生成時にターゲット列を除外")
        print("  2. ATFT用データ変換時にターゲット列を分離")
        print("  3. 特徴量選択を見直し")
        print("\n影響:")
        print("  → これが原因でATFT-GAT-FANが学習できていない")
        print("  → Val RankIC = 0.0719 の固定値になっている")
    else:
        print("\n🟢 No obvious data leakage detected")
        print("But the high baseline RankIC (0.99+) suggests hidden leakage")
        print("Recommend manual review of feature engineering pipeline")

if __name__ == "__main__":
    detect_data_leakage()