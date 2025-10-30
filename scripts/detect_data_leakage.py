#!/usr/bin/env python3
"""
Critical: Detect and fix data leakage
目的: ターゲット変数が特徴量に含まれていないか確認
"""

import numpy as np
import pandas as pd


def check_normalization_stats(df, feature_cols, verbose=True):
    """
    Check normalization statistics validity

    Verifies that feature statistics (mean, std) are reasonable
    and not indicative of data quality issues.

    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
        verbose: Print detailed info

    Returns:
        bool: True if stats look reasonable, False if warnings detected
    """
    if verbose:
        print("\n" + "=" * 60)
        print("5️⃣ Checking normalization statistics...")
        print("=" * 60)

    # Compute statistics for numeric features
    numeric_features = [
        col for col in feature_cols
        if df[col].dtype in ['float64', 'float32', 'int64', 'int32']
    ]

    if len(numeric_features) == 0:
        if verbose:
            print("⚠️ No numeric features found")
        return False

    # Sample for efficiency (100k rows)
    sample_df = df.sample(min(100000, len(df)))

    means = sample_df[numeric_features].mean()
    stds = sample_df[numeric_features].std()

    if verbose:
        print(f"\n📊 Statistics from {len(sample_df)} samples:")
        print(f"   Mean range: [{means.min():.4f}, {means.max():.4f}]")
        print(f"   Std range:  [{stds.min():.6f}, {stds.max():.4f}]")

    # Detect anomalies (conservative thresholds)
    warnings = []

    # 1. Extremely large absolute means (> 1000)
    extreme_means = [c for c in numeric_features if abs(means[c]) > 1000]
    if extreme_means:
        warnings.append(
            f"⚠️  {len(extreme_means)} features with |mean| > 1000 (may need scaling)"
        )
        if verbose:
            print(f"\n   Extreme means: {extreme_means[:5]}")

    # 2. Near-constant features (std < 1e-8)
    tiny_stds = [c for c in numeric_features if stds[c] < 1e-8]
    if tiny_stds:
        warnings.append(
            f"⚠️  {len(tiny_stds)} features with std < 1e-8 (near-constant, consider removal)"
        )
        if verbose:
            print(f"   Near-constant features: {tiny_stds[:5]}")

    # 3. Extremely large std (> 10000)
    huge_stds = [c for c in numeric_features if stds[c] > 10000]
    if huge_stds:
        warnings.append(
            f"⚠️  {len(huge_stds)} features with std > 10000 (outliers or scale issue)"
        )
        if verbose:
            print(f"   Huge std features: {huge_stds[:5]}")

    # 4. NaN/Inf check
    nan_counts = sample_df[numeric_features].isna().sum()
    high_nan_features = [c for c in numeric_features if nan_counts[c] > len(sample_df) * 0.5]
    if high_nan_features:
        warnings.append(
            f"⚠️  {len(high_nan_features)} features with >50% missing values"
        )
        if verbose:
            print(f"   High missing features: {high_nan_features[:5]}")

    # Print summary
    if verbose:
        print("\n" + "-" * 60)
        if warnings:
            for w in warnings:
                print(w)
            print("\n💡 Recommendation: Review these features before training")
            return False
        else:
            print("✅ Normalization statistics look reasonable")
            return True

    return len(warnings) == 0


def detect_data_leakage():
    """データリークを検出"""

    print("=" * 60)
    print("🔍 DATA LEAKAGE DETECTION")
    print("=" * 60)

    # データ読み込み
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else 'output/ml_dataset_latest_full.parquet'
    print(f"\n📂 Loading data: {data_path}")
    df = pd.read_parquet(data_path)
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

    # 5. Normalization statistics check (newly added v3.0)
    check_normalization_stats(df, feature_cols, verbose=True)

    # 6. 最終診断
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
