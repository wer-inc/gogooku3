#!/usr/bin/env python3
"""
Phase 0: Test baseline model to verify data predictability
目的: シンプルなモデルでRankICが計算可能か確認
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

def calculate_rankic(y_true, y_pred):
    """RankIC (Spearman correlation) を計算"""
    # NaNを除去
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 10:
        return np.nan

    corr, _ = spearmanr(y_true[mask], y_pred[mask])
    return corr

def test_baseline_models(data_path=None):
    """ベースラインモデルでRankICをテスト"""

    print("=" * 60)
    print("🧪 Baseline Model RankIC Test")
    print("=" * 60)

    # データ読み込み
    print("\n📂 Loading data...")
    if data_path:
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_parquet('/home/ubuntu/gogooku3-standalone/output/ml_dataset_latest_full.parquet')
    print(f"✅ Data shape: {df.shape}")

    # 特徴量とターゲットを分離
    feature_cols = [col for col in df.columns if not col.startswith('returns_') and col not in ['Date', 'Code']]
    target_cols = ['returns_1d', 'returns_5d', 'returns_10d', 'returns_20d']

    print(f"\n📊 Features: {len(feature_cols)}")
    print(f"🎯 Targets: {target_cols}")

    # 日付でソート
    df = df.sort_values('Date')

    # Train/Val分割 (時系列を考慮)
    split_date = df['Date'].quantile(0.8)
    train_df = df[df['Date'] < split_date]
    val_df = df[df['Date'] >= split_date]

    print(f"\n📅 Train: {train_df['Date'].min()} to {train_df['Date'].max()} ({len(train_df):,} samples)")
    print(f"📅 Val: {val_df['Date'].min()} to {val_df['Date'].max()} ({len(val_df):,} samples)")

    # 欠損値処理
    print("\n🔧 Preparing data...")
    # 数値列のみを選択
    numeric_features = [col for col in feature_cols if train_df[col].dtype in ['float64', 'float32', 'int64', 'int32']]
    print(f"   Using {len(numeric_features)} numeric features (from {len(feature_cols)} total)")

    X_train = train_df[numeric_features].fillna(0)
    X_val = val_df[numeric_features].fillna(0)

    # 異常値クリッピング
    X_train = X_train.clip(-10, 10)
    X_val = X_val.clip(-10, 10)

    print("✅ Data prepared")

    # 各ターゲットに対してテスト
    results = {}

    for target_col in target_cols:
        print(f"\n" + "=" * 60)
        print(f"🎯 Testing: {target_col}")
        print("=" * 60)

        y_train = train_df[target_col].fillna(0)
        y_val = val_df[target_col].fillna(0)

        # 1. ランダム予測（ベースライン）
        print("\n1️⃣ Random Baseline:")
        y_pred_random = np.random.randn(len(y_val)) * y_train.std() + y_train.mean()
        rankic_random = calculate_rankic(y_val.values, y_pred_random)
        print(f"   RankIC: {rankic_random:.6f} (expected ~0)")
        results[f"{target_col}_random"] = rankic_random

        # 2. 平均予測
        print("\n2️⃣ Mean Baseline:")
        y_pred_mean = np.full(len(y_val), y_train.mean())
        rankic_mean = calculate_rankic(y_val.values, y_pred_mean)
        print(f"   RankIC: {rankic_mean:.6f} (expected ~0)")
        results[f"{target_col}_mean"] = rankic_mean

        # 3. 線形回帰
        print("\n3️⃣ Linear Regression:")
        lr = LinearRegression()

        # サンプルサイズを制限（メモリ対策）
        sample_size = min(100000, len(X_train))
        indices = np.random.choice(len(X_train), sample_size, replace=False)

        lr.fit(X_train.iloc[indices], y_train.iloc[indices])
        y_pred_lr = lr.predict(X_val)
        rankic_lr = calculate_rankic(y_val.values, y_pred_lr)
        print(f"   RankIC: {rankic_lr:.6f}")
        results[f"{target_col}_lr"] = rankic_lr

        # 4. LightGBM (シンプル設定)
        print("\n4️⃣ LightGBM:")
        lgb = LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        )

        lgb.fit(
            X_train.iloc[indices],
            y_train.iloc[indices],
            eval_set=[(X_val.iloc[:10000], y_val.iloc[:10000])],
            callbacks=[lambda x: None]  # Suppress output
        )
        y_pred_lgb = lgb.predict(X_val)
        rankic_lgb = calculate_rankic(y_val.values, y_pred_lgb)
        print(f"   RankIC: {rankic_lgb:.6f}")
        results[f"{target_col}_lgb"] = rankic_lgb

        # 5. 固定値予測テスト（0.0719の謎を解明）
        print("\n5️⃣ Fixed Value Test (debugging 0.0719):")
        # いくつかの固定値でテスト
        test_values = [0, 0.001, 0.01, 0.1, 1.0]
        for test_val in test_values:
            y_pred_fixed = np.full(len(y_val), test_val)
            rankic_fixed = calculate_rankic(y_val.values, y_pred_fixed)
            print(f"   Fixed {test_val}: RankIC = {rankic_fixed:.6f}")

            # 0.0719に近い値を発見したら詳細分析
            if abs(rankic_fixed - 0.0719) < 0.001:
                print(f"   🔴 FOUND! Fixed value {test_val} gives RankIC ≈ 0.0719")

        # 6. わずかなノイズを加えた固定値
        print("\n6️⃣ Fixed + Small Noise:")
        y_pred_noise = np.full(len(y_val), y_train.mean()) + np.random.randn(len(y_val)) * 1e-6
        rankic_noise = calculate_rankic(y_val.values, y_pred_noise)
        print(f"   RankIC: {rankic_noise:.6f}")

    # 結果サマリー
    print("\n" + "=" * 60)
    print("📊 Summary Results")
    print("=" * 60)

    for target_col in target_cols:
        print(f"\n{target_col}:")
        print(f"  Random:  {results[f'{target_col}_random']:.6f}")
        print(f"  Mean:    {results[f'{target_col}_mean']:.6f}")
        print(f"  Linear:  {results[f'{target_col}_lr']:.6f}")
        print(f"  LightGBM: {results[f'{target_col}_lgb']:.6f}")

    # 診断
    print("\n" + "=" * 60)
    print("🏁 Diagnosis")
    print("=" * 60)

    # LightGBMのRankICをチェック
    lgb_rankics = [results[f'{t}_lgb'] for t in target_cols]
    max_rankic = max(lgb_rankics)

    if max_rankic < 0.01:
        print("🔴 CRITICAL: Even LightGBM cannot achieve meaningful RankIC!")
        print("   Possible issues:")
        print("   1. Data leakage in preprocessing")
        print("   2. Target-feature temporal misalignment")
        print("   3. Features not predictive of targets")
    elif max_rankic < 0.05:
        print("🟡 WARNING: Low predictability in data")
        print("   Maximum RankIC from LightGBM: {:.4f}".format(max_rankic))
        print("   The ATFT model should be able to exceed this")
    else:
        print("🟢 OK: Data has predictive signal")
        print("   Maximum RankIC from LightGBM: {:.4f}".format(max_rankic))
        print("   → ATFT-GAT-FAN should achieve > {:.4f}".format(max_rankic))

    # 0.0719の謎
    print("\n🔍 About the stuck RankIC = 0.0719:")
    if any(abs(results[f'{t}_mean'] - 0.0719) < 0.01 for t in target_cols):
        print("   → Likely the model is outputting constant/near-constant predictions")
        print("   → Check model initialization and gradient flow")
    else:
        print("   → The value doesn't match simple baselines")
        print("   → Could be a specific pattern in model outputs")

if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    test_baseline_models(data_path)