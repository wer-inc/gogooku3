#!/usr/bin/env python3
"""
ベースラインモデル（LightGBM）の特徴量重要度を分析して、
ATFT-GAT-FANモデルの改善に活用
"""

import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")


def calculate_rankic(y_true, y_pred):
    """RankIC計算"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 10:
        return np.nan
    corr, _ = spearmanr(y_true[mask], y_pred[mask])
    return corr


def analyze_feature_importance(data_path="output/ml_dataset_future_returns.parquet"):
    """特徴量重要度を分析してATFTの改善に活用"""

    print("=" * 60)
    print("🔍 Feature Importance Analysis for ATFT Improvement")
    print("=" * 60)

    # データ読み込み
    print("\n📂 Loading data...")
    df = pd.read_parquet(data_path)
    print(f"✅ Data shape: {df.shape}")

    # 特徴量とターゲット
    feature_cols = [
        col
        for col in df.columns
        if not col.startswith("returns_") and col not in ["Date", "Code"]
    ]

    # 数値特徴量のみ
    numeric_features = [
        col
        for col in feature_cols
        if df[col].dtype in ["float64", "float32", "int64", "int32"]
    ]

    print(f"\n📊 Features: {len(numeric_features)}")

    # データ準備
    df = df.sort_values("Date")

    # Train/Val分割
    split_date = df["Date"].quantile(0.8)
    train_df = df[df["Date"] < split_date]
    val_df = df[df["Date"] >= split_date]

    X_train = train_df[numeric_features].fillna(0).clip(-10, 10)
    X_val = val_df[numeric_features].fillna(0).clip(-10, 10)

    # 複数のターゲットで特徴量重要度を収集
    feature_importance_dict = {}
    performance_dict = {}

    for target_col in ["returns_1d", "returns_5d", "returns_10d", "returns_20d"]:
        print(f"\n🎯 Analyzing {target_col}...")

        y_train = train_df[target_col].fillna(0)
        y_val = val_df[target_col].fillna(0)

        # サンプルサイズ制限
        sample_size = min(100000, len(X_train))
        indices = np.random.choice(len(X_train), sample_size, replace=False)

        # LightGBMモデル（最適化済み）
        lgb = LGBMRegressor(
            n_estimators=300,  # 増加
            max_depth=7,  # 増加
            learning_rate=0.05,  # 調整
            num_leaves=63,  # 増加
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,  # L1正則化
            reg_lambda=0.1,  # L2正則化
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        )

        # 学習
        lgb.fit(
            X_train.iloc[indices],
            y_train.iloc[indices],
            eval_set=[(X_val.iloc[:10000], y_val.iloc[:10000])],
            callbacks=[lambda x: None],
        )

        # 予測とRankIC計算
        y_pred = lgb.predict(X_val)
        rankic = calculate_rankic(y_val.values, y_pred)

        print(f"   RankIC: {rankic:.6f}")
        performance_dict[target_col] = rankic

        # 特徴量重要度
        importance = pd.DataFrame(
            {"feature": numeric_features, "importance": lgb.feature_importances_}
        ).sort_values("importance", ascending=False)

        feature_importance_dict[target_col] = importance

        # Top10特徴量
        print("   Top 10 features:")
        for idx, row in importance.head(10).iterrows():
            print(f"     {row['feature']}: {row['importance']:.0f}")

    # 全体の特徴量重要度集計
    print("\n" + "=" * 60)
    print("📈 Aggregated Feature Importance")
    print("=" * 60)

    # 各ターゲットの重要度を平均
    all_features = set()
    for importance_df in feature_importance_dict.values():
        all_features.update(importance_df["feature"].values)

    aggregated_importance = []
    for feature in all_features:
        importances = []
        for target_col, importance_df in feature_importance_dict.items():
            imp = importance_df[importance_df["feature"] == feature][
                "importance"
            ].values
            if len(imp) > 0:
                importances.append(imp[0])
            else:
                importances.append(0)
        aggregated_importance.append(
            {
                "feature": feature,
                "avg_importance": np.mean(importances),
                "max_importance": np.max(importances),
                "std_importance": np.std(importances),
            }
        )

    aggregated_df = pd.DataFrame(aggregated_importance).sort_values(
        "avg_importance", ascending=False
    )

    print("\nTop 20 Most Important Features (Averaged):")
    for idx, row in aggregated_df.head(20).iterrows():
        print(
            f"  {row['feature']}: avg={row['avg_importance']:.1f}, max={row['max_importance']:.1f}"
        )

    # 特徴量カテゴリ分析
    print("\n" + "=" * 60)
    print("🏷️ Feature Category Analysis")
    print("=" * 60)

    # カテゴリ分類
    categories = {
        "price": ["Open", "High", "Low", "Close", "VWAP"],
        "volume": ["Volume", "TurnoverValue"],
        "return": ["ret_", "return"],
        "technical": ["rsi", "bb_", "macd", "sma", "ema", "adx", "cci", "stoch"],
        "market_cap": ["MarketCap", "market_cap"],
        "fundamental": ["per", "pbr", "psr", "dividend"],
        "volatility": ["volatility", "std", "atr"],
        "momentum": ["momentum", "roc"],
        "liquidity": ["liquidity", "spread", "turnover"],
    }

    category_importance = {}
    for category, keywords in categories.items():
        category_features = []
        for feature in aggregated_df["feature"]:
            for keyword in keywords:
                if keyword.lower() in feature.lower():
                    category_features.append(feature)
                    break

        if category_features:
            cat_importance = aggregated_df[
                aggregated_df["feature"].isin(category_features)
            ]["avg_importance"].sum()
            category_importance[category] = cat_importance
            print(f"  {category}: {cat_importance:.1f} (n={len(category_features)})")

    # パフォーマンスサマリー
    print("\n" + "=" * 60)
    print("🎯 Performance Summary")
    print("=" * 60)

    for target, rankic in performance_dict.items():
        print(f"  {target}: RankIC = {rankic:.6f}")

    avg_rankic = np.mean(list(performance_dict.values()))
    print(f"\n  Average RankIC: {avg_rankic:.6f}")

    # 改善提案
    print("\n" + "=" * 60)
    print("💡 Recommendations for ATFT-GAT-FAN")
    print("=" * 60)

    top_features = aggregated_df.head(20)["feature"].tolist()

    print("\n1. **Feature Engineering**:")
    print("   重要度の高い特徴量を強調:")
    for i, feat in enumerate(top_features[:5], 1):
        print(f"   {i}. {feat}")

    print("\n2. **Feature Groups for Attention**:")
    sorted_categories = sorted(
        category_importance.items(), key=lambda x: x[1], reverse=True
    )
    for cat, imp in sorted_categories[:3]:
        print(f"   - {cat.upper()}: 高重要度カテゴリ")

    print("\n3. **Model Architecture**:")
    if avg_rankic > 0.07:
        print("   ✅ データに予測可能性あり")
        print("   → より深いネットワーク（hidden_size=256以上）")
        print("   → Feature-wise attention強化")
    else:
        print("   ⚠️ 予測可能性が低い")
        print("   → 正則化強化")
        print("   → アンサンブル手法検討")

    print("\n4. **Training Strategy**:")
    print("   - 重要特徴量に基づくカリキュラム学習")
    print("   - Feature dropoutで過学習防止")
    print("   - Multi-task learning（複数horizon同時学習）")

    return aggregated_df, performance_dict


if __name__ == "__main__":
    import sys

    data_path = (
        sys.argv[1] if len(sys.argv) > 1 else "output/ml_dataset_future_returns.parquet"
    )

    feature_importance, performance = analyze_feature_importance(data_path)

    # CSVに保存
    output_path = "output/feature_importance_analysis.csv"
    feature_importance.to_csv(output_path, index=False)
    print(f"\n📊 Feature importance saved to: {output_path}")
