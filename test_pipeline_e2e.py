#!/usr/bin/env python
"""
エンドツーエンドパイプライン検証スクリプト
データ取得 → データ生成 → データセット作成 → ML学習
"""

import sys
import os
import polars as pl
import numpy as np
from datetime import datetime, timedelta
import pytz  # タイムゾーンサポート
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# JSTタイムゾーン設定
JST = pytz.timezone('Asia/Tokyo')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "./"))

# Import our modules
from src.gogooku3.features.ta_core import TechnicalIndicators, CrossSectionalNormalizer
from src.gogooku3.joins.intervals import MarketDataJoiner
from src.gogooku3.contracts.schemas import DataSchemas, SchemaValidator

print("=" * 80)
print("エンドツーエンド MLパイプライン検証")
print("=" * 80)
print()

# ===============================
# STEP 1: データ取得（サンプルデータ生成）
# ===============================
print("STEP 1: データ取得")
print("-" * 40)

def generate_sample_data(n_stocks=10, n_days=500):
    """サンプル株価データを生成"""
    np.random.seed(42)
    
    stocks = [f"{1300+i}" for i in range(n_stocks)]
    start_date = datetime(2023, 1, 1, tzinfo=JST)  # JSTで初期化
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    data = []
    for stock in stocks:
        base_price = 1000 + np.random.rand() * 2000
        prices = []
        
        # Generate realistic price movement
        for i in range(n_days):
            if i == 0:
                price = base_price
            else:
                # Random walk with drift
                returns = np.random.randn() * 0.02  # 2% daily volatility
                price = prices[-1] * (1 + returns)
            prices.append(price)
        
        # Create OHLCV data
        for i, (date, close) in enumerate(zip(dates, prices)):
            high = close * (1 + abs(np.random.randn() * 0.01))
            low = close * (1 - abs(np.random.randn() * 0.01))
            open_price = (high + low) / 2 + np.random.randn() * (high - low) * 0.1
            volume = int(1000000 * (1 + abs(np.random.randn())))
            
            data.append({
                "Code": stock,
                "Date": date.date(),
                "Open": open_price,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
                "TurnoverValue": close * volume,
                "adj_factor": 1.0,
                "SharesOutstanding": 100000000
            })
    
    return pl.DataFrame(data)

# Generate sample data
print(f"サンプルデータ生成中...")
df_quotes = generate_sample_data(n_stocks=10, n_days=500)
print(f"✅ 生成完了: {len(df_quotes)}行 × {len(df_quotes.columns)}列")
print(f"   銘柄数: {df_quotes['Code'].n_unique()}")
print(f"   期間: {df_quotes['Date'].min()} 〜 {df_quotes['Date'].max()}")
print()

# ===============================
# STEP 2: テクニカル指標生成
# ===============================
print("STEP 2: テクニカル指標・特徴量生成")
print("-" * 40)

# Apply technical indicators per stock
print("テクニカル指標を計算中...")
result_dfs = []
for code in df_quotes["Code"].unique():
    code_df = df_quotes.filter(pl.col("Code") == code).sort("Date")
    
    # Add all technical indicators
    code_df = TechnicalIndicators.add_all_indicators(
        code_df,
        return_periods=[1, 5, 10, 20],
        ma_windows=[5, 20, 60],  # Skip 200 for short sample
        vol_windows=[20, 60],
        realized_vol_method="parkinson",
        target_horizons=[1, 5, 10, 20]
    )
    result_dfs.append(code_df)

df_with_ta = pl.concat(result_dfs)

# Add px_ prefix
feature_cols = [col for col in df_with_ta.columns 
                if col not in ["Code", "Date", "adj_factor", "SharesOutstanding"] 
                and not col.startswith("y_")]

rename_dict = {col: f"px_{col.lower()}" if not col.startswith("px_") else col 
               for col in feature_cols}
df_with_ta = df_with_ta.rename(rename_dict)

print(f"✅ テクニカル指標生成完了: {len(df_with_ta.columns)}列")
print(f"   追加された指標数: {len(df_with_ta.columns) - len(df_quotes.columns)}")
print()

# ===============================
# STEP 3: 市場特徴量生成
# ===============================
print("STEP 3: 市場特徴量生成")
print("-" * 40)

# Create market index (simple average)
print("市場インデックスを作成中...")
market_df = df_quotes.group_by("Date").agg([
    pl.col("Close").mean().alias("Close"),
    pl.col("High").mean().alias("High"),
    pl.col("Low").mean().alias("Low")
]).sort("Date")

# Add market features
for horizon in [1, 5, 10, 20]:
    market_df = market_df.with_columns(
        (pl.col("Close") / pl.col("Close").shift(horizon) - 1).alias(f"mkt_ret_{horizon}d")
    )

# Add market volatility
market_df = market_df.with_columns(
    (pl.col("mkt_ret_1d").rolling_std(20) * np.sqrt(252)).alias("mkt_vol_20d")
)

# Add simple regime flags
market_df = market_df.with_columns([
    (pl.col("mkt_ret_20d") > 0).cast(pl.Int8).alias("mkt_bull_200"),
    (pl.col("mkt_vol_20d") > pl.col("mkt_vol_20d").median()).cast(pl.Int8).alias("mkt_high_vol")
])

print(f"✅ 市場特徴量生成完了: {len(market_df.columns)}列")
print()

# ===============================
# STEP 4: データ統合（MLパネル作成）
# ===============================
print("STEP 4: MLパネル作成")
print("-" * 40)

# Prepare for join
df_panel = df_with_ta.select([
    pl.col("Code").alias("meta_code"),
    pl.col("Date").alias("meta_date"),
    *[col for col in df_with_ta.columns if col.startswith("px_") or col.startswith("y_")]
])

# Add dummy section
df_panel = df_panel.with_columns(
    pl.lit("Prime").alias("meta_section").cast(pl.Categorical)
)

# Join market features
print("市場特徴量を結合中...")
df_panel = df_panel.join(market_df, left_on="meta_date", right_on="Date", how="left")

# Calculate cross features (simplified)
print("クロス特徴量を計算中...")
if "px_returns_1d" in df_panel.columns and "mkt_ret_1d" in df_panel.columns:
    df_panel = df_panel.with_columns([
        (pl.col("px_returns_1d") - pl.col("mkt_ret_1d")).alias("x_alpha_1d"),
        (pl.col("px_returns_1d") / (pl.col("mkt_ret_1d") + 1e-12)).alias("x_rel_strength_1d")
    ])

# Add placeholder flow/financial features
flow_features = ["flow_buy_ratio", "flow_sell_ratio", "flow_net_ratio"]
stmt_features = ["stmt_revenue_yoy", "stmt_roe", "stmt_roa"]  # 仕様に準拠: stmt_* プレフィックス

for feat in flow_features + stmt_features:
    df_panel = df_panel.with_columns(pl.lit(0.0).alias(feat))

print(f"✅ MLパネル作成完了: {len(df_panel)}行 × {len(df_panel.columns)}列")
print(f"   特徴量カテゴリ:")
print(f"   - 価格系 (px_*): {len([c for c in df_panel.columns if c.startswith('px_')])}列")
print(f"   - 市場系 (mkt_*): {len([c for c in df_panel.columns if c.startswith('mkt_')])}列")
print(f"   - クロス系 (x_*): {len([c for c in df_panel.columns if c.startswith('x_')])}列")
print(f"   - フロー系 (flow_*): {len([c for c in df_panel.columns if c.startswith('flow_')])}列")
print(f"   - 財務系 (stmt_*): {len([c for c in df_panel.columns if c.startswith('stmt_')])}列")
print(f"   - ターゲット (y_*): {len([c for c in df_panel.columns if c.startswith('y_')])}列")
print()

# ===============================
# STEP 5: 断面正規化
# ===============================
print("STEP 5: 断面正規化")
print("-" * 40)

# Select features for normalization
feature_cols = [col for col in df_panel.columns 
                if not col.startswith("meta_") and not col.startswith("y_")]

print(f"正規化対象: {len(feature_cols)}列")

# Apply cross-sectional normalization (optional for this test)
# df_panel = CrossSectionalNormalizer.normalize_daily(
#     df_panel, feature_cols, method="zscore", robust=True, winsorize_pct=0.01
# )

print("✅ 正規化スキップ（テスト用）")
print()

# ===============================
# STEP 6: ML学習（LightGBM）
# ===============================
print("STEP 6: ML学習")
print("-" * 40)

# Prepare data for training
print("学習データを準備中...")

# Remove rows with NaN in target
df_train = df_panel.filter(pl.col("y_5d").is_not_null())

# Select features
feature_cols = [col for col in df_train.columns 
                if col.startswith(("px_", "mkt_", "x_", "flow_", "stmt_"))
                and not col.startswith("y_")]

# Remove features with too many NaNs
valid_features = []
for col in feature_cols:
    null_rate = df_train[col].null_count() / len(df_train)
    if null_rate < 0.5:  # Keep features with <50% nulls
        valid_features.append(col)

print(f"有効な特徴量: {len(valid_features)}/{len(feature_cols)}")

# Fill NaN with 0 (simple approach for test)
df_train = df_train.fill_null(0)

# Convert to numpy
X = df_train.select(valid_features).to_numpy()
y = df_train["y_5d"].to_numpy()

# Train/test split (time-based)
dates_sorted = df_train["meta_date"].sort().unique()
split_idx = int(len(dates_sorted) * 0.8)
split_date = dates_sorted[split_idx]
train_mask = df_train["meta_date"] <= split_date
test_mask = df_train["meta_date"] > split_date

X_train = X[train_mask]
y_train = y[train_mask]
X_test = X[test_mask]
y_test = y[test_mask]

print(f"学習データ: {X_train.shape}")
print(f"テストデータ: {X_test.shape}")

# Train LightGBM
print("\nLightGBM学習中...")
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'verbose': -1,
    'seed': 42
}

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

model = lgb.train(
    params,
    train_data,
    valid_sets=[valid_data],
    num_boost_round=100,
    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
)

print(f"✅ 学習完了: {model.num_trees()}ラウンド")
print()

# ===============================
# STEP 7: 評価
# ===============================
print("STEP 7: モデル評価")
print("-" * 40)

# Predict
y_pred = model.predict(X_test, num_iteration=model.best_iteration)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
corr = np.corrcoef(y_test, y_pred)[0, 1]

# Calculate IC (Information Coefficient) per date
test_df = df_train.filter(test_mask).select(["meta_date", "meta_code"])
test_df = test_df.with_columns([
    pl.Series("y_true", y_test),
    pl.Series("y_pred", y_pred)
])

ic_per_date = []
for date in test_df["meta_date"].unique():
    date_df = test_df.filter(pl.col("meta_date") == date)
    if len(date_df) > 1:
        date_ic = np.corrcoef(date_df["y_true"], date_df["y_pred"])[0, 1]
        if not np.isnan(date_ic):
            ic_per_date.append(date_ic)

mean_ic = np.mean(ic_per_date) if ic_per_date else 0

print("📊 評価結果:")
print(f"   RMSE: {rmse:.6f}")
print(f"   MAE:  {mae:.6f}")
print(f"   相関係数: {corr:.4f}")
print(f"   平均IC: {mean_ic:.4f}")
print()

# Feature importance
importance = model.feature_importance(importance_type='gain')
feature_imp = sorted(zip(valid_features, importance), key=lambda x: x[1], reverse=True)

print("📈 上位10特徴量:")
for i, (feat, imp) in enumerate(feature_imp[:10], 1):
    print(f"   {i:2d}. {feat:30s}: {imp:10.2f}")

print()
print("=" * 80)
print("✅ エンドツーエンドパイプライン検証完了!")
print("=" * 80)

# Save results
output_dir = "output/e2e_test"
os.makedirs(output_dir, exist_ok=True)

# Save panel data
df_panel.write_parquet(f"{output_dir}/ml_panel_e2e.parquet")
print(f"\nデータ保存: {output_dir}/ml_panel_e2e.parquet")

# Save model
model.save_model(f"{output_dir}/model_e2e.txt")
print(f"モデル保存: {output_dir}/model_e2e.txt")

print("\n検証結果サマリー:")
print(f"  ✅ データ取得: {df_quotes['Code'].n_unique()}銘柄 × {len(df_quotes['Date'].unique())}日")
print(f"  ✅ 特徴量生成: {len(df_panel.columns)}列")
print(f"  ✅ ML学習: LightGBM {model.num_trees()}ラウンド")
print(f"  ✅ 予測精度: IC={mean_ic:.4f}, Corr={corr:.4f}")