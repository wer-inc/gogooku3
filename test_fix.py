#!/usr/bin/env python3
"""
財務特徴量の修正確認テストスクリプト
"""

import sys
import polars as pl
import numpy as np
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.append('/home/ubuntu/gogooku3-standalone')
sys.path.append('/home/ubuntu/gogooku3-standalone/src')

# 必要なモジュールをインポート
from src.features.safe_joiner import SafeJoiner
from src.features.market_features import MarketFeaturesGenerator, CrossMarketFeaturesGenerator

def test_financial_features():
    """財務諸表特徴量の結合をテスト"""
    print("=" * 60)
    print("Testing Financial Statement Features")
    print("=" * 60)
    
    # サンプルの基盤データ（価格）
    base_df = pl.DataFrame({
        "Code": ["7203", "7203", "7203", "7203", "7203"],  # トヨタ
        "Date": [
            datetime(2024, 1, 10).date(),
            datetime(2024, 1, 11).date(),
            datetime(2024, 1, 12).date(),
            datetime(2024, 1, 15).date(),
            datetime(2024, 1, 16).date(),
        ],
        "Close": [2500.0, 2510.0, 2520.0, 2530.0, 2540.0],
        "Volume": [10000000, 11000000, 12000000, 13000000, 14000000]
    })
    
    # サンプルの財務諸表データ
    statements_df = pl.DataFrame({
        "Code": ["7203", "7203"],
        "DisclosedDate": [
            datetime(2024, 1, 10).date(),
            datetime(2024, 1, 15).date()
        ],
        "DisclosedTime": ["14:00", "16:00"],  # 1つ目は15:00前、2つ目は15:00後
        "FiscalYear": [2024, 2024],
        "TypeOfCurrentPeriod": ["3Q", "3Q"],
        "NetSales": [10000000.0, 10500000.0],
        "OperatingProfit": [1000000.0, 1100000.0],
        "Profit": [800000.0, 900000.0],
        "ForecastOperatingProfit": [1200000.0, 1300000.0],
        "ForecastProfit": [1000000.0, 1100000.0],
        "ForecastEarningsPerShare": [100.0, 110.0],
        "ForecastDividendPerShareAnnual": [50.0, 55.0],
        "Equity": [20000000.0, 21000000.0],
        "TotalAssets": [50000000.0, 52000000.0]
    })
    
    # SafeJoinerで結合
    joiner = SafeJoiner()
    result = joiner.join_statements_asof(
        base_df=base_df,
        statements_df=statements_df,
        use_time_cutoff=True
    )
    
    # 結果を表示
    print("\n✅ Joined result shape:", result.shape)
    print("\n✅ Statement columns added:")
    stmt_cols = [col for col in result.columns if col.startswith("stmt_")]
    for col in stmt_cols:
        print(f"  - {col}")
    
    # 値をチェック
    print("\n✅ Sample values:")
    for col in ["stmt_yoy_sales", "stmt_opm", "stmt_roe"]:
        if col in result.columns:
            values = result[col].to_list()
            non_zero = sum(1 for v in values if v != 0.0 and v is not None)
            print(f"  {col}: {non_zero}/{len(values)} non-zero values")
            print(f"    Values: {values}")
    
    return result


def test_market_features():
    """市場特徴量（idio_vol_ratio修正済み）のテスト"""
    print("\n" + "=" * 60)
    print("Testing Market Features (Fixed idio_vol_ratio)")
    print("=" * 60)
    
    # サンプルデータ
    stock_df = pl.DataFrame({
        "Code": ["7203", "7203", "7203"],
        "Date": [
            datetime(2024, 1, 10).date(),
            datetime(2024, 1, 11).date(),
            datetime(2024, 1, 12).date()
        ],
        "returns_1d": [0.01, -0.005, 0.008],
        "volatility_20d": [0.02, 0.021, 0.019]  # 銘柄のボラティリティ
    })
    
    market_df = pl.DataFrame({
        "Date": [
            datetime(2024, 1, 10).date(),
            datetime(2024, 1, 11).date(),
            datetime(2024, 1, 12).date()
        ],
        "mkt_ret_1d": [0.005, -0.002, 0.003],
        "mkt_vol_20d": [0.015, 0.016, 0.014]  # 市場のボラティリティ
    })
    
    # クロス特徴量生成器
    cross_gen = CrossMarketFeaturesGenerator()
    result = cross_gen.attach_market_and_cross(stock_df, market_df)
    
    if "idio_vol_ratio" in result.columns:
        print("\n✅ idio_vol_ratio calculation (FIXED):")
        print("  Formula: volatility_20d / (mkt_vol_20d + 1e-12)")
        
        for row in result.iter_rows(named=True):
            expected = row["volatility_20d"] / (row["mkt_vol_20d"] + 1e-12)
            actual = row["idio_vol_ratio"]
            print(f"  Date {row['Date']}: {actual:.4f} (expected: {expected:.4f})")
            
            # 検証
            if abs(actual - expected) < 1e-6:
                print("    ✅ Correct!")
            else:
                print("    ❌ Mismatch!")
    
    return result


def main():
    """メイン実行"""
    # 財務特徴量テスト
    stmt_result = test_financial_features()
    
    # 市場特徴量テスト（修正済み）
    market_result = test_market_features()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    print("\n✅ Financial features:")
    stmt_cols = [col for col in stmt_result.columns if col.startswith("stmt_")]
    non_zero_count = 0
    for col in stmt_cols:
        if col not in ["stmt_change_in_est", "stmt_nc_flag", "is_stmt_valid"]:
            non_zero = (stmt_result[col] != 0.0).sum()
            if non_zero > 0:
                non_zero_count += 1
    print(f"  {non_zero_count}/{len(stmt_cols)} features have non-zero values")
    
    print("\n✅ Market features:")
    if "idio_vol_ratio" in market_result.columns:
        idio_values = market_result["idio_vol_ratio"].to_list()
        print(f"  idio_vol_ratio range: {min(idio_values):.2f} - {max(idio_values):.2f}")
        print("  (Expected range: 1.0 - 3.0 for typical stocks)")
    
    print("\n🎯 NEXT STEP:")
    print("  Run full pipeline to regenerate dataset with fixes:")
    print("  python scripts/pipelines/run_pipeline_v4_optimized.py --jquants")


if __name__ == "__main__":
    main()