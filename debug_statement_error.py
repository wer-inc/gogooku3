#!/usr/bin/env python3
"""
財務諸表結合の詳細デバッグ
"""

import sys
import polars as pl
import traceback
from datetime import datetime, date

sys.path.append('/home/ubuntu/gogooku3-standalone')
sys.path.append('/home/ubuntu/gogooku3-standalone/src')

from features.safe_joiner import SafeJoiner


# パイプラインから実際のデータ形式を再現
statements_df = pl.DataFrame({
    'Code': [7203, 6501],  # Int64型
    'LocalCode': ['7203', '6501'],
    'DisclosedDate': '2025-09-01',  # 文字列
    'DisclosedTime': ['14:00', '15:30'],
    'FiscalYear': 2025,  # 数値
    'TypeOfCurrentPeriod': ['1Q', '2Q'],
    'NetSales': [10000000000.0, 5000000000.0],
    'OperatingProfit': [1000000000.0, 500000000.0],
    'Profit': [800000000.0, 400000000.0],
    'ForecastOperatingProfit': [1200000000.0, 600000000.0],
    'ForecastProfit': [1000000000.0, 500000000.0],
    'ForecastEarningsPerShare': [100.0, 50.0],
    'ForecastDividendPerShareAnnual': [50.0, 25.0],
    'Equity': [20000000000.0, 10000000000.0],
    'TotalAssets': [50000000000.0, 25000000000.0],
})

# 価格データ
base_df = pl.DataFrame({
    'Code': ['7203', '6501'],
    'Date': ['2025-09-01', '2025-09-02'],  # 文字列で来ることもある
    'Close': [2500.0, 1500.0],
})

print("Input DataFrames Schema:")
print("\nstatements_df:")
for col, dtype in statements_df.schema.items():
    print(f"  {col}: {dtype}")
    
print("\nbase_df:")
for col, dtype in base_df.schema.items():
    print(f"  {col}: {dtype}")

# Date列を確実に日付型に変換
base_df = base_df.with_columns([
    pl.col("Date").str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
])

print("\nConverted base_df Date type:", base_df.schema["Date"])

# SafeJoinerでテスト
joiner = SafeJoiner()

print("\n" + "=" * 60)
print("Testing join_statements_asof")
print("=" * 60)

try:
    result = joiner.join_statements_asof(
        base_df=base_df,
        statements_df=statements_df,
        use_time_cutoff=True
    )
    print("✅ Success!")
    print(f"Result shape: {result.shape}")
    
    # stmt_* 列の確認
    stmt_cols = [col for col in result.columns if col.startswith("stmt_")]
    print(f"Statement columns: {len(stmt_cols)}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    
    # どこでエラーが発生したか特定
    print("\n" + "=" * 60)
    print("Debugging the error step by step...")
    print("=" * 60)
    
    # ステップ1: Code列の変換
    print("\n1. Converting Code columns...")
    try:
        base_df_converted = base_df.with_columns([
            pl.col("Code").cast(pl.Utf8).str.zfill(4)
        ])
        print("   base_df Code conversion: OK")
        
        stm = statements_df.with_columns([
            pl.col("Code").cast(pl.Utf8).str.zfill(4)
        ])
        print("   statements_df Code conversion: OK")
    except Exception as e2:
        print(f"   Error in Code conversion: {e2}")
    
    # ステップ2: 日付変換
    print("\n2. Converting date columns...")
    try:
        # FiscalYearの処理確認
        if "FiscalYear" in stm.columns:
            print(f"   FiscalYear type: {stm.schema['FiscalYear']}")
            # FiscalYearを明示的にInt32に変換
            stm = stm.with_columns([
                pl.col("FiscalYear").cast(pl.Int32, strict=False)
            ])
        
        # DisclosedDateの処理
        if stm.schema.get("DisclosedDate") == pl.Utf8:
            stm = stm.with_columns([
                pl.col("DisclosedDate").str.strptime(pl.Date, format="%Y-%m-%d", strict=False).alias("disclosed_date")
            ])
            print("   DisclosedDate conversion: OK")
        else:
            stm = stm.with_columns([
                pl.col("DisclosedDate").cast(pl.Date).alias("disclosed_date")
            ])
    except Exception as e3:
        print(f"   Error in date conversion: {e3}")
        traceback.print_exc()