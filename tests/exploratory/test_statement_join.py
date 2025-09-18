#!/usr/bin/env python3
"""
財務諸表結合の修正テスト
"""

import sys
import polars as pl
from datetime import datetime, date

sys.path.append('/home/ubuntu/gogooku3-standalone')
sys.path.append('/home/ubuntu/gogooku3-standalone/src')

from features.safe_joiner import SafeJoiner


def test_statement_join():
    """JQuantsから取得される実際のデータ形式でテスト"""
    print("=" * 60)
    print("Testing Statement Join with Real Data Format")
    print("=" * 60)
    
    # 実際のJQuantsデータのフォーマット（Code は数値型）
    statements_df = pl.DataFrame({
        'Code': [7203, 6501, 8306],  # Int64型
        'LocalCode': ['7203', '6501', '8306'],
        'DisclosedDate': ['2025-09-01', '2025-09-02', '2025-09-03'],
        'DisclosedTime': ['14:00', '15:30', '16:00'],
        'FiscalYear': [2025, 2025, 2025],
        'TypeOfCurrentPeriod': ['1Q', '2Q', '1Q'],
        'NetSales': [10000000000, 5000000000, 3000000000],
        'OperatingProfit': [1000000000, 500000000, 400000000],
        'Profit': [800000000, 400000000, 300000000],
        'ForecastOperatingProfit': [1200000000, 600000000, 500000000],
        'ForecastProfit': [1000000000, 500000000, 400000000],
        'ForecastEarningsPerShare': [100.0, 50.0, 40.0],
        'ForecastDividendPerShareAnnual': [50, 25, 20],
        'Equity': [20000000000, 10000000000, 8000000000],
        'TotalAssets': [50000000000, 25000000000, 20000000000],
    })
    
    # 価格データ（Code は文字列型）
    base_df = pl.DataFrame({
        'Code': ['7203', '7203', '7203', '6501', '6501', '8306', '8306'],
        'Date': [
            date(2025, 9, 1),
            date(2025, 9, 2),
            date(2025, 9, 3),
            date(2025, 9, 2),
            date(2025, 9, 3),
            date(2025, 9, 3),
            date(2025, 9, 4),
        ],
        'Close': [2500.0, 2510.0, 2520.0, 1500.0, 1510.0, 900.0, 910.0],
        'Volume': [10000000, 11000000, 12000000, 5000000, 5100000, 3000000, 3100000]
    })
    
    print("Input DataFrames:")
    print(f"  statements_df: {statements_df.shape}")
    print(f"    Code type: {statements_df.schema['Code']}")
    print(f"  base_df: {base_df.shape}")
    print(f"    Code type: {base_df.schema['Code']}")
    
    # SafeJoinerで結合
    joiner = SafeJoiner()
    result = joiner.join_statements_asof(
        base_df=base_df,
        statements_df=statements_df,
        use_time_cutoff=True
    )
    
    print("\n✅ Joined Result:")
    print(f"  Shape: {result.shape}")
    
    # stmt_* 列の確認
    stmt_cols = [col for col in result.columns if col.startswith("stmt_")]
    print(f"  Statement columns added: {len(stmt_cols)}")
    
    # 実際の値を確認
    if stmt_cols:
        print("\n✅ Sample values (non-zero check):")
        for col in ["stmt_opm", "stmt_roe", "stmt_roa"]:
            if col in result.columns:
                values = result[col].to_list()
                non_zero = sum(1 for v in values if v != 0.0 and v is not None)
                print(f"  {col}: {non_zero}/{len(values)} non-zero values")
                if non_zero > 0:
                    non_zero_vals = [v for v in values if v != 0.0 and v is not None]
                    print(f"    Sample values: {non_zero_vals[:3]}")
    
    # Coverage確認
    if "is_stmt_valid" in result.columns:
        coverage = (result["is_stmt_valid"] == 1).sum() / len(result)
        print(f"\n✅ Statement coverage: {coverage:.1%}")
    
    return result


def main():
    result = test_statement_join()
    
    print("\n" + "=" * 60)
    print("TEST RESULTS:")
    print("=" * 60)
    
    # 成功判定
    stmt_cols = [col for col in result.columns if col.startswith("stmt_")]
    success = False
    
    if stmt_cols:
        # マージン系の列で値があるか確認
        for col in ["stmt_opm", "stmt_npm", "stmt_roe", "stmt_roa"]:
            if col in result.columns:
                non_zero = (result[col] != 0.0).sum()
                if non_zero > 0:
                    success = True
                    break
    
    if success:
        print("✅ Statement join is working correctly!")
        print("   Financial features are being calculated")
    else:
        print("❌ Statement join still has issues")
        print("   No financial values found")
    
    print("\nNext step:")
    print("  Run full pipeline with fixed statement join:")
    print("  python scripts/pipelines/run_pipeline_v4_optimized.py --jquants")


if __name__ == "__main__":
    main()