#!/usr/bin/env python3
"""
財務諸表データの頻度と結合の動作を検証
"""

import polars as pl
from datetime import date, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def create_sample_data():
    """サンプルデータを作成"""
    
    # 日次価格データ（3ヶ月分）
    dates = []
    for i in range(90):
        d = date(2024, 1, 1) + timedelta(days=i)
        dates.append(d)
    
    daily_quotes = pl.DataFrame({
        "Code": ["1301"] * 90,
        "Date": dates,
        "Close": [1000 + i for i in range(90)]
    })
    
    # 四半期決算データ（3ヶ月に1回）
    statements = pl.DataFrame({
        "Code": ["1301", "1301", "1301"],
        "DisclosedDate": [date(2023, 11, 10), date(2024, 2, 9), date(2024, 5, 10)],
        "DisclosedTime": ["14:00", "16:00", "15:30"],  # 15:00前、後、後
        "TypeOfCurrentPeriod": ["3Q", "FY", "1Q"],
        "NetSales": [1000000, 1500000, 400000],
        "OperatingProfit": [100000, 150000, 40000],
        "Profit": [80000, 120000, 32000],  # 純利益
        "ForecastOperatingProfit": [200000, 200000, 160000],
        "ForecastProfit": [160000, 160000, 128000],
        "Equity": [1000000, 1100000, 1200000],
        "TotalAssets": [2000000, 2200000, 2400000],
        "FiscalYear": [2023, 2023, 2024]
    })
    
    return daily_quotes, statements


def analyze_join_behavior():
    """as-of結合の動作を分析"""
    from features.safe_joiner import SafeJoiner
    
    daily, stmts = create_sample_data()
    
    joiner = SafeJoiner()
    
    # 結合実行
    result = joiner.join_statements_asof(
        daily, 
        stmts,
        use_time_cutoff=True,
        cutoff_time=pl.time(15, 0)
    )
    
    # 結果を分析
    print("=== 財務諸表データの結合結果 ===\n")
    
    # effective_dateの計算を確認
    print("1. T+1ルールの適用:")
    print("-" * 50)
    
    # effective_dateを再計算して表示
    stmts_with_effective = stmts.with_columns([
        pl.col("DisclosedDate").cast(pl.Date),
        pl.when(
            pl.col("DisclosedTime").str.slice(0, 2).cast(pl.Int32) < 15
        ).then(pl.col("DisclosedDate"))  # 15:00前 → 当日
        .otherwise(
            # 簡易的に+1日（実際は営業日カレンダーを使用）
            pl.col("DisclosedDate") + pl.duration(days=1)
        ).alias("effective_date")
    ])
    
    print(stmts_with_effective.select([
        "Code", "DisclosedDate", "DisclosedTime", "effective_date", "TypeOfCurrentPeriod"
    ]))
    
    print("\n2. データの展開頻度:")
    print("-" * 50)
    
    # 各日付でどの決算データが使用されているか確認
    analysis = result.group_by("stmt_days_since_statement").agg([
        pl.count().alias("days_count"),
        pl.col("Date").min().alias("first_date"),
        pl.col("Date").max().alias("last_date")
    ]).sort("stmt_days_since_statement")
    
    print(analysis)
    
    print("\n3. カバレッジ統計:")
    print("-" * 50)
    
    # 有効なデータを持つ日数
    valid_days = result.filter(pl.col("is_stmt_valid") == 1).height
    total_days = result.height
    coverage = valid_days / total_days * 100
    
    print(f"総日数: {total_days}")
    print(f"財務データ有効日数: {valid_days}")
    print(f"カバレッジ: {coverage:.1f}%")
    
    # 決算からの経過日数の分布
    print("\n4. 決算からの経過日数分布:")
    print("-" * 50)
    
    days_dist = result.filter(pl.col("is_stmt_valid") == 1).select("stmt_days_since_statement")
    print(f"最小: {days_dist.min()[0, 0]} 日")
    print(f"最大: {days_dist.max()[0, 0]} 日")
    print(f"平均: {days_dist.mean()[0, 0]:.1f} 日")
    print(f"中央値: {days_dist.median()[0, 0]} 日")
    
    # 実際のデータサンプル
    print("\n5. 実際の結合結果（サンプル）:")
    print("-" * 50)
    
    sample_dates = [
        date(2024, 1, 1),   # 初日
        date(2024, 2, 9),    # 決算発表日（16:00）
        date(2024, 2, 10),   # 決算発表翌日
        date(2024, 3, 15),   # 中間
        date(2024, 3, 31)    # 期末
    ]
    
    for d in sample_dates:
        row = result.filter(pl.col("Date") == d).select([
            "Date", "stmt_yoy_sales", "stmt_opm", 
            "stmt_imp_statement", "stmt_days_since_statement"
        ])
        if not row.is_empty():
            print(f"{d}: 決算インパクト={row[0, 'stmt_imp_statement']}, "
                  f"経過日数={row[0, 'stmt_days_since_statement']}, "
                  f"YoY売上={row[0, 'stmt_yoy_sales']:.3f if row[0, 'stmt_yoy_sales'] else 'N/A'}")


if __name__ == "__main__":
    analyze_join_behavior()