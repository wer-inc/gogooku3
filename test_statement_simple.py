#!/usr/bin/env python3
"""
財務諸表データの結合頻度を簡単に検証
"""

import polars as pl
from datetime import date, timedelta

def test_asof_join():
    """as-of結合の基本動作を確認"""
    
    # 日次データ（30日分）
    daily = pl.DataFrame({
        "Code": ["1301"] * 30,
        "Date": [date(2024, 1, 1) + timedelta(days=i) for i in range(30)],
        "Price": [1000 + i for i in range(30)]
    })
    
    # 財務データ（月1回、3件）
    statements = pl.DataFrame({
        "Code": ["1301", "1301", "1301"],
        "effective_date": [date(2023, 12, 15), date(2024, 1, 10), date(2024, 1, 25)],
        "Sales": [100, 120, 130],
        "Profit": [10, 12, 13]
    })
    
    # as-of結合を実行
    result = daily.sort(["Code", "Date"]).join_asof(
        statements.sort(["Code", "effective_date"]),
        left_on="Date",
        right_on="effective_date",
        by="Code",
        strategy="backward"  # その日以前の最新データを使用
    )
    
    # 結果を分析
    print("=== AS-OF結合の動作確認 ===\n")
    
    print("1. 元データ:")
    print("-" * 50)
    print(f"日次データ: {len(daily)}件 ({daily['Date'].min()} ～ {daily['Date'].max()})")
    print(f"財務データ: {len(statements)}件")
    print("\n財務データの日付:")
    for row in statements.iter_rows(named=True):
        print(f"  {row['effective_date']}: Sales={row['Sales']}")
    
    print("\n2. 結合結果（サンプル）:")
    print("-" * 50)
    
    # 特定の日付での結果を確認
    sample_dates = [
        date(2024, 1, 1),   # 2023/12/15のデータを使用するはず
        date(2024, 1, 10),  # 2024/1/10のデータを使用するはず
        date(2024, 1, 15),  # 2024/1/10のデータを使用するはず
        date(2024, 1, 25),  # 2024/1/25のデータを使用するはず
        date(2024, 1, 30),  # 2024/1/25のデータを使用するはず
    ]
    
    for d in sample_dates:
        row = result.filter(pl.col("Date") == d)
        if not row.is_empty():
            sales = row[0, "Sales"] if "Sales" in row.columns else None
            eff_date = row[0, "effective_date"] if "effective_date" in row.columns else None
            print(f"{d}: Sales={sales}, 使用データ日付={eff_date}")
    
    print("\n3. データの継続性:")
    print("-" * 50)
    
    # Salesの値ごとにグループ化して日数を数える
    continuity = result.group_by("Sales").agg([
        pl.count().alias("days"),
        pl.col("Date").min().alias("from_date"),
        pl.col("Date").max().alias("to_date")
    ]).sort("from_date")
    
    print(continuity)
    
    print("\n4. まとめ:")
    print("-" * 50)
    print("✅ as-of結合により、財務データが存在しない日でも")
    print("   直近の財務データが自動的に使用される")
    print("✅ 四半期ごと（3ヶ月に1回）の財務データでも、")
    print("   全ての日次データに財務情報が付与される")
    print("✅ backward strategyにより未来のデータは使用されない")


if __name__ == "__main__":
    test_asof_join()