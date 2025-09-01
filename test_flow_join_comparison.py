#!/usr/bin/env python3
"""
週次フローデータと日次データの結合方法の比較テスト
"""

import polars as pl
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data():
    """テスト用のサンプルデータを作成"""
    
    # 日次株価データ（20営業日分）
    dates = []
    base_date = datetime(2024, 1, 8)  # 月曜日から開始
    for i in range(20):
        if base_date.weekday() < 5:  # 平日のみ
            dates.append(base_date)
        base_date += timedelta(days=1 if base_date.weekday() < 4 else 3)
    
    stock_data = []
    for code in ["1301", "5401", "7203"]:  # 3銘柄
        for date in dates:
            stock_data.append({
                "Code": code,
                "Date": date.date(),
                "Close": 1000.0
            })
    
    stock_df = pl.DataFrame(stock_data)
    
    # 週次フローデータ（金曜日に公表）
    flow_data = [
        {
            "PublishedDate": datetime(2024, 1, 5).date(),  # 1/5(金)公表
            "Section": "TSEPrime",
            "ForeignersBalance": 100000000,
            "IndividualsBalance": -50000000,
            "flow_foreign_net_ratio": 0.2,
            "flow_smart_money_idx": 1.5
        },
        {
            "PublishedDate": datetime(2024, 1, 12).date(),  # 1/12(金)公表
            "Section": "TSEPrime",
            "ForeignersBalance": 150000000,
            "IndividualsBalance": -75000000,
            "flow_foreign_net_ratio": 0.3,
            "flow_smart_money_idx": 2.0
        },
        {
            "PublishedDate": datetime(2024, 1, 19).date(),  # 1/19(金)公表
            "Section": "TSEPrime",
            "ForeignersBalance": 80000000,
            "IndividualsBalance": -40000000,
            "flow_foreign_net_ratio": 0.15,
            "flow_smart_money_idx": 1.2
        }
    ]
    
    flow_df = pl.DataFrame(flow_data)
    
    return stock_df, flow_df

def test_current_join_method():
    """現在の結合方法（Date = effective_date の完全一致）"""
    stock_df, flow_df = create_sample_data()
    
    # 現在の方法: effective_date（T+1）を追加
    flow_df = flow_df.with_columns(
        (pl.col("PublishedDate") + timedelta(days=1)).alias("effective_date")
    )
    
    # 銘柄にSectionを付与（簡易版）
    stock_df = stock_df.with_columns(
        pl.lit("TSEPrime").alias("Section")
    )
    
    # 完全一致結合
    result = stock_df.join(
        flow_df.select(["effective_date", "flow_foreign_net_ratio", "flow_smart_money_idx"]),
        left_on="Date",
        right_on="effective_date",
        how="left"
    )
    
    logger.info("=== 現在の結合方法（完全一致） ===")
    logger.info(f"結合結果: {result.shape}")
    logger.info(f"フロー特徴量が付与された行数: {result['flow_foreign_net_ratio'].is_not_null().sum()}")
    print(result.head(10))
    
    return result

def test_improved_join_method():
    """改善版の結合方法（週次データの前方補完）"""
    stock_df, flow_df = create_sample_data()
    
    # 改善版: 有効期間を設定（金曜公表→翌月曜から次の金曜まで有効）
    flow_df = flow_df.with_columns([
        (pl.col("PublishedDate") + timedelta(days=3)).alias("effective_start"),  # 翌月曜
        (pl.col("PublishedDate") + timedelta(days=9)).alias("effective_end")     # 次の金曜
    ])
    
    # 銘柄にSectionを付与
    stock_df = stock_df.with_columns(
        pl.lit("TSEPrime").alias("Section")
    )
    
    # 範囲結合（各日付に対して有効なフローデータを使用）
    result = stock_df.clone()
    
    # フロー特徴量を初期化
    result = result.with_columns([
        pl.lit(None).cast(pl.Float64).alias("flow_foreign_net_ratio"),
        pl.lit(None).cast(pl.Float64).alias("flow_smart_money_idx")
    ])
    
    # 各フローデータの有効期間内の日付に値を設定
    for row in flow_df.iter_rows(named=True):
        mask = (
            (result["Date"] >= row["effective_start"]) &
            (result["Date"] <= row["effective_end"])
        )
        
        result = result.with_columns([
            pl.when(mask)
                .then(pl.lit(row["flow_foreign_net_ratio"]))
                .otherwise(pl.col("flow_foreign_net_ratio"))
                .alias("flow_foreign_net_ratio"),
            
            pl.when(mask)
                .then(pl.lit(row["flow_smart_money_idx"]))
                .otherwise(pl.col("flow_smart_money_idx"))
                .alias("flow_smart_money_idx")
        ])
    
    logger.info("\n=== 改善版の結合方法（前方補完） ===")
    logger.info(f"結合結果: {result.shape}")
    logger.info(f"フロー特徴量が付与された行数: {result['flow_foreign_net_ratio'].is_not_null().sum()}")
    print(result.head(10))
    
    return result

def compare_results():
    """両方法の結果を比較"""
    logger.info("\n" + "=" * 60)
    logger.info("週次フローデータ結合方法の比較")
    logger.info("=" * 60)
    
    current = test_current_join_method()
    improved = test_improved_join_method()
    
    logger.info("\n=== 比較結果 ===")
    logger.info(f"現在の方法: {current['flow_foreign_net_ratio'].is_not_null().sum()}/{len(current)} 行にデータあり")
    logger.info(f"改善版: {improved['flow_foreign_net_ratio'].is_not_null().sum()}/{len(improved)} 行にデータあり")
    
    # カバレッジの改善率
    current_coverage = current['flow_foreign_net_ratio'].is_not_null().sum() / len(current)
    improved_coverage = improved['flow_foreign_net_ratio'].is_not_null().sum() / len(improved)
    
    if current_coverage > 0:
        improvement = (improved_coverage - current_coverage) / current_coverage * 100
    else:
        improvement = float('inf') if improved_coverage > 0 else 0
    
    logger.info(f"\nカバレッジ改善率: {improvement:.1f}%")
    logger.info(f"現在: {current_coverage:.1%} → 改善版: {improved_coverage:.1%}")
    
    # 週ごとの詳細
    logger.info("\n=== 週ごとのカバレッジ ===")
    for week in range(3):
        week_start = datetime(2024, 1, 8) + timedelta(weeks=week)
        week_end = week_start + timedelta(days=4)
        
        current_week = current.filter(
            (pl.col("Date") >= week_start.date()) &
            (pl.col("Date") <= week_end.date())
        )
        improved_week = improved.filter(
            (pl.col("Date") >= week_start.date()) &
            (pl.col("Date") <= week_end.date())
        )
        
        logger.info(f"Week {week+1} ({week_start.date()} - {week_end.date()}):")
        logger.info(f"  現在: {current_week['flow_foreign_net_ratio'].is_not_null().sum()}/{len(current_week)} 行")
        logger.info(f"  改善: {improved_week['flow_foreign_net_ratio'].is_not_null().sum()}/{len(improved_week)} 行")

if __name__ == "__main__":
    compare_results()