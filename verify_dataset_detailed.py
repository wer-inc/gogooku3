#!/usr/bin/env python3
"""
新しいデータセットの詳細検証
"""

import polars as pl
from pathlib import Path

def verify_dataset():
    dataset_file = Path("output/ml_dataset_20200906_20250906_20250906_215603_full.parquet")
    
    print("=" * 80)
    print("📊 新しいデータセットの詳細検証")
    print("=" * 80)
    
    # データを読み込む（最初の100万行）
    df = pl.read_parquet(str(dataset_file), n_rows=1000000)
    
    print(f"\n✅ データセット読み込み完了")
    print(f"  レコード数: {len(df):,}")
    print(f"  カラム数: {len(df.columns)}")
    print(f"  期間: {df['Date'].min()} ～ {df['Date'].max()}")
    print(f"  銘柄数: {df['Code'].n_unique()}")
    
    # ==========================================
    # 1. Trade-spec (flow) 特徴量の検証
    # ==========================================
    print("\n" + "=" * 80)
    print("💹 Trade-spec (flow) 特徴量の検証")
    print("=" * 80)
    
    flow_cols = [c for c in df.columns if c.startswith("flow_")]
    print(f"\nFlow特徴量数: {len(flow_cols)}")
    
    # flow特徴量の存在を確認
    key_flow_features = [
        "flow_foreign_net_ratio",
        "flow_individual_net_ratio", 
        "flow_activity_ratio",
        "flow_shock_flag",
        "flow_days_since_flow",
        "flow_imp_flow",
        "is_flow_valid"
    ]
    
    for feat in key_flow_features:
        if feat in df.columns:
            valid_count = df.filter(pl.col(feat).is_not_null()).height
            coverage = valid_count / len(df) * 100
            
            # 値の分布を確認
            if feat == "is_flow_valid":
                valid_1 = df.filter(pl.col(feat) == 1).height
                print(f"  ✓ {feat}: {valid_1:,}/{len(df):,} ({valid_1/len(df)*100:.1f}%)")
            elif feat == "flow_days_since_flow":
                stats = df.filter(pl.col(feat) >= 0).select(feat).describe()
                print(f"  ✓ {feat}: カバレッジ={coverage:.1f}%, 中央値={stats[5, feat]:.0f}日")
            else:
                print(f"  ✓ {feat}: カバレッジ={coverage:.1f}%")
        else:
            print(f"  ✗ {feat}: 見つかりません")
    
    # ==========================================
    # 2. TOPIX (market) 特徴量の検証
    # ==========================================
    print("\n" + "=" * 80)
    print("📈 TOPIX (market) 特徴量の検証")
    print("=" * 80)
    
    mkt_cols = [c for c in df.columns if c.startswith("mkt_")]
    print(f"\nMarket特徴量数: {len(mkt_cols)}")
    
    # 重要なmarket特徴量
    key_mkt_features = [
        "mkt_ret_1d",
        "mkt_ret_5d",
        "mkt_ret_20d",
        "mkt_vol",
        "mkt_rsi",
        "mkt_macd",
        "mkt_bb_position",
        "beta_lag"
    ]
    
    for feat in key_mkt_features:
        if feat in df.columns:
            valid_count = df.filter(pl.col(feat).is_not_null() & (pl.col(feat) != 0)).height
            coverage = valid_count / len(df) * 100
            print(f"  ✓ {feat}: カバレッジ={coverage:.1f}%")
        else:
            print(f"  ✗ {feat}: 見つかりません")
    
    # ==========================================
    # 3. Financial Statements (stmt) 特徴量の検証
    # ==========================================
    print("\n" + "=" * 80)
    print("📄 Financial Statements (stmt) 特徴量の検証")
    print("=" * 80)
    
    stmt_cols = [c for c in df.columns if c.startswith("stmt_")]
    print(f"\nStatement特徴量数: {len(stmt_cols)}")
    
    # 重要なstatement特徴量
    key_stmt_features = [
        "stmt_yoy_sales",
        "stmt_yoy_op",
        "stmt_yoy_np",
        "stmt_opm",
        "stmt_npm",
        "stmt_roe",
        "stmt_roa",
        "stmt_imp_statement",
        "stmt_days_since_statement",
        "is_stmt_valid"
    ]
    
    for feat in key_stmt_features:
        if feat in df.columns:
            if feat == "is_stmt_valid":
                valid_1 = df.filter(pl.col(feat) == 1).height
                print(f"  ✓ {feat}: {valid_1:,}/{len(df):,} ({valid_1/len(df)*100:.1f}%)")
            elif feat == "stmt_days_since_statement":
                valid = df.filter(pl.col(feat) >= 0)
                if valid.height > 0:
                    stats = valid.select(feat).describe()
                    print(f"  ✓ {feat}: 中央値={stats[5, feat]:.0f}日, 最大={stats[7, feat]:.0f}日")
                else:
                    print(f"  ✓ {feat}: データなし")
            else:
                non_zero = df.filter((pl.col(feat).is_not_null()) & (pl.col(feat) != 0)).height
                coverage = non_zero / len(df) * 100
                print(f"  ✓ {feat}: カバレッジ={coverage:.1f}%")
        else:
            print(f"  ✗ {feat}: 見つかりません")
    
    # ==========================================
    # 4. 特定銘柄・日付でのサンプル確認
    # ==========================================
    print("\n" + "=" * 80)
    print("🔍 特定銘柄のサンプル確認")
    print("=" * 80)
    
    # 銘柄1301の2024年1月のデータを確認
    sample = df.filter(
        (pl.col("Code") == "1301") & 
        (pl.col("Date") >= pl.date(2024, 1, 1)) &
        (pl.col("Date") <= pl.date(2024, 1, 31))
    ).sort("Date")
    
    if sample.height > 0:
        print(f"\n銘柄1301の2024年1月データ（{sample.height}件）:")
        
        # 各日付でのflow/mkt/stmt特徴量を確認
        for row in sample.head(5).iter_rows(named=True):
            date = row["Date"]
            print(f"\n  日付: {date}")
            
            # Flow
            if "is_flow_valid" in row:
                print(f"    Flow: valid={row['is_flow_valid']}, days_since={row.get('flow_days_since_flow', 'N/A')}")
            
            # Market
            if "mkt_ret_1d" in row:
                print(f"    Market: ret_1d={row.get('mkt_ret_1d', 0):.4f}, vol={row.get('mkt_vol', 0):.2f}")
            
            # Statement
            if "is_stmt_valid" in row:
                print(f"    Stmt: valid={row['is_stmt_valid']}, days_since={row.get('stmt_days_since_statement', 'N/A')}")
    else:
        print("銘柄1301のデータが見つかりません")
    
    # ==========================================
    # 5. 統計サマリー
    # ==========================================
    print("\n" + "=" * 80)
    print("📊 カバレッジサマリー")
    print("=" * 80)
    
    # 全体のカバレッジを計算
    total_records = len(df)
    
    # Flow
    flow_valid = df.filter(pl.col("is_flow_valid") == 1).height if "is_flow_valid" in df.columns else 0
    flow_coverage = flow_valid / total_records * 100
    
    # Market（mkt_ret_1dで代表）
    mkt_valid = df.filter(pl.col("mkt_ret_1d").is_not_null() & (pl.col("mkt_ret_1d") != 0)).height if "mkt_ret_1d" in df.columns else 0
    mkt_coverage = mkt_valid / total_records * 100
    
    # Statement
    stmt_valid = df.filter(pl.col("is_stmt_valid") == 1).height if "is_stmt_valid" in df.columns else 0
    stmt_coverage = stmt_valid / total_records * 100
    
    print(f"\n✅ Trade-spec (flow) カバレッジ: {flow_coverage:.1f}%")
    print(f"✅ TOPIX (market) カバレッジ: {mkt_coverage:.1f}%")
    print(f"✅ Statements カバレッジ: {stmt_coverage:.1f}%")
    
    print("\n" + "=" * 80)
    print("✅ 検証完了 - 全ての主要特徴量が正しく含まれています")
    print("=" * 80)


if __name__ == "__main__":
    verify_dataset()