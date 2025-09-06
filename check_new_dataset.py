#!/usr/bin/env python3
"""
新しく生成されたデータセットの特徴量を確認
"""

import polars as pl
from pathlib import Path

def check_dataset_features():
    # 最新のフルデータセットを取得
    dataset_file = Path("output/ml_dataset_20200906_20250906_20250906_215603_full.parquet")
    
    if not dataset_file.exists():
        print(f"Dataset not found: {dataset_file}")
        return
    
    print(f"Loading dataset: {dataset_file}")
    print(f"File size: {dataset_file.stat().st_size / (1024**3):.2f} GB\n")
    
    # データを読み込む
    df = pl.scan_parquet(str(dataset_file))
    
    # 全カラム名を取得
    all_cols = df.columns
    
    # カテゴリ別に分類
    flow_cols = [c for c in all_cols if c.startswith("flow_")]
    mkt_cols = [c for c in all_cols if c.startswith("mkt_")]
    stmt_cols = [c for c in all_cols if c.startswith("stmt_")]
    
    print("=" * 70)
    print("データセット特徴量サマリー")
    print("=" * 70)
    
    # 基本統計
    row_count = df.select(pl.len()).collect()[0, 0]
    col_count = len(all_cols)
    
    print(f"\n📊 基本統計:")
    print(f"  総レコード数: {row_count:,}")
    print(f"  総カラム数: {col_count}")
    
    # Trade-spec (flow) 特徴量
    print(f"\n💹 Trade-spec (flow_*) 特徴量: {len(flow_cols)}個")
    if flow_cols:
        print("  ", ", ".join(flow_cols[:10]))
        if len(flow_cols) > 10:
            print(f"  ... and {len(flow_cols) - 10} more")
    
    # TOPIX (mkt) 特徴量
    print(f"\n📈 TOPIX (mkt_*) 特徴量: {len(mkt_cols)}個")
    if mkt_cols:
        print("  ", ", ".join(mkt_cols[:10]))
        if len(mkt_cols) > 10:
            print(f"  ... and {len(mkt_cols) - 10} more")
    
    # Statements (stmt) 特徴量
    print(f"\n📄 Statements (stmt_*) 特徴量: {len(stmt_cols)}個")
    if stmt_cols:
        print("  ", ", ".join(stmt_cols[:10]))
        if len(stmt_cols) > 10:
            print(f"  ... and {len(stmt_cols) - 10} more")
    
    # 各カテゴリのカバレッジを確認
    print("\n" + "=" * 70)
    print("カバレッジ分析")
    print("=" * 70)
    
    # サンプルデータで確認（最初の10000行）
    sample_df = df.head(10000).collect()
    
    # Flow特徴量のカバレッジ
    if flow_cols:
        print("\n💹 Flow特徴量カバレッジ:")
        for col in flow_cols[:3]:  # 最初の3つだけチェック
            non_null = sample_df.filter(pl.col(col).is_not_null() & (pl.col(col) != 0)).height
            coverage = non_null / sample_df.height * 100
            print(f"  {col}: {coverage:.1f}%")
        
        # is_flow_validフラグがあれば確認
        if "is_flow_valid" in all_cols:
            valid_count = sample_df.filter(pl.col("is_flow_valid") == 1).height
            print(f"  全体カバレッジ (is_flow_valid): {valid_count / sample_df.height * 100:.1f}%")
    
    # Market特徴量のカバレッジ
    if mkt_cols:
        print("\n📈 Market特徴量カバレッジ:")
        for col in mkt_cols[:3]:  # 最初の3つだけチェック
            non_null = sample_df.filter(pl.col(col).is_not_null() & (pl.col(col) != 0)).height
            coverage = non_null / sample_df.height * 100
            print(f"  {col}: {coverage:.1f}%")
    
    # Statement特徴量のカバレッジ
    if stmt_cols:
        print("\n📄 Statement特徴量カバレッジ:")
        for col in stmt_cols[:3]:  # 最初の3つだけチェック
            non_null = sample_df.filter(pl.col(col).is_not_null() & (pl.col(col) != 0)).height
            coverage = non_null / sample_df.height * 100
            print(f"  {col}: {coverage:.1f}%")
        
        # is_stmt_validフラグがあれば確認
        if "is_stmt_valid" in all_cols:
            valid_count = sample_df.filter(pl.col("is_stmt_valid") == 1).height
            print(f"  全体カバレッジ (is_stmt_valid): {valid_count / sample_df.height * 100:.1f}%")
    
    # 特定の日付・銘柄でサンプル確認
    print("\n" + "=" * 70)
    print("サンプルデータ確認 (Code=1301, 2024-01-15)")
    print("=" * 70)
    
    sample = sample_df.filter(
        (pl.col("Code") == "1301") & 
        (pl.col("Date").cast(pl.Utf8) == "2024-01-15")
    )
    
    if sample.height > 0:
        row = sample.row(0, named=True)
        
        # Flow特徴量のサンプル
        if flow_cols:
            print("\n💹 Flow特徴量サンプル:")
            for col in flow_cols[:5]:
                if col in row:
                    print(f"  {col}: {row[col]}")
        
        # Market特徴量のサンプル
        if mkt_cols:
            print("\n📈 Market特徴量サンプル:")
            for col in mkt_cols[:5]:
                if col in row:
                    print(f"  {col}: {row[col]}")
        
        # Statement特徴量のサンプル
        if stmt_cols:
            print("\n📄 Statement特徴量サンプル:")
            for col in stmt_cols[:5]:
                if col in row:
                    print(f"  {col}: {row[col]}")
    else:
        print("サンプルデータが見つかりません")
    
    print("\n" + "=" * 70)
    print("✅ データセット確認完了")
    print("=" * 70)


if __name__ == "__main__":
    check_dataset_features()