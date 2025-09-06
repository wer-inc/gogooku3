#!/usr/bin/env python3
"""
実際のデータセットで財務諸表カバレッジを確認
"""

import polars as pl
from pathlib import Path
import glob

def analyze_statement_coverage():
    """財務諸表データのカバレッジを分析"""
    
    # outputディレクトリのparquetファイルを探す
    output_dir = Path("/home/ubuntu/gogooku3-standalone/output")
    
    # 最新のml_datasetファイルを探す
    dataset_files = sorted(glob.glob(str(output_dir / "ml_dataset*.parquet")))
    
    if not dataset_files:
        print("No dataset files found in output/")
        return
    
    # 最新のファイルを読み込む
    latest_file = dataset_files[-1]
    print(f"Loading: {latest_file}")
    
    df = pl.scan_parquet(latest_file)
    
    # stmt_* カラムを探す
    stmt_cols = [col for col in df.columns if col.startswith("stmt_")]
    
    if not stmt_cols:
        print("No statement features found in dataset")
        return
    
    print("=== 財務諸表特徴量のカバレッジ分析 ===\n")
    
    # カラム数を表示
    print(f"1. 財務諸表特徴量: {len(stmt_cols)}個")
    print("-" * 50)
    for i, col in enumerate(stmt_cols, 1):
        print(f"  {i:2d}. {col}")
    
    # カバレッジを計算
    print("\n2. データカバレッジ:")
    print("-" * 50)
    
    # is_stmt_validフラグがあればそれを使用
    if "is_stmt_valid" in df.columns:
        valid_count = df.filter(pl.col("is_stmt_valid") == 1).select(pl.len()).collect()[0, 0]
        total_count = df.select(pl.len()).collect()[0, 0]
        coverage = valid_count / total_count * 100
        
        print(f"総レコード数: {total_count:,}")
        print(f"財務データ有効レコード: {valid_count:,}")
        print(f"カバレッジ: {coverage:.2f}%")
    
    # stmt_days_since_statementの分布を確認
    if "stmt_days_since_statement" in df.columns:
        print("\n3. 決算からの経過日数分布:")
        print("-" * 50)
        
        days_stats = df.filter(pl.col("stmt_days_since_statement") >= 0).select([
            pl.col("stmt_days_since_statement").min().alias("min"),
            pl.col("stmt_days_since_statement").max().alias("max"),
            pl.col("stmt_days_since_statement").mean().alias("mean"),
            pl.col("stmt_days_since_statement").median().alias("median"),
            pl.col("stmt_days_since_statement").quantile(0.25).alias("q25"),
            pl.col("stmt_days_since_statement").quantile(0.75).alias("q75")
        ]).collect()
        
        print(f"最小: {days_stats[0, 'min']} 日")
        print(f"最大: {days_stats[0, 'max']} 日")
        print(f"平均: {days_stats[0, 'mean']:.1f} 日")
        print(f"中央値: {days_stats[0, 'median']} 日")
        print(f"第1四分位: {days_stats[0, 'q25']} 日")
        print(f"第3四分位: {days_stats[0, 'q75']} 日")
        
        # 日数範囲別の分布
        print("\n4. 経過日数範囲別分布:")
        print("-" * 50)
        
        ranges = [
            (0, 7, "1週間以内"),
            (8, 30, "1ヶ月以内"),
            (31, 60, "2ヶ月以内"),
            (61, 90, "3ヶ月以内"),
            (91, 180, "6ヶ月以内"),
            (181, 365, "1年以内"),
            (366, 9999, "1年超")
        ]
        
        for min_days, max_days, label in ranges:
            count = df.filter(
                (pl.col("stmt_days_since_statement") >= min_days) &
                (pl.col("stmt_days_since_statement") <= max_days)
            ).select(pl.len()).collect()[0, 0]
            
            if count > 0:
                pct = count / total_count * 100
                print(f"{label:12s}: {count:10,} ({pct:5.2f}%)")
    
    # インパルス（開示当日）の統計
    if "stmt_imp_statement" in df.columns:
        print("\n5. 決算開示当日のレコード:")
        print("-" * 50)
        
        impulse_count = df.filter(pl.col("stmt_imp_statement") == 1).select(pl.len()).collect()[0, 0]
        print(f"開示当日レコード数: {impulse_count:,}")
        print(f"開示当日の割合: {impulse_count / total_count * 100:.4f}%")
        print(f"（四半期決算なら約1/60～1/90が期待値）")


if __name__ == "__main__":
    analyze_statement_coverage()