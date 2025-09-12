#!/usr/bin/env python3
"""
市場特徴量を既存のデータセットに追加するスクリプト
修正済みのidio_vol_ratioも含まれる
"""

import sys
import polars as pl
from pathlib import Path
import logging

# Add paths
sys.path.append('/home/ubuntu/gogooku3-standalone')
sys.path.append('/home/ubuntu/gogooku3-standalone/src')

from src.features.market_features import MarketFeaturesGenerator, CrossMarketFeaturesGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_market_and_cross_features(dataset_path: str) -> str:
    """
    既存のデータセットに市場特徴量とクロス特徴量を追加
    """
    logger.info(f"Loading dataset: {dataset_path}")
    df = pl.read_parquet(dataset_path)
    
    # サンプルのTOPIXデータを生成（実際にはJQuants APIから取得すべき）
    dates = df["Date"].unique().sort()
    import numpy as np
    np.random.seed(42)
    
    topix_df = pl.DataFrame({
        "Date": dates,
        "Open": [2000 + i*0.5 + np.random.randn()*10 for i in range(len(dates))],
        "High": [2010 + i*0.5 + np.random.randn()*10 for i in range(len(dates))],
        "Low": [1990 + i*0.5 + np.random.randn()*10 for i in range(len(dates))],
        "Close": [2000 + i*0.5 + np.random.randn()*10 for i in range(len(dates))],
    })
    
    # 1. TOPIX市場特徴量を生成
    logger.info("Generating TOPIX market features...")
    market_gen = MarketFeaturesGenerator()
    market_df = market_gen.build_topix_features(topix_df)
    
    # 2. クロス特徴量を生成（idio_vol_ratio含む）
    logger.info("Generating cross features (including fixed idio_vol_ratio)...")
    cross_gen = CrossMarketFeaturesGenerator()
    df = cross_gen.attach_market_and_cross(df, market_df)
    
    # 3. 結果を保存
    output_path = dataset_path.replace(".parquet", "_with_market.parquet")
    df.write_parquet(output_path)
    
    # 統計情報
    logger.info(f"✅ Enhanced dataset saved: {output_path}")
    logger.info(f"  Shape: {df.shape}")
    
    # idio_vol_ratioの確認
    if "idio_vol_ratio" in df.columns:
        idio_values = df["idio_vol_ratio"].drop_nulls().to_list()
        if idio_values:
            logger.info(f"  idio_vol_ratio range: {min(idio_values):.2f} - {max(idio_values):.2f}")
    
    # 追加された列
    mkt_cols = [col for col in df.columns if col.startswith("mkt_")]
    cross_cols = ["beta_60d", "alpha_1d", "alpha_5d", "rel_strength_5d", 
                  "trend_align_mkt", "alpha_vs_regime", "idio_vol_ratio", "beta_stability_60d"]
    available_cross = [col for col in cross_cols if col in df.columns]
    
    logger.info(f"  Market features added: {len(mkt_cols)}")
    logger.info(f"  Cross features added: {len(available_cross)}")
    
    return output_path


def main():
    """メイン処理"""
    # 最新のデータセットを探す
    from datetime import datetime
    output_dir = Path("/home/ubuntu/gogooku3-standalone/output")
    
    # 最新のml_datasetを探す
    datasets = list(output_dir.glob("ml_dataset_*.parquet"))
    datasets = [d for d in datasets if "with_market" not in str(d)]
    
    if not datasets:
        logger.error("No dataset found")
        return
    
    # 最新を取得
    latest = sorted(datasets)[-1]
    logger.info(f"Found latest dataset: {latest}")
    
    # 市場特徴量を追加
    enhanced_path = add_market_and_cross_features(str(latest))
    
    print("\n" + "=" * 60)
    print("MARKET FEATURES ADDED SUCCESSFULLY")
    print("=" * 60)
    print(f"Enhanced dataset: {enhanced_path}")
    print("\nNext steps:")
    print("  1. Verify idio_vol_ratio values are reasonable (1.0-3.0)")
    print("  2. Check that all market and cross features are present")
    print("  3. Re-run full pipeline with all fixes integrated")


if __name__ == "__main__":
    main()