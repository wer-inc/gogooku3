#!/usr/bin/env python3
"""
Phase 2特徴量追加スクリプト

GAT修正と組み合わせて使用する、セクター・市場指数特徴量の追加。

Usage:
    python scripts/pipelines/add_phase2_features.py
    python scripts/pipelines/add_phase2_features.py --input output/ml_dataset_latest_full.parquet
"""
import argparse
import logging
from pathlib import Path

import polars as pl

# gogooku3パッケージからセクター集約特徴をインポート
from gogooku3.features.sector_aggregation import add_sector_aggregation_features

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def add_market_index_features(df: pl.DataFrame) -> pl.DataFrame:
    """市場指数特徴量を追加（TOPIX beta等）"""
    logger.info("Adding market index features...")

    # TOPIX関連のカラムがあるか確認
    topix_cols = [c for c in df.columns if "topix" in c.lower()]
    if not topix_cols:
        logger.warning("No TOPIX columns found. Skipping market index features.")
        return df

    # returns_1dカラムを探す
    ret_col = None
    for candidate in ["returns_1d", "feat_ret_1d", "ret_1d"]:
        if candidate in df.columns:
            ret_col = candidate
            break

    if ret_col is None:
        logger.warning("No returns column found. Skipping market index features.")
        return df

    # TOPIX retカラムを探す
    topix_ret_col = None
    for candidate in ["topix_ret_1d", "topix_return_1d", "TOPIX_ret_1d"]:
        if candidate in df.columns:
            topix_ret_col = candidate
            break

    if topix_ret_col is None:
        logger.warning("No TOPIX return column found. Skipping market index features.")
        return df

    logger.info(f"Using {ret_col} and {topix_ret_col} for market features")

    # 60日TOPIX beta
    df = df.with_columns(
        [
            (
                pl.col(ret_col).rolling_cov(pl.col(topix_ret_col), 60)
                / (pl.col(topix_ret_col).rolling_var(60) + 1e-8)
            ).alias("beta_topix_60"),
        ]
    )

    # 20日市場連動度（相関）
    df = df.with_columns(
        [
            pl.col(ret_col)
            .rolling_corr(pl.col(topix_ret_col), 20)
            .alias("corr_topix_20"),
        ]
    )

    # 市場超過リターン（alpha）
    df = df.with_columns(
        [
            (pl.col(ret_col) - pl.col(topix_ret_col)).alias("excess_ret_topix_1d"),
        ]
    )

    logger.info(
        "✅ Market index features added: beta_topix_60, corr_topix_20, excess_ret_topix_1d"
    )
    return df


def enrich_with_phase2_features(df: pl.DataFrame) -> pl.DataFrame:
    """Phase 2特徴量を追加（セクター+市場指数）"""
    logger.info("=" * 60)
    logger.info("Phase 2特徴量追加開始")
    logger.info(f"入力データ: {len(df):,} rows, {len(df.columns)} columns")
    logger.info("=" * 60)

    # 1. セクター集約特徴（~30列）
    try:
        df = add_sector_aggregation_features(df, min_members=5)
        logger.info("✅ Sector aggregation features added")
    except Exception as e:
        logger.error(f"Failed to add sector features: {e}")

    # 2. 市場指数特徴（~3列）
    try:
        df = add_market_index_features(df)
        logger.info("✅ Market index features added")
    except Exception as e:
        logger.error(f"Failed to add market index features: {e}")

    logger.info("=" * 60)
    logger.info(f"出力データ: {len(df):,} rows, {len(df.columns)} columns")
    logger.info(
        f"追加特徴量: {len(df.columns)} columns (入力から +{len(df.columns) - len(df.columns)} 列増加)"
    )
    logger.info("=" * 60)

    return df


def main():
    parser = argparse.ArgumentParser(description="Phase 2特徴量追加")
    parser.add_argument(
        "--input",
        type=str,
        default="output/ml_dataset_latest_full.parquet",
        help="入力データセットパス",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/ml_dataset_phase2_enriched.parquet",
        help="出力データセットパス",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    # データセット読み込み
    logger.info(f"📂 Loading dataset from: {input_path}")
    df = pl.read_parquet(input_path)

    # Phase 2特徴量追加
    df_enriched = enrich_with_phase2_features(df)

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"💾 Saving enriched dataset to: {output_path}")
    df_enriched.write_parquet(output_path)

    logger.info("✅ Phase 2 features added successfully!")
    logger.info(f"   Output: {output_path}")
    logger.info(f"   Shape: {df_enriched.shape}")

    # 追加された特徴量のサンプル表示
    new_cols = [c for c in df_enriched.columns if c not in df.columns]
    if new_cols:
        logger.info(f"   New features ({len(new_cols)}): {', '.join(new_cols[:10])}...")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
