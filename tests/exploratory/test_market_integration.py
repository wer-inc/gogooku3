#!/usr/bin/env python3
"""
Test market features integration with actual data
"""

import logging
import sys
from pathlib import Path

import polars as pl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(".")


def test_market_integration():
    """Test market features integration with existing dataset"""

    # Load existing dataset
    parquet_path = Path("data/processed/ml_dataset_20250831_031541.parquet")
    if not parquet_path.exists():
        logger.error(f"Dataset not found: {parquet_path}")
        return

    logger.info(f"Loading dataset from {parquet_path}")
    df = pl.read_parquet(parquet_path)

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {len(df.columns)}")

    # Sample some TOPIX data from existing dataset
    logger.info("\nCreating sample TOPIX data...")

    # Create a simple TOPIX proxy from average of all stocks
    topix_data = []
    for date in df["Date"].unique().sort():
        day_data = df.filter(pl.col("Date") == date)
        if not day_data.is_empty():
            avg_close = day_data["Close"].mean()
            if avg_close is not None:
                topix_data.append(
                    {
                        "Date": date,
                        "Close": avg_close * 10,  # Scale up to TOPIX range
                        "Open": avg_close * 10 * 0.995,
                        "High": avg_close * 10 * 1.01,
                        "Low": avg_close * 10 * 0.99,
                    }
                )

    topix_df = pl.DataFrame(topix_data)
    logger.info(f"TOPIX proxy shape: {topix_df.shape}")

    # Import market features module
    from src.features.market_features import (
        CrossMarketFeaturesGenerator,
        MarketFeaturesGenerator,
    )

    # Generate market features
    logger.info("\nGenerating market features...")
    market_gen = MarketFeaturesGenerator(z_score_window=60)
    market_features = market_gen.build_topix_features(topix_df)

    logger.info(f"Market features generated: {market_features.shape}")

    # Sample a subset for testing cross features
    sample_df = df.head(10000)  # Take first 10k rows for testing

    # Ensure required columns exist
    if "returns_1d" not in sample_df.columns:
        sample_df = sample_df.with_columns(
            [
                pl.col("Close").pct_change().over("Code").alias("returns_1d"),
                pl.col("Close").pct_change(5).over("Code").alias("returns_5d"),
            ]
        )

    # Generate cross features
    logger.info("\nGenerating cross features...")
    cross_gen = CrossMarketFeaturesGenerator()
    enhanced_df = cross_gen.attach_market_and_cross(sample_df, market_features)

    # Analysis
    logger.info("\n" + "=" * 60)
    logger.info("MARKET FEATURES INTEGRATION ANALYSIS")
    logger.info("=" * 60)

    # Count feature types
    original_cols = df.columns
    market_cols = [c for c in enhanced_df.columns if c.startswith("mkt_")]
    cross_cols = [
        "beta_60d",
        "alpha_1d",
        "alpha_5d",
        "rel_strength_5d",
        "trend_align_mkt",
        "alpha_vs_regime",
        "idio_vol_ratio",
        "beta_stability_60d",
    ]
    cross_cols = [c for c in cross_cols if c in enhanced_df.columns]

    logger.info(f"Original features: {len(original_cols)}")
    logger.info(f"Market features (mkt_*): {len(market_cols)}")
    logger.info(f"Cross features: {len(cross_cols)}")
    logger.info(f"Total features after enhancement: {len(enhanced_df.columns)}")
    logger.info(f"New features added: {len(enhanced_df.columns) - len(original_cols)}")

    # Sample market features
    logger.info("\n" + "=" * 60)
    logger.info("SAMPLE MARKET FEATURES")
    logger.info("=" * 60)

    for col in market_cols[:5]:
        non_null = enhanced_df.filter(pl.col(col).is_not_null())[col]
        if len(non_null) > 0:
            logger.info(f"{col}:")
            logger.info(f"  Mean: {non_null.mean():.6f}")
            logger.info(f"  Std: {non_null.std():.6f}")
            logger.info(f"  Min: {non_null.min():.6f}")
            logger.info(f"  Max: {non_null.max():.6f}")

    # Sample cross features
    logger.info("\n" + "=" * 60)
    logger.info("SAMPLE CROSS FEATURES")
    logger.info("=" * 60)

    for col in cross_cols[:3]:
        non_null = enhanced_df.filter(pl.col(col).is_not_null())[col]
        if len(non_null) > 0:
            logger.info(f"{col}:")
            logger.info(f"  Mean: {non_null.mean():.6f}")
            logger.info(f"  Std: {non_null.std():.6f}")
            logger.info(f"  Min: {non_null.min():.6f}")
            logger.info(f"  Max: {non_null.max():.6f}")

    # Check beta distribution
    if "beta_60d" in enhanced_df.columns:
        beta_non_null = enhanced_df.filter(pl.col("beta_60d").is_not_null())["beta_60d"]
        if len(beta_non_null) > 0:
            logger.info("\n" + "=" * 60)
            logger.info("BETA DISTRIBUTION ANALYSIS")
            logger.info("=" * 60)
            logger.info(f"Beta values calculated: {len(beta_non_null)}")
            logger.info(f"Beta mean: {beta_non_null.mean():.3f}")
            logger.info(f"Beta median: {beta_non_null.median():.3f}")
            logger.info(f"Beta < 0.5 (defensive): {(beta_non_null < 0.5).sum()} stocks")
            logger.info(
                f"Beta 0.5-1.5 (normal): {((beta_non_null >= 0.5) & (beta_non_null <= 1.5)).sum()} stocks"
            )
            logger.info(
                f"Beta > 1.5 (aggressive): {(beta_non_null > 1.5).sum()} stocks"
            )

    logger.info("\n✅ Market features integration test completed successfully!")

    return enhanced_df


if __name__ == "__main__":
    result = test_market_integration()
