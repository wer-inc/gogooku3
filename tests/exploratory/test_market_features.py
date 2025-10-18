#!/usr/bin/env python3
"""
Test market features integration
"""

import os
import sys

sys.path.append("/home/ubuntu/gogooku3-standalone")
os.chdir("/home/ubuntu/gogooku3-standalone")

import logging
from datetime import datetime

import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the market features module
from src.features.market_features import (
    CrossMarketFeaturesGenerator,
    MarketFeaturesGenerator,
)


def create_sample_data():
    """Create sample stock and TOPIX data for testing"""

    # Create date range
    dates = pl.date_range(
        datetime(2024, 1, 1), datetime(2024, 3, 31), interval="1d", eager=True
    )

    # Filter for business days only (Mon-Fri)
    business_dates = [d for d in dates if d.weekday() < 5]

    # Create TOPIX data
    np.random.seed(42)
    n_days = len(business_dates)

    # Generate TOPIX prices with realistic movement
    topix_returns = np.random.normal(0.0002, 0.01, n_days)
    topix_prices = 2000 * np.exp(np.cumsum(topix_returns))

    topix_df = pl.DataFrame(
        {
            "Date": business_dates,
            "Open": topix_prices * np.random.uniform(0.99, 1.01, n_days),
            "High": topix_prices * np.random.uniform(1.005, 1.02, n_days),
            "Low": topix_prices * np.random.uniform(0.98, 0.995, n_days),
            "Close": topix_prices,
        }
    )

    # Create stock data (3 stocks)
    stock_data = []
    for code in ["1001", "1002", "1003"]:
        # Each stock has different beta to market
        beta = np.random.uniform(0.5, 1.5)
        alpha = np.random.normal(0, 0.002)

        # Generate stock returns correlated with market
        stock_returns = (
            beta * topix_returns + alpha + np.random.normal(0, 0.015, n_days)
        )
        stock_prices = 1000 * np.exp(np.cumsum(stock_returns))

        for i, date in enumerate(business_dates):
            stock_data.append(
                {
                    "Code": code,
                    "Date": date,
                    "Close": stock_prices[i],
                    "Volume": np.random.randint(1000000, 5000000),
                }
            )

    stock_df = pl.DataFrame(stock_data)

    # Add basic features that cross features need
    stock_df = stock_df.sort(["Code", "Date"])
    stock_df = stock_df.with_columns(
        [
            pl.col("Close").pct_change().over("Code").alias("returns_1d"),
            pl.col("Close").pct_change(5).over("Code").alias("returns_5d"),
            pl.col("Close").pct_change(10).over("Code").alias("returns_10d"),
            pl.col("Close").pct_change(20).over("Code").alias("returns_20d"),
        ]
    )

    # Add MA features for cross feature calculation
    stock_df = stock_df.with_columns(
        [
            pl.col("Close").ewm_mean(span=5, adjust=False).over("Code").alias("ema_5"),
            pl.col("Close")
            .ewm_mean(span=20, adjust=False)
            .over("Code")
            .alias("ema_20"),
        ]
    )

    stock_df = stock_df.with_columns(
        [
            ((pl.col("ema_5") - pl.col("ema_20")) / pl.col("ema_20")).alias(
                "ma_gap_5_20"
            ),
            pl.col("Close")
            .pct_change()
            .over("Code")
            .rolling_std(20)
            .alias("volatility_20d"),
        ]
    )

    return stock_df, topix_df


def test_market_features():
    """Test the market features generation"""

    logger.info("Creating sample data...")
    stock_df, topix_df = create_sample_data()

    logger.info(f"Stock data shape: {stock_df.shape}")
    logger.info(f"TOPIX data shape: {topix_df.shape}")

    # Test MarketFeaturesGenerator
    logger.info("\n" + "=" * 60)
    logger.info("Testing MarketFeaturesGenerator...")
    logger.info("=" * 60)

    market_gen = MarketFeaturesGenerator(z_score_window=60)  # Shorter window for test
    market_features = market_gen.build_topix_features(topix_df)

    logger.info(f"Market features shape: {market_features.shape}")
    logger.info(f"Market feature columns: {market_features.columns[:10]}...")

    # Check specific market features exist
    expected_market_cols = [
        "mkt_ret_1d",
        "mkt_ret_5d",
        "mkt_vol_20d",
        "mkt_ema_20",
        "mkt_bb_bw",
        "mkt_dd_from_peak",
        "mkt_bull_200",
        "mkt_trend_up",
    ]

    for col in expected_market_cols:
        if col in market_features.columns:
            logger.info(f"✅ {col} exists")
        else:
            logger.warning(f"❌ {col} missing")

    # Test CrossMarketFeaturesGenerator
    logger.info("\n" + "=" * 60)
    logger.info("Testing CrossMarketFeaturesGenerator...")
    logger.info("=" * 60)

    cross_gen = CrossMarketFeaturesGenerator()
    enhanced_df = cross_gen.attach_market_and_cross(stock_df, market_features)

    logger.info(f"Enhanced data shape: {enhanced_df.shape}")

    # Check cross features exist
    expected_cross_cols = [
        "beta_60d",
        "alpha_1d",
        "alpha_5d",
        "rel_strength_5d",
        "trend_align_mkt",
        "alpha_vs_regime",
        "idio_vol_ratio",
        "beta_stability_60d",
    ]

    for col in expected_cross_cols:
        if col in enhanced_df.columns:
            logger.info(f"✅ {col} exists")
            # Show sample values
            sample = enhanced_df.filter(pl.col(col).is_not_null()).select(col).head(5)
            if not sample.is_empty():
                values = sample[col].to_list()
                logger.info(
                    f"   Sample values: {[f'{v:.4f}' if v is not None else 'null' for v in values]}"
                )
        else:
            logger.warning(f"❌ {col} missing")

    # Summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("Feature Summary")
    logger.info("=" * 60)

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

    logger.info(f"Total features: {len(enhanced_df.columns)}")
    logger.info(f"Market features (mkt_*): {len(market_cols)}")
    logger.info(f"Cross features: {len(cross_cols)}")
    logger.info(f"Original features: {len(stock_df.columns)}")
    logger.info(
        f"New features added: {len(enhanced_df.columns) - len(stock_df.columns)}"
    )

    # Check for null values
    logger.info("\n" + "=" * 60)
    logger.info("Data Quality Check")
    logger.info("=" * 60)

    for col in market_cols[:5] + cross_cols[:3]:
        if col in enhanced_df.columns:
            null_count = enhanced_df[col].is_null().sum()
            null_pct = null_count / len(enhanced_df) * 100
            logger.info(f"{col}: {null_count} nulls ({null_pct:.1f}%)")

    logger.info("\n✅ Market features test completed successfully!")

    return enhanced_df


if __name__ == "__main__":
    result_df = test_market_features()
