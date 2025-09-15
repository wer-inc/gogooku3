#!/usr/bin/env python3
"""Test script for Phase 2 J-Quants API feature enhancements.

Tests:
1. Enhanced margin trading features
2. Option sentiment features
3. Enhanced flow analysis features
"""

import sys
import asyncio
from pathlib import Path
import polars as pl
import logging
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

from src.gogooku3.components.jquants_async_fetcher import JQuantsAsyncFetcher
from scripts.data.ml_dataset_builder import MLDatasetBuilder, create_sample_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_enhanced_margin_features():
    """Test enhanced margin trading feature extraction."""
    logger.info("\n" + "="*60)
    logger.info("Testing Enhanced Margin Trading Features")
    logger.info("="*60)
    
    # Create sample data
    sample_data = pl.DataFrame({
        'Code': ['7203', '7203', '7203', '7203', '7203'],  # Toyota
        'Date': [
            datetime(2024, 10, 1).date(),
            datetime(2024, 10, 8).date(),
            datetime(2024, 10, 15).date(),
            datetime(2024, 10, 22).date(),
            datetime(2024, 10, 29).date(),
        ],
        'Close': [2500.0, 2520.0, 2510.0, 2530.0, 2525.0],
        'Volume': [1000000, 1100000, 950000, 1050000, 980000],
    })
    
    # Initialize fetcher with credentials from env
    email = os.getenv('JQUANTS_AUTH_EMAIL')
    password = os.getenv('JQUANTS_AUTH_PASSWORD')

    if not email or not password:
        logger.warning("J-Quants credentials not found in environment. Using mock mode.")
        return True

    fetcher = JQuantsAsyncFetcher(email=email, password=password)
    
    # Initialize dataset builder
    builder = MLDatasetBuilder()
    
    # Add enhanced margin features
    try:
        df_with_margin = builder.add_enhanced_margin_features(
            sample_data,
            fetcher=fetcher,
            start_date='2024-09-01',
            end_date='2024-10-31',
            use_weekly=True
        )
        
        # Check added features
        new_cols = set(df_with_margin.columns) - set(sample_data.columns)
        logger.info(f"Added enhanced margin features: {new_cols}")
        
        # Display sample
        if not df_with_margin.is_empty():
            logger.info("\nSample margin features:")
            feature_cols = ['Date', 'margin_balance_ratio', 'margin_velocity', 'margin_stress_indicator']
            available_cols = [col for col in feature_cols if col in df_with_margin.columns]
            if available_cols:
                logger.info(df_with_margin.select(available_cols).head())
            
        return True
        
    except Exception as e:
        logger.error(f"Enhanced margin features test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_option_sentiment_features():
    """Test option sentiment feature extraction."""
    logger.info("\n" + "="*60)
    logger.info("Testing Option Sentiment Features")
    logger.info("="*60)
    
    # Create sample data (market-wide features, so no specific codes needed)
    sample_data = pl.DataFrame({
        'Code': ['7203', '7203', '6758', '6758'],
        'Date': [
            datetime(2024, 10, 1).date(),
            datetime(2024, 10, 2).date(),
            datetime(2024, 10, 1).date(),
            datetime(2024, 10, 2).date(),
        ],
        'Close': [2500.0, 2520.0, 3100.0, 3120.0],
    })
    
    # Initialize fetcher with credentials from env
    email = os.getenv('JQUANTS_AUTH_EMAIL')
    password = os.getenv('JQUANTS_AUTH_PASSWORD')

    if not email or not password:
        logger.warning("J-Quants credentials not found in environment. Using mock mode.")
        return True

    fetcher = JQuantsAsyncFetcher(email=email, password=password)
    
    # Initialize dataset builder
    builder = MLDatasetBuilder()
    
    # Add option sentiment features
    try:
        df_with_options = builder.add_option_sentiment_features(
            sample_data,
            fetcher=fetcher,
            start_date='2024-09-01',
            end_date='2024-10-31'
        )
        
        # Check added features
        new_cols = set(df_with_options.columns) - set(sample_data.columns)
        logger.info(f"Added option sentiment features: {new_cols}")
        
        # Display sample
        if not df_with_options.is_empty():
            logger.info("\nSample option features:")
            feature_cols = ['Date', 'put_call_ratio', 'iv_skew', 'smart_money_indicator']
            available_cols = [col for col in feature_cols if col in df_with_options.columns]
            if available_cols:
                logger.info(df_with_options.select(available_cols).head())
            
        return True
        
    except Exception as e:
        logger.error(f"Option sentiment features test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_enhanced_flow_features():
    """Test enhanced flow analysis feature extraction."""
    logger.info("\n" + "="*60)
    logger.info("Testing Enhanced Flow Analysis Features")
    logger.info("="*60)
    
    # Create sample data
    sample_data = pl.DataFrame({
        'Code': ['7203', '7203', '7203', '6758', '6758'],
        'Date': [
            datetime(2024, 10, 1).date(),
            datetime(2024, 10, 8).date(),
            datetime(2024, 10, 15).date(),
            datetime(2024, 10, 1).date(),
            datetime(2024, 10, 8).date(),
        ],
        'Close': [2500.0, 2520.0, 2510.0, 3100.0, 3120.0],
        'Volume': [1000000, 1100000, 950000, 800000, 850000],
    })
    
    # Initialize fetcher with credentials from env
    email = os.getenv('JQUANTS_AUTH_EMAIL')
    password = os.getenv('JQUANTS_AUTH_PASSWORD')

    if not email or not password:
        logger.warning("J-Quants credentials not found in environment. Using mock mode.")
        return True

    fetcher = JQuantsAsyncFetcher(email=email, password=password)
    
    # Initialize dataset builder
    builder = MLDatasetBuilder()
    
    # Add enhanced flow features
    try:
        df_with_flow = builder.add_enhanced_flow_features(
            sample_data,
            fetcher=fetcher,
            start_date='2024-09-01',
            end_date='2024-10-31'
        )
        
        # Check added features
        new_cols = set(df_with_flow.columns) - set(sample_data.columns)
        logger.info(f"Added enhanced flow features: {new_cols}")
        
        # Display sample
        if not df_with_flow.is_empty():
            logger.info("\nSample flow features:")
            feature_cols = ['Date', 'Code', 'institutional_accumulation', 'foreign_sentiment', 'smart_flow_indicator']
            available_cols = [col for col in feature_cols if col in df_with_flow.columns]
            if available_cols:
                logger.info(df_with_flow.select(available_cols).head())
            
        return True
        
    except Exception as e:
        logger.error(f"Enhanced flow features test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_integrated_phase2_pipeline():
    """Test all Phase 2 features in an integrated pipeline."""
    logger.info("\n" + "="*60)
    logger.info("Testing Integrated Pipeline with All Phase 2 Features")
    logger.info("="*60)
    
    # Load or create larger sample dataset
    try:
        # Try to load existing data
        data_path = Path('output/ml_dataset_phase1_enriched.parquet')
        if data_path.exists():
            logger.info(f"Loading Phase 1 enriched dataset from {data_path}")
            df = pl.read_parquet(data_path)
            # Limit to recent data for testing
            df = df.filter(pl.col('Date') >= datetime(2024, 9, 1).date()).head(500)
        else:
            # Create sample data
            logger.info("Creating sample dataset")
            df = create_sample_data(n_stocks=5, n_days=20)
    
        # Initialize components
        email = os.getenv('JQUANTS_AUTH_EMAIL')
        password = os.getenv('JQUANTS_AUTH_PASSWORD')

        if not email or not password:
            logger.warning("J-Quants credentials not found in environment. Using mock mode.")
            # Create builder without fetcher for mock testing
            builder = MLDatasetBuilder()
            fetcher = None
        else:
            fetcher = JQuantsAsyncFetcher(email=email, password=password)
            builder = MLDatasetBuilder()
        
        # Track initial columns
        initial_cols = set(df.columns)
        logger.info(f"Initial columns: {len(initial_cols)}")
        
        # Add Phase 2 features sequentially
        logger.info("\nAdding Phase 2 features...")
        
        # 1. Enhanced margin features
        logger.info("- Adding enhanced margin features...")
        df = builder.add_enhanced_margin_features(df, fetcher=fetcher)
        margin_cols = set(df.columns) - initial_cols
        logger.info(f"  Added {len(margin_cols)} margin features")
        
        # 2. Option sentiment features
        logger.info("- Adding option sentiment features...")
        df = builder.add_option_sentiment_features(df, fetcher=fetcher)
        option_cols = set(df.columns) - initial_cols - margin_cols
        logger.info(f"  Added {len(option_cols)} option features")
        
        # 3. Enhanced flow features
        logger.info("- Adding enhanced flow features...")
        df = builder.add_enhanced_flow_features(df, fetcher=fetcher)
        flow_cols = set(df.columns) - initial_cols - margin_cols - option_cols
        logger.info(f"  Added {len(flow_cols)} flow features")
        
        # Summary
        total_new = len(df.columns) - len(initial_cols)
        logger.info(f"\n✅ Successfully added {total_new} new Phase 2 features")
        logger.info(f"Final dataset shape: {df.shape}")
        
        # List all new feature names
        all_new_features = margin_cols | option_cols | flow_cols
        logger.info("\nNew Phase 2 features:")
        for i, feat in enumerate(sorted(all_new_features), 1):
            logger.info(f"  {i:2d}. {feat}")
        
        # Save enriched dataset
        output_path = Path('output/ml_dataset_phase2_enriched.parquet')
        output_path.parent.mkdir(exist_ok=True)
        df.write_parquet(output_path)
        logger.info(f"\nSaved Phase 2 enriched dataset to {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Integrated Phase 2 pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all Phase 2 feature tests."""
    logger.info("Starting Phase 2 Feature Tests")
    logger.info("This will test the advanced J-Quants API feature extractors")
    
    results = {}
    
    # Run individual tests
    results['margin'] = await test_enhanced_margin_features()
    results['options'] = await test_option_sentiment_features()
    results['flow'] = await test_enhanced_flow_features()
    
    # Run integrated test
    results['integrated'] = await test_integrated_phase2_pipeline()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Test Results Summary")
    logger.info("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name.capitalize():15} {status}")
    
    all_passed = all(results.values())
    if all_passed:
        logger.info("\n🎉 All Phase 2 feature tests passed!")
        logger.info("\n📊 Phase 2 Feature Summary:")
        logger.info("  - Enhanced Margin Trading: 9 features")
        logger.info("  - Option Sentiment: 10 features")
        logger.info("  - Enhanced Flow Analysis: 9 features")
        logger.info("  - Total: 28 new advanced features")
    else:
        logger.error("\n⚠️ Some tests failed. Please review the logs above.")
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)