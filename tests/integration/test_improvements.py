#!/usr/bin/env python3
"""
Test script to verify improvements made to gogooku3-standalone.

Tests:
1. UnifiedFeatureConverter - ATFT data conversion
2. LightGBMFinancialBaseline - Actual training and evaluation
3. DayBatchSampler - Date-based batching
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("test_improvements")


def test_unified_feature_converter():
    """Test UnifiedFeatureConverter implementation."""
    logger.info("=" * 60)
    logger.info("Testing UnifiedFeatureConverter...")

    try:
        from scripts.models.unified_feature_converter import UnifiedFeatureConverter

        # Create sample data
        n_samples = 1000
        n_features = 50
        n_codes = 10

        dates = pd.date_range("2023-01-01", periods=n_samples // n_codes, freq="D")

        data = {
            "Code": np.repeat([f"CODE_{i:04d}" for i in range(n_codes)], len(dates)),
            "Date": np.tile(dates, n_codes),
            "Close": np.random.randn(n_samples) * 100 + 1000,
            "Volume": np.random.randint(100000, 1000000, n_samples),
        }

        # Add feature columns
        for i in range(n_features):
            data[f"feature_{i}"] = np.random.randn(n_samples)

        # Add target columns
        data["feat_ret_1d"] = np.random.randn(n_samples) * 0.01
        data["feat_ret_5d"] = np.random.randn(n_samples) * 0.02

        df = pl.DataFrame(data)

        # Test converter
        converter = UnifiedFeatureConverter(sequence_length=20)

        # Test conversion (use temp directory for output)
        output_dir = Path("output/test_atft_data")
        result = converter.convert_to_atft_format(df, str(output_dir))

        logger.info("✅ UnifiedFeatureConverter test passed")
        logger.info(f"  - Created {result['metadata']['n_sequences']} sequences")
        logger.info(f"  - Features: {result['metadata']['n_features']}")
        logger.info(f"  - Train files: {len(result['train_files'])}")

        # Cleanup
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)

        return True

    except Exception as e:
        logger.error(f"❌ UnifiedFeatureConverter test failed: {e}")
        return False


def test_lightgbm_baseline():
    """Test LightGBMFinancialBaseline implementation."""
    logger.info("=" * 60)
    logger.info("Testing LightGBMFinancialBaseline...")

    try:
        from src.gogooku3.models.lightgbm_baseline import LightGBMFinancialBaseline

        # Create sample data
        n_samples = 1000
        n_features = 20

        data = {
            "Code": np.random.choice(["CODE_0001", "CODE_0002", "CODE_0003"], n_samples),
            "Date": pd.date_range("2023-01-01", periods=n_samples, freq="D"),
        }

        # Add features
        for i in range(n_features):
            data[f"feature_{i}"] = np.random.randn(n_samples)

        # Add targets
        data["feat_ret_1d"] = np.random.randn(n_samples) * 0.01
        data["feat_ret_5d"] = np.random.randn(n_samples) * 0.02

        df = pd.DataFrame(data)

        # Test baseline
        baseline = LightGBMFinancialBaseline(
            prediction_horizons=[1, 5],
            n_estimators=10,  # Small for testing
            verbose=False,
        )

        # Train
        baseline.fit(df)

        # Evaluate
        performance = baseline.evaluate_performance()

        # Get feature importance
        importance = baseline.get_feature_importance(horizon=1, top_k=5)

        # Get summary
        summary = baseline.get_results_summary()

        logger.info("✅ LightGBMFinancialBaseline test passed")
        logger.info(f"  - Models trained: {summary.get('n_models', 0)}")
        logger.info(f"  - Mean IC: {summary.get('mean_ic', 0):.4f}")
        logger.info(f"  - Mean RankIC: {summary.get('mean_rank_ic', 0):.4f}")
        logger.info(f"  - Top features: {importance['feature'].tolist() if not importance.empty else []}")

        return True

    except Exception as e:
        logger.error(f"❌ LightGBMFinancialBaseline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_day_batch_sampler():
    """Test DayBatchSampler implementation."""
    logger.info("=" * 60)
    logger.info("Testing DayBatchSampler...")

    try:
        from src.gogooku3.data.samplers.day_batch_sampler import DayBatchSampler

        # Create mock dataset
        class MockDataset:
            def __init__(self, n_samples=1000):
                self.data = pd.DataFrame({
                    "date": pd.date_range("2023-01-01", periods=n_samples // 10, freq="D").repeat(10),
                    "code": np.tile([f"CODE_{i:04d}" for i in range(10)], n_samples // 10),
                    "feature": np.random.randn(n_samples),
                })

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data.iloc[idx].to_dict()

        dataset = MockDataset(1000)

        # Test sampler
        sampler = DayBatchSampler(
            dataset,
            batch_size=32,
            shuffle_within_day=True,
            shuffle_days=False,
        )

        # Get statistics
        stats = sampler.get_statistics()

        # Test iteration
        n_batches = 0
        total_samples = 0
        for batch_indices in sampler:
            n_batches += 1
            total_samples += len(batch_indices)
            if n_batches > 5:  # Test first few batches
                break

        logger.info("✅ DayBatchSampler test passed")
        logger.info(f"  - Days: {stats['n_days']}")
        logger.info(f"  - Total batches: {stats['n_batches']}")
        logger.info(f"  - Avg samples/day: {stats['avg_samples_per_day']:.1f}")
        logger.info(f"  - Batches iterated: {n_batches}")

        return True

    except Exception as e:
        logger.error(f"❌ DayBatchSampler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("Starting improvement tests for gogooku3-standalone")
    logger.info("=" * 60)

    results = {
        "UnifiedFeatureConverter": test_unified_feature_converter(),
        "LightGBMFinancialBaseline": test_lightgbm_baseline(),
        "DayBatchSampler": test_day_batch_sampler(),
    }

    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for test_name, test_passed in results.items():
        status = "✅ PASSED" if test_passed else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")

    logger.info(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All improvements successfully implemented and tested!")
        return 0
    else:
        logger.error("⚠️  Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())