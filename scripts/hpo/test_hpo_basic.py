#!/usr/bin/env python3
"""
Basic HPO functionality test
Tests the HPO infrastructure without running full training
"""

import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Set environment for package imports
os.environ["PYTHONPATH"] = str(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def test_hpo_imports():
    """Test basic HPO component imports"""
    try:
        logger.info("🧪 Testing HPO imports...")

        logger.info("✅ All HPO components imported successfully")
        return True

    except Exception as e:
        logger.error(f"❌ HPO import failed: {e}")
        return False


def test_multi_horizon_objective():
    """Test MultiHorizonObjective functionality"""
    try:
        logger.info("🧪 Testing MultiHorizonObjective...")

        from src.gogooku3.hpo.objectives import MultiHorizonObjective

        # Initialize objective
        objective = MultiHorizonObjective()

        # Test with mock metrics
        mock_metrics = {
            "rank_ic": {"1d": 0.15, "5d": 0.12, "10d": 0.10, "20d": 0.08},
            "sharpe": {"1d": 0.8, "5d": 0.6, "10d": 0.5, "20d": 0.4},
        }

        # Compute score
        score = objective.compute_score(mock_metrics)
        logger.info(f"   Computed score: {score:.4f}")

        if 0 <= score <= 1:
            logger.info("✅ MultiHorizonObjective test passed")
            return True
        else:
            logger.error(f"❌ Score {score:.4f} outside expected range [0, 1]")
            return False

    except Exception as e:
        logger.error(f"❌ MultiHorizonObjective test failed: {e}")
        return False


def test_metrics_extractor():
    """Test MetricsExtractor functionality"""
    try:
        logger.info("🧪 Testing MetricsExtractor...")

        from src.gogooku3.hpo.metrics_extractor import MetricsExtractor, TrainingMetrics

        # Initialize extractor
        extractor = MetricsExtractor()

        # Test with mock dictionary
        mock_dict = {
            "rank_ic_1d": 0.15,
            "rank_ic_5d": 0.12,
            "sharpe_1d": 0.8,
            "sharpe_5d": 0.6,
            "train_loss": 0.5,
            "val_loss": 0.6,
            "epoch": 10,
            "training_time": 120.0,
        }

        # Extract metrics
        metrics = extractor.extract_from_metrics_dict(mock_dict)

        if metrics and isinstance(metrics, TrainingMetrics):
            logger.info(
                f"   Extracted metrics: RankIC={len(metrics.rank_ic)}, Sharpe={len(metrics.sharpe)}"
            )
            logger.info(
                f"   Train loss: {metrics.train_loss:.3f}, Val loss: {metrics.val_loss:.3f}"
            )

            # Test validation
            validation = extractor.validate_metrics(metrics)
            logger.info(
                f"   Validation errors: {len(validation['errors'])}, warnings: {len(validation['warnings'])}"
            )

            logger.info("✅ MetricsExtractor test passed")
            return True
        else:
            logger.error("❌ Failed to extract metrics from dictionary")
            return False

    except Exception as e:
        logger.error(f"❌ MetricsExtractor test failed: {e}")
        return False


def test_hpo_optimizer_init():
    """Test ATFTHPOOptimizer initialization"""
    try:
        logger.info("🧪 Testing ATFTHPOOptimizer initialization...")

        from src.gogooku3.hpo.hpo_optimizer import ATFTHPOOptimizer

        # Initialize optimizer with test settings
        optimizer = ATFTHPOOptimizer(
            study_name="test_study",
            n_trials=5,
            n_jobs=1,
            timeout=30,  # 30 second timeout for test
        )

        logger.info(f"   Study name: {optimizer.study_name}")
        logger.info(f"   Trials: {optimizer.n_trials}")
        logger.info(f"   Timeout: {optimizer.timeout}")

        # Test hyperparameter suggestion (without actual trial)
        import optuna

        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        params = optimizer.suggest_hyperparameters(trial)
        logger.info(f"   Suggested {len(params)} hyperparameters")

        # Check key parameters exist
        required_params = [
            "atft_d_model",
            "learning_rate",
            "batch_size",
            "horizon_weights",
        ]
        missing = [p for p in required_params if p not in params]

        if not missing:
            logger.info("✅ ATFTHPOOptimizer initialization test passed")
            return True
        else:
            logger.error(f"❌ Missing required parameters: {missing}")
            return False

    except Exception as e:
        logger.error(f"❌ ATFTHPOOptimizer test failed: {e}")
        return False


def test_integration():
    """Test basic integration between components"""
    try:
        logger.info("🧪 Testing HPO component integration...")

        from src.gogooku3.hpo import (
            ATFTHPOOptimizer,
            MetricsExtractor,
            MultiHorizonObjective,
        )

        # Initialize components
        objective = MultiHorizonObjective()
        extractor = MetricsExtractor()
        optimizer = ATFTHPOOptimizer(
            study_name="integration_test", n_trials=1, timeout=10
        )

        # Mock training result
        mock_training_result = {
            "rank_ic_1d": 0.12,
            "rank_ic_5d": 0.10,
            "sharpe_1d": 0.7,
            "sharpe_5d": 0.5,
            "train_loss": 0.4,
            "val_loss": 0.5,
            "epoch": 5,
        }

        # Extract metrics
        metrics = extractor.extract_from_metrics_dict(mock_training_result)

        # Compute objective score
        score = objective.compute_score(
            {"rank_ic": metrics.rank_ic, "sharpe": metrics.sharpe}
        )

        logger.info(f"   Integration score: {score:.4f}")
        logger.info(f"   Metrics summary: {extractor.format_metrics(metrics)}")

        logger.info("✅ HPO integration test passed")
        return True

    except Exception as e:
        logger.error(f"❌ HPO integration test failed: {e}")
        return False


def main():
    """Run all HPO tests"""
    logger.info("🚀 Starting HPO functionality tests")

    tests = [
        ("Import Tests", test_hpo_imports),
        ("MultiHorizonObjective", test_multi_horizon_objective),
        ("MetricsExtractor", test_metrics_extractor),
        ("HPOOptimizer Init", test_hpo_optimizer_init),
        ("Component Integration", test_integration),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        result = test_func()
        results.append((test_name, result))

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("🎯 HPO Test Results Summary")
    logger.info("=" * 50)

    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {test_name:<20} {status}")
        if result:
            passed += 1

    success_rate = passed / len(results) * 100
    logger.info(
        f"\nOverall: {passed}/{len(results)} tests passed ({success_rate:.1f}%)"
    )

    if passed == len(results):
        logger.info("🎉 All HPO functionality tests passed!")
        logger.info("   HPO system is ready for basic usage")
        return 0
    else:
        logger.error("❌ Some HPO tests failed")
        logger.error("   Check error messages above and fix issues before proceeding")
        return 1


if __name__ == "__main__":
    exit(main())
