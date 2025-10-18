#!/usr/bin/env python3
"""
Test RDBStorage functionality for HPO system
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Set environment for package imports
os.environ["PYTHONPATH"] = str(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_sqlite_storage():
    """Test SQLite storage functionality"""
    try:
        logger.info("🧪 Testing SQLite storage...")

        from src.gogooku3.hpo import ATFTHPOOptimizer

        # Use temporary database
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            storage_url = f"sqlite:///{db_path}"

            logger.info(f"   Test database: {db_path}")

            # Test 1: Create new study
            optimizer1 = ATFTHPOOptimizer(
                study_name="test_storage", storage=storage_url, n_trials=2, timeout=10
            )

            # Mock optimization (simplified)
            import optuna

            def mock_objective(trial):
                lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
                batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
                # Simple score based on parameters
                score = (lr * 1000) + (batch_size / 1000)
                return score

            # Run first batch
            study1 = optuna.create_study(
                study_name="test_storage",
                storage=storage_url,
                direction="maximize",
                load_if_exists=False,
            )
            study1.optimize(mock_objective, n_trials=2)

            logger.info(f"   First batch: {len(study1.trials)} trials completed")
            logger.info(f"   Best score: {study1.best_value:.6f}")

            # Test 2: Load existing study
            study2 = optuna.load_study(study_name="test_storage", storage=storage_url)

            logger.info(f"   Loaded study: {len(study2.trials)} trials found")

            # Test 3: Resume optimization
            study2.optimize(mock_objective, n_trials=1)

            logger.info(f"   After resume: {len(study2.trials)} trials total")
            logger.info(f"   Final best score: {study2.best_value:.6f}")

            # Test 4: Get study status via optimizer
            status = optimizer1.get_study_status()
            logger.info(
                f"   Status check: {status['completed_trials']} completed trials"
            )

            # Verify database file exists
            if db_path.exists():
                logger.info(f"   Database file created: {db_path.stat().st_size} bytes")
            else:
                raise FileNotFoundError("Database file not created")

            logger.info("✅ SQLite storage test passed")
            return True

    except Exception as e:
        logger.error(f"❌ SQLite storage test failed: {e}")
        return False


def test_study_persistence():
    """Test study persistence across sessions"""
    try:
        logger.info("🧪 Testing study persistence...")

        import optuna

        from src.gogooku3.hpo import ATFTHPOOptimizer

        # Use temporary database
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "persistence_test.db"
            storage_url = f"sqlite:///{db_path}"

            # Session 1: Create and run trials
            def simple_objective(trial):
                x = trial.suggest_float("x", -10, 10)
                return -((x - 2) ** 2)  # Maximum at x=2

            study1 = optuna.create_study(
                study_name="persistence_test", storage=storage_url, direction="maximize"
            )
            study1.optimize(simple_objective, n_trials=3)

            best_value_session1 = study1.best_value
            best_params_session1 = study1.best_params

            logger.info(f"   Session 1: best_value={best_value_session1:.4f}")
            logger.info(f"   Session 1: best_params={best_params_session1}")

            # Session 2: Load and continue
            study2 = optuna.load_study(
                study_name="persistence_test", storage=storage_url
            )

            # Verify data persisted
            if len(study2.trials) != 3:
                raise ValueError(f"Expected 3 trials, got {len(study2.trials)}")

            if abs(study2.best_value - best_value_session1) > 1e-6:
                raise ValueError("Best value not preserved")

            # Continue optimization
            study2.optimize(simple_objective, n_trials=2)

            logger.info(f"   Session 2: {len(study2.trials)} total trials")
            logger.info(f"   Session 2: best_value={study2.best_value:.4f}")

            # Test with optimizer interface
            optimizer = ATFTHPOOptimizer(
                study_name="persistence_test",
                storage=storage_url,
                n_trials=0,  # No trials, just status
            )

            status = optimizer.get_study_status()

            if status["completed_trials"] != 5:
                raise ValueError(
                    f"Expected 5 trials in status, got {status['completed_trials']}"
                )

            logger.info(
                f"   Optimizer status: {status['completed_trials']} trials, success_rate={status['success_rate']:.1f}%"
            )

            logger.info("✅ Study persistence test passed")
            return True

    except Exception as e:
        logger.error(f"❌ Study persistence test failed: {e}")
        return False


def test_resume_functionality():
    """Test resume functionality"""
    try:
        logger.info("🧪 Testing resume functionality...")

        import optuna

        from src.gogooku3.hpo import ATFTHPOOptimizer

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "resume_test.db"
            storage_url = f"sqlite:///{db_path}"

            # Initial optimizer
            optimizer1 = ATFTHPOOptimizer(
                study_name="resume_test", storage=storage_url, n_trials=3, timeout=5
            )

            # Mock some initial trials
            def simple_objective(trial):
                x = trial.suggest_float("x", 0, 1)
                return x**2

            study = optuna.create_study(
                study_name="resume_test", storage=storage_url, direction="maximize"
            )
            study.optimize(simple_objective, n_trials=2)

            logger.info(f"   Initial: {len(study.trials)} trials")

            # Resume with additional trials
            optimizer2 = ATFTHPOOptimizer(
                study_name="resume_test",
                storage=storage_url,
                n_trials=5,  # Will run 3 additional trials (5 - 2)
            )

            # Test resume_study method
            resumed_study = optimizer2.resume_study(additional_trials=1)

            logger.info(f"   After resume: {len(resumed_study.trials)} trials")

            if len(resumed_study.trials) != 3:
                raise ValueError(
                    f"Expected 3 trials after resume, got {len(resumed_study.trials)}"
                )

            # Check status
            status = optimizer2.get_study_status()
            logger.info(f"   Final status: {status['completed_trials']} completed")

            logger.info("✅ Resume functionality test passed")
            return True

    except Exception as e:
        logger.error(f"❌ Resume functionality test failed: {e}")
        return False


def main():
    """Run all storage tests"""
    logger.info("🚀 Starting RDBStorage functionality tests")

    tests = [
        ("SQLite Storage", test_sqlite_storage),
        ("Study Persistence", test_study_persistence),
        ("Resume Functionality", test_resume_functionality),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        result = test_func()
        results.append((test_name, result))

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("🎯 RDBStorage Test Results")
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
        logger.info("🎉 All RDBStorage tests passed!")
        logger.info("   RDB storage system is ready for production use")
        return 0
    else:
        logger.error("❌ Some RDBStorage tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
