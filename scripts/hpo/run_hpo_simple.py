#!/usr/bin/env python3
"""
Simple HPO runner for ATFT-GAT-FAN optimization
Demonstrates basic Optuna integration with the training pipeline
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Set environment for package imports
os.environ["PYTHONPATH"] = str(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("hpo_simple.log")],
)
logger = logging.getLogger(__name__)


def run_simple_hpo(
    n_trials: int = 5,
    timeout: int = 300,
    study_name: str = "atft_simple_hpo",
    storage_url: Optional[str] = None,
    resume: bool = False,
):
    """
    Run simple HPO optimization

    Args:
        n_trials: Number of optimization trials
        timeout: Timeout in seconds for each trial
        study_name: Optuna study name
        storage_url: Database URL for persistence
        resume: Resume existing study if True
    """
    try:
        logger.info("🚀 Starting Simple HPO Optimization")
        logger.info(f"   Trials: {n_trials}")
        logger.info(f"   Timeout: {timeout}s")
        logger.info(f"   Study: {study_name}")
        logger.info(f"   Storage: {storage_url or 'in-memory'}")
        logger.info(f"   Resume: {resume}")

        # Import HPO components
        from src.gogooku3.hpo import ATFTHPOOptimizer

        # Initialize optimizer
        optimizer = ATFTHPOOptimizer(
            study_name=study_name,
            storage=storage_url,
            n_trials=n_trials,
            n_jobs=1,  # Single job for simplicity
            timeout=timeout,
            base_config_path="configs/atft/config.yaml",
        )

        # Run optimization (resume if requested)
        if resume:
            study = optimizer.resume_study()
        else:
            study = optimizer.optimize()

        # Display results
        logger.info("\n" + "=" * 60)
        logger.info("🎯 HPO Optimization Results")
        logger.info("=" * 60)
        logger.info(f"Best score: {study.best_value:.4f}")
        logger.info(f"Total trials: {len(study.trials)}")
        logger.info(
            f"Completed trials: {len([t for t in study.trials if t.state.name == 'COMPLETE'])}"
        )

        # Save best configuration
        output_dir = Path("output/hpo")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save study results
        study_path = output_dir / f"{study_name}.db"
        storage_url = f"sqlite:///{study_path}"

        # Save best config
        best_config_path = output_dir / f"{study_name}_best_config.yaml"
        optimizer.save_best_config(study, best_config_path)

        logger.info("\n📊 Results saved to:")
        logger.info(f"   Study database: {study_path}")
        logger.info(f"   Best config: {best_config_path}")

        # Display best parameters summary
        best_params_formatted = optimizer.objective_func.format_best_params(
            study.best_params
        )
        logger.info(f"\n{best_params_formatted}")

        return study

    except Exception as e:
        logger.error(f"❌ HPO optimization failed: {e}")
        raise


def get_study_status(
    study_name: str = "atft_simple_hpo", storage_url: Optional[str] = None
):
    """
    Get status of an existing study

    Args:
        study_name: Optuna study name
        storage_url: Database URL for persistence
    """
    try:
        from src.gogooku3.hpo import ATFTHPOOptimizer

        # Initialize optimizer (no trials needed for status)
        optimizer = ATFTHPOOptimizer(
            study_name=study_name, storage=storage_url, n_trials=0
        )

        # Get status
        status = optimizer.get_study_status()

        if "error" in status:
            logger.error(f"❌ Cannot get study status: {status['error']}")
            return

        # Display status
        logger.info("\n" + "=" * 50)
        logger.info("📊 HPO Study Status")
        logger.info("=" * 50)
        logger.info(f"Study Name: {status['study_name']}")
        logger.info(f"Storage: {status['storage']}")
        logger.info(f"Total Trials: {status['total_trials']}")
        logger.info(f"Completed: {status['completed_trials']}")
        logger.info(f"Failed: {status['failed_trials']}")
        logger.info(f"Pruned: {status['pruned_trials']}")
        logger.info(f"Success Rate: {status['success_rate']:.1f}%")

        if status["best_value"] is not None:
            logger.info(f"Best Score: {status['best_value']:.4f}")

        if "avg_trial_duration" in status:
            logger.info(f"Avg Trial Duration: {status['avg_trial_duration']:.1f}s")
            logger.info(f"Total Time: {status['total_optimization_time']:.1f}s")

        return status

    except Exception as e:
        logger.error(f"❌ Failed to get study status: {e}")
        raise


def run_mock_hpo(n_trials: int = 3):
    """
    Run mock HPO without actual training (for testing)
    """
    try:
        logger.info("🧪 Running Mock HPO (no training)")

        import random
        import time

        import optuna

        from src.gogooku3.hpo import MetricsExtractor, MultiHorizonObjective

        # Initialize components
        objective_func = MultiHorizonObjective()
        extractor = MetricsExtractor()

        def mock_objective(trial):
            """Mock objective function that simulates training results"""
            # Suggest hyperparameters (similar to real optimizer)
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
            d_model = trial.suggest_categorical("atft_d_model", [128, 256, 384, 512])

            logger.info(
                f"Trial {trial.number}: lr={learning_rate:.2e}, batch={batch_size}, d_model={d_model}"
            )

            # Simulate training time
            time.sleep(1)

            # Generate mock metrics (simulate varying performance)
            base_rank_ic = 0.05 + random.random() * 0.15  # 0.05-0.20
            base_sharpe = 0.3 + random.random() * 0.7  # 0.3-1.0

            # Add some parameter-dependent variance
            lr_penalty = (
                abs(learning_rate - 3e-4) * 100
            )  # Penalty for being far from 3e-4
            batch_bonus = 1.2 if batch_size == 256 else 1.0
            model_bonus = 1.1 if d_model == 256 else 1.0

            mock_metrics = {
                "rank_ic": {
                    "1d": base_rank_ic * batch_bonus * model_bonus - lr_penalty * 0.1,
                    "5d": base_rank_ic * 0.8 * batch_bonus * model_bonus
                    - lr_penalty * 0.05,
                    "10d": base_rank_ic * 0.6 * batch_bonus * model_bonus,
                    "20d": base_rank_ic * 0.4 * batch_bonus * model_bonus,
                },
                "sharpe": {
                    "1d": base_sharpe * batch_bonus * model_bonus - lr_penalty * 0.2,
                    "5d": base_sharpe * 0.8 * batch_bonus * model_bonus
                    - lr_penalty * 0.1,
                    "10d": base_sharpe * 0.6 * batch_bonus * model_bonus,
                    "20d": base_sharpe * 0.4 * batch_bonus * model_bonus,
                },
            }

            # Clamp values to realistic ranges
            for horizon in mock_metrics["rank_ic"]:
                mock_metrics["rank_ic"][horizon] = max(
                    -0.1, min(0.3, mock_metrics["rank_ic"][horizon])
                )
            for horizon in mock_metrics["sharpe"]:
                mock_metrics["sharpe"][horizon] = max(
                    -1.0, min(2.0, mock_metrics["sharpe"][horizon])
                )

            # Compute objective score
            score = objective_func.compute_score(mock_metrics)

            logger.info(f"   RankIC@1d: {mock_metrics['rank_ic']['1d']:.3f}")
            logger.info(f"   Sharpe@1d: {mock_metrics['sharpe']['1d']:.3f}")
            logger.info(f"   Score: {score:.4f}")

            return score

        # Create study
        study = optuna.create_study(direction="maximize", study_name="mock_hpo_test")

        # Run optimization
        study.optimize(mock_objective, n_trials=n_trials)

        # Display results
        logger.info("\n" + "=" * 50)
        logger.info("🎯 Mock HPO Results")
        logger.info("=" * 50)
        logger.info(f"Best score: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        return study

    except Exception as e:
        logger.error(f"❌ Mock HPO failed: {e}")
        raise


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Simple HPO runner for ATFT-GAT-FAN")

    # Action selection
    parser.add_argument(
        "action",
        nargs="?",
        default="run",
        choices=["run", "resume", "status", "mock"],
        help="Action to perform (default: run)",
    )

    # Study configuration
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of optimization trials (default: 5)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per trial in seconds (default: 300)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="atft_simple_hpo",
        help="Optuna study name (default: atft_simple_hpo)",
    )

    # Storage configuration
    parser.add_argument(
        "--storage", type=str, help="Storage URL (e.g., sqlite:///output/hpo/optuna.db)"
    )

    # Legacy option
    parser.add_argument(
        "--mock", action="store_true", help="Run mock HPO without actual training"
    )

    args = parser.parse_args()

    # Handle legacy mock option
    if args.mock:
        args.action = "mock"

    try:
        if args.action == "mock":
            study = run_mock_hpo(args.trials)
        elif args.action == "status":
            get_study_status(args.study_name, args.storage)
            return 0
        elif args.action == "resume":
            study = run_simple_hpo(
                n_trials=args.trials,
                timeout=args.timeout,
                study_name=args.study_name,
                storage_url=args.storage,
                resume=True,
            )
        else:  # "run"
            study = run_simple_hpo(
                n_trials=args.trials,
                timeout=args.timeout,
                study_name=args.study_name,
                storage_url=args.storage,
                resume=False,
            )

        logger.info("✅ HPO run completed successfully")
        return 0

    except KeyboardInterrupt:
        logger.info("🛑 HPO interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ HPO run failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
