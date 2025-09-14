#!/usr/bin/env python3
"""
Simple HPO Test for ATFT-GAT-FAN
Minimal implementation to validate the HPO concept before full integration
"""

import json
import os
import sys
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any
import subprocess
import time

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleATFTHPO:
    """Simple HPO test for ATFT-GAT-FAN using smoke training"""

    def __init__(self, n_trials: int = 5, max_epochs: int = 3):
        self.n_trials = n_trials
        self.max_epochs = max_epochs
        self.study_name = "atft_simple_hpo_test"

    def run_smoke_training(self, hparams: Dict[str, Any]) -> Dict[str, float]:
        """Run a quick smoke training with given hyperparameters"""
        try:
            # Create a simple synthetic score for testing
            # In real implementation, this would run actual training

            # Simulate training time
            time.sleep(2)

            # Simulate metrics based on hyperparameters
            # Better learning rates and reasonable dropout should score higher
            lr = hparams['lr']
            dropout = hparams['dropout']

            # Simple heuristic: penalize extreme learning rates and high dropout
            lr_score = 1.0 - abs(lr - 0.001) * 100  # Optimal around 0.001
            dropout_penalty = dropout * 0.5  # Lower dropout is better

            base_score = 0.1 + lr_score * 0.05 - dropout_penalty

            # Add some noise to simulate real training variability
            import random
            noise = random.gauss(0, 0.02)

            mock_metrics = {
                "rank_ic": {
                    "1d": max(-0.1, min(0.3, base_score + noise)),
                    "5d": max(-0.1, min(0.3, base_score * 0.9 + noise * 0.8)),
                    "10d": max(-0.1, min(0.3, base_score * 0.8 + noise * 0.6)),
                    "20d": max(-0.1, min(0.3, base_score * 0.7 + noise * 0.4)),
                },
                "sharpe": {
                    "1d": max(-2.0, min(2.0, base_score * 5 + noise * 2)),
                    "5d": max(-2.0, min(2.0, base_score * 4 + noise * 1.5)),
                    "10d": max(-2.0, min(2.0, base_score * 3.5 + noise * 1.2)),
                    "20d": max(-2.0, min(2.0, base_score * 3 + noise)),
                },
                "training_time": 2.0,
                "test_mode": True
            }

            logger.info(f"Mock training completed: RankIC_5d={mock_metrics['rank_ic']['5d']:.3f}")
            return mock_metrics

        except Exception as e:
            logger.error(f"Mock training failed: {e}")
            return {
                "rank_ic": {"1d": -0.1, "5d": -0.1, "10d": -0.1, "20d": -0.1},
                "sharpe": {"1d": -1.0, "5d": -1.0, "10d": -1.0, "20d": -1.0},
                "training_failed": True
            }

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function for testing"""

        # Define test hyperparameter space
        hparams = {
            "lr": trial.suggest_float("lr", 5e-5, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
            "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8]),
        }

        # Run mock training
        metrics = self.run_smoke_training(hparams)

        if metrics.get("training_failed", False):
            return 999.0  # High penalty for failed trials

        # Calculate weighted multi-horizon RankIC score (for minimization)
        rank_ic = metrics["rank_ic"]
        weights = {"1d": 0.2, "5d": 0.35, "10d": 0.35, "20d": 0.1}

        weighted_score = sum(
            weights[horizon] * rank_ic.get(horizon, -0.1)
            for horizon in weights.keys()
        )

        # Report for pruning (using 5d as primary metric)
        primary_metric = rank_ic.get("5d", -0.1)
        trial.report(primary_metric, step=0)

        # Return negative for minimization (Optuna minimizes by default)
        return -weighted_score

    def run_test_optimization(self):
        """Run the HPO test"""

        logger.info(f"Starting HPO test: {self.n_trials} trials")

        # Create study
        study = optuna.create_study(
            study_name=self.study_name,
            direction="minimize",
            sampler=TPESampler(
                multivariate=True,
                n_startup_trials=2,
                seed=42
            ),
            pruner=MedianPruner(
                n_startup_trials=2,
                n_warmup_steps=1
            ),
        )

        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
        )

        # Report results
        best_trial = study.best_trial
        logger.info(f"🎯 HPO Test Results:")
        logger.info(f"   Best score: {-best_trial.value:.4f}")
        logger.info(f"   Best params: {best_trial.params}")

        # Test basic functionality
        all_trials = study.trials
        completed_trials = [t for t in all_trials if t.state == optuna.trial.TrialState.COMPLETE]
        failed_trials = [t for t in all_trials if t.state == optuna.trial.TrialState.FAIL]

        logger.info(f"   Completed trials: {len(completed_trials)}/{len(all_trials)}")
        logger.info(f"   Failed trials: {len(failed_trials)}")

        # Validate that optimization is working (scores should improve over trials)
        if len(completed_trials) >= 3:
            first_score = -completed_trials[0].value
            best_score = -best_trial.value
            improvement = best_score - first_score

            logger.info(f"   Score improvement: {improvement:.4f}")

            if improvement > 0:
                logger.info("✅ HPO test PASSED: Optimization is improving scores")
                return True
            else:
                logger.warning("⚠️ HPO test CONCERN: No clear improvement observed")
                return False
        else:
            logger.warning("⚠️ HPO test INCOMPLETE: Too few completed trials")
            return False

    def demonstrate_hpo_features(self):
        """Demonstrate key HPO features for ATFT-GAT-FAN"""

        logger.info("🔬 Demonstrating HPO Features:")

        # 1. TPE Sampler benefits
        logger.info("1. TPE Sampler: Learns from previous trials to suggest better hyperparameters")

        # 2. Pruning benefits
        logger.info("2. MedianPruner: Stops unpromising trials early to save compute")

        # 3. Multi-horizon optimization
        logger.info("3. Multi-horizon objective: Optimizes 1d, 5d, 10d, 20d predictions jointly")

        # 4. GPU utilization
        logger.info("4. GPU optimization: Ready for bf16, GPU memory management")

        # 5. Persistence and resumability
        logger.info("5. Study persistence: Can resume interrupted optimization runs")


def main():
    """Main entry point for HPO testing"""

    print("=" * 60)
    print("ATFT-GAT-FAN HPO System Test")
    print("Testing Optuna integration with mock training")
    print("=" * 60)

    # Create HPO tester
    hpo = SimpleATFTHPO(n_trials=8, max_epochs=3)

    # Demonstrate features
    hpo.demonstrate_hpo_features()

    print("\n" + "=" * 60)
    print("Running HPO test with mock training...")
    print("=" * 60)

    # Run test
    success = hpo.run_test_optimization()

    print("\n" + "=" * 60)
    if success:
        print("✅ HPO System Test PASSED")
        print("Ready for integration with real ATFT-GAT-FAN training")
        print("\nNext steps:")
        print("1. Integrate with scripts/train_atft.py")
        print("2. Add real metrics extraction")
        print("3. Configure GPU-optimized parameters")
        print("4. Set up persistent storage (SQLite/PostgreSQL)")
    else:
        print("❌ HPO System Test FAILED")
        print("Check logs for issues")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())