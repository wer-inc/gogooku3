#!/usr/bin/env python3
"""
ATFT-GAT-FAN Optuna Hyperparameter Optimization
Integrates with existing training pipeline using TPE + Pruning + GPU optimization
"""

import json
import os
import subprocess
import sys
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ATFTOptunaOptimizer:
    """Optuna-based hyperparameter optimization for ATFT-GAT-FAN"""

    def __init__(self,
                 study_name: str = "atft_gat_fan_hpo",
                 n_trials: int = 40,
                 n_jobs: int = 1,
                 storage_url: str = None):
        self.study_name = study_name
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.storage_url = storage_url or f"sqlite:///optuna_studies/{study_name}.db"

        # Create storage directory
        Path("optuna_studies").mkdir(exist_ok=True)

    def run_single_trial(self, hparams: Dict[str, Any]) -> Dict[str, Any]:
        """Execute single training trial with given hyperparameters"""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = Path(tmpdir) / "hpo_metrics.json"

            # Build Hydra overrides for hyperparameters
            overrides = [
                # Learning parameters
                f"optimizer.lr={hparams['lr']}",
                f"optimizer.weight_decay={hparams['weight_decay']}",

                # Model architecture
                f"model.gat.layer_config.dropout={hparams['dropout']}",
                f"model.gat.layer_config.edge_dropout={hparams['edge_dropout']}",
                f"model.gat.architecture.heads=[{hparams['n_heads']},{hparams['n_heads']//2}]",
                f"model.hidden_size={hparams['hidden_size']}",

                # Loss configuration
                f"loss.huber_delta={hparams['huber_delta']}",
                f"loss.multi_horizon_weights=[{','.join(map(str, hparams['horizon_weights']))}]",

                # Training settings (optimized for HPO)
                "trainer.max_epochs=15",  # Reduced for faster iteration
                "trainer.enable_checkpointing=false",
                "trainer.num_sanity_val_steps=0",
                "trainer.accelerator=gpu",
                "trainer.devices=1",
                "trainer.precision=bf16-mixed",  # GPU optimization

                # Early stopping for faster pruning
                "early_stopping.patience=5",
                "early_stopping.min_delta=0.0001",

                # HPO output
                f"hpo.output_metrics_json={out_json}",
                f"hpo.enabled=true",
            ]

            # Execute training
            cmd = [
                sys.executable,
                "scripts/integrated_ml_training_pipeline.py",
                *overrides
            ]

            try:
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    cwd=ROOT
                )

                # Read metrics
                with open(out_json) as f:
                    metrics = json.load(f)

                return metrics

            except subprocess.CalledProcessError as e:
                logger.error(f"Training failed: {e.stderr}")
                # Return poor metrics for failed trials
                return {
                    "rank_ic": {"1d": -0.1, "5d": -0.1, "10d": -0.1, "20d": -0.1},
                    "sharpe": {"1d": -1.0, "5d": -1.0, "10d": -1.0, "20d": -1.0},
                    "training_failed": True
                }

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function - maximizing weighted multi-horizon RankIC"""

        # Define hyperparameter search space
        hparams = {
            # Learning parameters
            "lr": trial.suggest_float("lr", 1e-5, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),

            # Model architecture
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "edge_dropout": trial.suggest_float("edge_dropout", 0.0, 0.3),
            "n_heads": trial.suggest_categorical("n_heads", [2, 4, 6, 8]),
            "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256, 384]),

            # Loss configuration
            "huber_delta": trial.suggest_float("huber_delta", 0.005, 0.05),

            # Multi-horizon weights (normalized to sum=4 for consistency)
            "horizon_weights": self._suggest_horizon_weights(trial),
        }

        # Run training trial
        metrics = self.run_single_trial(hparams)

        if metrics.get("training_failed", False):
            return -999.0  # Large penalty for failed trials

        # Calculate weighted multi-horizon RankIC score
        rank_ic = metrics.get("rank_ic", {})

        # Weights prioritize medium-term performance (5d, 10d)
        weights = {"1d": 0.2, "5d": 0.35, "10d": 0.35, "20d": 0.1}

        weighted_score = sum(
            weights[horizon] * rank_ic.get(horizon, -0.1)
            for horizon in weights.keys()
        )

        # Report for pruning (use 5d RankIC as primary metric)
        primary_metric = rank_ic.get("5d", -0.1)
        trial.report(primary_metric, step=0)

        # Optuna minimizes, so return negative for maximization
        return -weighted_score

    def _suggest_horizon_weights(self, trial: optuna.Trial) -> list:
        """Suggest normalized horizon weights that sum to reasonable total"""
        raw_weights = [
            trial.suggest_float("weight_1d", 0.1, 2.0),
            trial.suggest_float("weight_5d", 0.1, 2.0),
            trial.suggest_float("weight_10d", 0.1, 2.0),
            trial.suggest_float("weight_20d", 0.1, 2.0),
        ]

        # Normalize to sum to 4.0 (consistent with current balanced weights)
        total = sum(raw_weights)
        return [4.0 * w / total for w in raw_weights]

    def optimize(self):
        """Run hyperparameter optimization study"""

        # Create or load study
        study = optuna.create_study(
            study_name=self.study_name,
            direction="minimize",  # We return negative scores
            sampler=TPESampler(
                multivariate=True,
                n_startup_trials=10,
                seed=42,
                consider_prior=True,
                prior_weight=1.0,
                consider_magic_clip=True,
                consider_endpoints=False,
            ),
            pruner=MedianPruner(
                n_startup_trials=8,
                n_warmup_steps=5,
                interval_steps=1,
            ),
            storage=self.storage_url,
            load_if_exists=True,
        )

        logger.info(f"Starting optimization: {self.n_trials} trials")
        logger.info(f"Study storage: {self.storage_url}")

        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            gc_after_trial=True,
            show_progress_bar=True,
        )

        # Report results
        best_trial = study.best_trial
        logger.info(f"Best trial score: {-best_trial.value:.4f}")
        logger.info(f"Best hyperparameters: {best_trial.params}")

        # Save results
        results_file = Path("optuna_studies") / f"{self.study_name}_results.json"
        with open(results_file, "w") as f:
            json.dump({
                "best_score": -best_trial.value,
                "best_params": best_trial.params,
                "n_trials": len(study.trials),
                "study_name": self.study_name,
            }, f, indent=2)

        logger.info(f"Results saved to: {results_file}")

        return study


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="ATFT-GAT-FAN Hyperparameter Optimization")
    parser.add_argument("--study-name", default="atft_gat_fan_hpo", help="Optuna study name")
    parser.add_argument("--n-trials", type=int, default=40, help="Number of optimization trials")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("--storage-url", help="Optuna storage URL (default: SQLite)")

    args = parser.parse_args()

    optimizer = ATFTOptunaOptimizer(
        study_name=args.study_name,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        storage_url=args.storage_url,
    )

    study = optimizer.optimize()

    return 0


if __name__ == "__main__":
    sys.exit(main())