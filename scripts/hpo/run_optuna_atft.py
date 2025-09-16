#!/usr/bin/env python3
"""
Optuna-based Hyperparameter Optimization for ATFT
ATFT用のOptunaベースのハイパーパラメータ最適化

train.* 名前空間で統一されたオーバーライドを発行
hpo.output_metrics_jsonでメトリクスを出力
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import optuna
from optuna import Trial

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


class ATFTOptunaOptimizer:
    """ATFT用のOptunaハイパーパラメータ最適化"""

    def __init__(
        self,
        data_path: str,
        study_name: str = "atft_optimization",
        n_trials: int = 20,
        max_epochs_per_trial: int = 10,
        output_dir: str = "output/hpo",
    ):
        self.data_path = Path(data_path)
        self.study_name = study_name
        self.n_trials = n_trials
        self.max_epochs_per_trial = max_epochs_per_trial
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create study
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction="maximize",  # Maximize Sharpe ratio
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=5,
            ),
        )

    def objective(self, trial: Trial) -> float:
        """Optuna objective function"""

        # Suggest hyperparameters using train.* namespace
        lr = trial.suggest_float("train.optimizer.lr", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("train.batch.train_batch_size", [512, 1024, 2048])
        dropout = trial.suggest_float("train.model.dropout", 0.0, 0.3)
        num_layers = trial.suggest_int("train.model.num_layers", 4, 8)
        hidden_dim = trial.suggest_categorical("train.model.hidden_dim", [128, 256, 512])

        # Create HPO metrics output path
        trial_dir = self.output_dir / f"trial_{trial.number}"
        trial_dir.mkdir(exist_ok=True)
        hpo_metrics_path = trial_dir / "metrics.json"

        # Build command with overrides
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "integrated_ml_training_pipeline_final.py"),
            "--data-path", str(self.data_path),
            f"--max-epochs={self.max_epochs_per_trial}",
            f"train.optimizer.lr={lr}",
            f"train.batch.train_batch_size={batch_size}",
            f"train.model.dropout={dropout}",
            f"train.model.num_layers={num_layers}",
            f"train.model.hidden_dim={hidden_dim}",
            f"hpo.output_metrics_json={hpo_metrics_path}",
        ]

        logger.info(f"Trial {trial.number}: Starting with lr={lr:.2e}, batch_size={batch_size}")

        try:
            # Run training
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )

            # Load metrics from JSON
            if hpo_metrics_path.exists():
                with open(hpo_metrics_path) as f:
                    metrics = json.load(f)

                # Get Sharpe ratio (or other metric)
                sharpe = metrics.get("sharpe", 0.0)

                # Report intermediate values for pruning
                for epoch in range(self.max_epochs_per_trial):
                    if f"epoch_{epoch}_sharpe" in metrics:
                        trial.report(metrics[f"epoch_{epoch}_sharpe"], epoch)
                        if trial.should_prune():
                            raise optuna.TrialPruned()

                logger.info(f"Trial {trial.number} completed: Sharpe={sharpe:.4f}")
                return sharpe

            else:
                logger.warning(f"Trial {trial.number}: No metrics file found")
                return 0.0

        except subprocess.TimeoutExpired:
            logger.error(f"Trial {trial.number} timed out")
            raise optuna.TrialPruned()
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()

    def optimize(self) -> None:
        """Run optimization"""
        logger.info(f"Starting optimization: {self.n_trials} trials")

        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            catch=(Exception,),
            callbacks=[self._callback],
        )

        # Save results
        self._save_results()

    def _callback(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Callback after each trial"""
        if trial.value is not None:
            logger.info(
                f"Trial {trial.number}: Sharpe={trial.value:.4f} "
                f"(Best: {study.best_value:.4f})"
            )

    def _save_results(self) -> None:
        """Save optimization results"""
        # Best parameters
        best_params_file = self.output_dir / "best_params.json"
        with open(best_params_file, "w") as f:
            json.dump(self.study.best_params, f, indent=2)

        # All trials
        trials_file = self.output_dir / "all_trials.json"
        trials_data = []
        for trial in self.study.trials:
            trials_data.append({
                "number": trial.number,
                "params": trial.params,
                "value": trial.value,
                "state": str(trial.state),
            })
        with open(trials_file, "w") as f:
            json.dump(trials_data, f, indent=2)

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("OPTIMIZATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Best Sharpe: {self.study.best_value:.4f}")
        logger.info("Best parameters:")
        for key, value in self.study.best_params.items():
            logger.info(f"  {key}: {value}")
        logger.info(f"Results saved to: {self.output_dir}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Optuna-based hyperparameter optimization for ATFT"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to ML dataset",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of optimization trials",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=10,
        help="Maximum epochs per trial",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="atft_optimization",
        help="Optuna study name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/hpo",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Run optimization
    optimizer = ATFTOptunaOptimizer(
        data_path=args.data_path,
        study_name=args.study_name,
        n_trials=args.n_trials,
        max_epochs_per_trial=args.max_epochs,
        output_dir=args.output_dir,
    )

    optimizer.optimize()


if __name__ == "__main__":
    main()
