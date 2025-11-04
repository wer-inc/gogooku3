"""
Hyperparameter Optimization with Optuna
Optunaを使用した自動ハイパーパラメータ最適化

PDFで提案された改善: ベースライン性能を初期値としたOptuna統合
"""

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import optuna
import torch
from omegaconf import DictConfig, OmegaConf
from optuna import Study, Trial
from optuna.pruners import HyperbandPruner, MedianPruner
from optuna.samplers import TPESampler

logger = logging.getLogger(__name__)


class ATFTHyperparameterOptimizer:
    """
    Hyperparameter optimizer for ATFT models using Optuna.

    ベースライン性能を参考にした効率的な探索を実現
    """

    def __init__(
        self,
        config: DictConfig,
        objective_fn: Callable | None = None,
        study_name: str = "atft_optimization",
        storage: str | None = None,
        baseline_metrics: dict[str, float] | None = None,
    ):
        """
        Initialize the hyperparameter optimizer.

        Args:
            config: Base configuration
            objective_fn: Objective function to optimize
            study_name: Name of the Optuna study
            storage: Database URL for distributed optimization
            baseline_metrics: Baseline model metrics for reference
        """
        self.config = config
        self.objective_fn = objective_fn
        self.study_name = study_name
        self.storage = storage
        self.baseline_metrics = baseline_metrics or {}

        # Optimization settings from config
        self.n_trials = config.optuna.n_trials
        self.timeout = config.optuna.timeout
        self.direction = config.optuna.direction
        self.metric = config.optuna.metric

        # Search space configuration
        self.search_space = config.optuna.search_space

        # Setup study
        self.study = self._create_study()

    def _create_study(self) -> Study:
        """Create Optuna study with appropriate sampler and pruner."""
        # Select sampler
        if self.config.optuna.sampler == "TPESampler":
            sampler = TPESampler(
                seed=self.config.experiment.seed,
                n_startup_trials=10,
                n_ei_candidates=24,
                multivariate=True,
            )
        elif self.config.optuna.sampler == "RandomSampler":
            sampler = optuna.samplers.RandomSampler(seed=self.config.experiment.seed)
        else:
            sampler = TPESampler(seed=self.config.experiment.seed)

        # Select pruner
        if self.config.optuna.pruner == "MedianPruner":
            pruner = MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1,
            )
        elif self.config.optuna.pruner == "HyperbandPruner":
            pruner = HyperbandPruner(
                min_resource=1,
                max_resource=self.config.train.trainer.max_epochs,
                reduction_factor=3,
            )
        else:
            pruner = None

        # Create study
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            sampler=sampler,
            pruner=pruner,
            direction=self.direction,
            load_if_exists=True,
        )

        # Add baseline trial if available
        if self.baseline_metrics:
            self._add_baseline_trial(study)

        return study

    def _add_baseline_trial(self, study: Study) -> None:
        """Add baseline performance as a reference trial."""
        try:
            # Create a trial with baseline hyperparameters
            baseline_params = {
                "learning_rate": self.config.train.optimizer.lr,
                "batch_size": self.config.train.batch.train_batch_size,
                "dropout": self.config.model.dropout,
                "num_layers": self.config.model.num_layers,
                "hidden_dim": self.config.model.hidden_dim,
            }

            # Get baseline metric value
            baseline_value = self.baseline_metrics.get(self.metric, 0.0)

            # Add as a completed trial
            study.add_trial(
                optuna.trial.create_trial(
                    params=baseline_params,
                    distributions={
                        "learning_rate": optuna.distributions.LogUniformDistribution(1e-6, 1e-3),
                        "batch_size": optuna.distributions.CategoricalDistribution([512, 1024, 2048, 4096]),
                        "dropout": optuna.distributions.UniformDistribution(0.0, 0.3),
                        "num_layers": optuna.distributions.IntDistribution(4, 8),
                        "hidden_dim": optuna.distributions.CategoricalDistribution([128, 256, 512]),
                    },
                    values=[baseline_value],
                )
            )

            logger.info(f"✅ Added baseline trial with {self.metric}={baseline_value:.4f}")

        except Exception as e:
            logger.warning(f"Could not add baseline trial: {e}")

    def suggest_hyperparameters(self, trial: Trial) -> dict[str, Any]:
        """
        Suggest hyperparameters for a trial.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of suggested hyperparameters
        """
        suggestions = {}

        for param_name, param_config in self.search_space.items():
            param_type = param_config.get("type", "float")

            if param_type == "float" or param_type == "uniform":
                suggestions[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                )
            elif param_type == "loguniform":
                suggestions[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=True,
                )
            elif param_type == "int":
                suggestions[param_name] = trial.suggest_int(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                )
            elif param_type == "categorical":
                suggestions[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config["choices"],
                )
            else:
                logger.warning(f"Unknown parameter type: {param_type} for {param_name}")

        return suggestions

    def create_config_override(self, suggestions: dict[str, Any]) -> DictConfig:
        """
        Create configuration override from suggestions.

        Args:
            suggestions: Suggested hyperparameters

        Returns:
            Updated configuration
        """
        # Deep copy base config
        config = OmegaConf.create(OmegaConf.to_yaml(self.config))

        # Apply suggestions
        param_mapping = {
            "learning_rate": "train.optimizer.lr",
            "batch_size": "train.batch.train_batch_size",
            "dropout": "model.dropout",
            "num_layers": "model.num_layers",
            "hidden_dim": "model.hidden_dim",
            "weight_decay": "train.optimizer.weight_decay",
            "gradient_clip": "train.trainer.gradient_clip_val",
        }

        for param_name, config_path in param_mapping.items():
            if param_name in suggestions:
                OmegaConf.update(config, config_path, suggestions[param_name])

        return config

    def optimize(
        self,
        train_fn: Callable | None = None,
        n_trials: int | None = None,
        timeout: int | None = None,
    ) -> Study:
        """
        Run hyperparameter optimization.

        Args:
            train_fn: Training function that returns metric value
            n_trials: Number of trials to run
            timeout: Timeout in seconds

        Returns:
            Completed Optuna study
        """
        n_trials = n_trials or self.n_trials
        timeout = timeout or self.timeout
        train_fn = train_fn or self.objective_fn

        if train_fn is None:
            raise ValueError("No training function provided")

        def objective(trial: Trial) -> float:
            """Objective function for Optuna."""
            # Suggest hyperparameters
            suggestions = self.suggest_hyperparameters(trial)
            logger.info(f"Trial {trial.number}: {suggestions}")

            # Create config override
            config = self.create_config_override(suggestions)

            # Run training
            try:
                # Train model with suggested hyperparameters
                result = train_fn(config, trial)

                # Extract metric value
                if isinstance(result, dict):
                    metric_value = result.get(self.metric, 0.0)
                else:
                    metric_value = float(result)

                # Report intermediate values for pruning
                if hasattr(trial, "_study"):
                    for epoch, value in enumerate(result.get("history", [])):
                        trial.report(value, epoch)
                        if trial.should_prune():
                            raise optuna.TrialPruned()

                return metric_value

            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}")
                raise optuna.TrialPruned()

        # Run optimization
        logger.info(f"🔬 Starting hyperparameter optimization: {n_trials} trials")
        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            catch=(Exception,),
            callbacks=[self._optimization_callback],
        )

        # Log results
        self._log_optimization_results()

        return self.study

    def _optimization_callback(self, study: Study, trial: optuna.trial.FrozenTrial) -> None:
        """Callback function called after each trial."""
        logger.info(
            f"Trial {trial.number} finished: {self.metric}={trial.value:.4f} "
            f"(best: {study.best_value:.4f})"
        )

        # Save intermediate results
        if trial.number % 10 == 0:
            self._save_study_results()

    def _log_optimization_results(self) -> None:
        """Log optimization results."""
        logger.info("\n" + "=" * 80)
        logger.info("🏆 HYPERPARAMETER OPTIMIZATION RESULTS")
        logger.info("=" * 80)

        # Best trial
        best_trial = self.study.best_trial
        logger.info(f"Best {self.metric}: {best_trial.value:.4f}")
        logger.info("Best hyperparameters:")
        for key, value in best_trial.params.items():
            logger.info(f"  {key}: {value}")

        # Baseline comparison if available
        if self.baseline_metrics:
            baseline_value = self.baseline_metrics.get(self.metric, 0.0)
            improvement = (best_trial.value - baseline_value) / baseline_value * 100
            logger.info(f"\n📊 Improvement over baseline: {improvement:.1f}%")

        # Statistics
        logger.info(f"\nTotal trials: {len(self.study.trials)}")
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        logger.info(f"Completed trials: {len(completed_trials)}")
        pruned_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]
        logger.info(f"Pruned trials: {len(pruned_trials)}")

    def _save_study_results(self) -> None:
        """Save study results to file."""
        output_dir = Path(self.config.experiment.output_dir) / "optuna"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save best parameters
        best_params_file = output_dir / f"{self.study_name}_best_params.json"
        with open(best_params_file, "w") as f:
            json.dump(self.study.best_params, f, indent=2)

        # Save all trials
        trials_file = output_dir / f"{self.study_name}_trials.json"
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

        logger.info(f"💾 Study results saved to {output_dir}")

    def get_best_config(self) -> DictConfig:
        """Get configuration with best hyperparameters."""
        best_params = self.study.best_params
        return self.create_config_override(best_params)

    def visualize_optimization(self, output_dir: Path | None = None) -> None:
        """Create visualization plots for optimization results."""
        try:
            import optuna.visualization as vis
            import plotly.io as pio

            output_dir = output_dir or Path(self.config.experiment.output_dir) / "optuna" / "plots"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Optimization history
            fig = vis.plot_optimization_history(self.study)
            pio.write_html(fig, output_dir / "optimization_history.html")

            # Parameter importance
            fig = vis.plot_param_importances(self.study)
            pio.write_html(fig, output_dir / "param_importances.html")

            # Parallel coordinate plot
            fig = vis.plot_parallel_coordinate(self.study)
            pio.write_html(fig, output_dir / "parallel_coordinate.html")

            # Slice plot
            fig = vis.plot_slice(self.study)
            pio.write_html(fig, output_dir / "slice_plot.html")

            logger.info(f"📊 Visualization plots saved to {output_dir}")

        except ImportError:
            logger.warning("Plotly not installed, skipping visualization")


def create_simple_objective(model_class, data_module_class, trainer_class):
    """
    Create a simple objective function for Optuna.

    Args:
        model_class: Model class to instantiate
        data_module_class: Data module class
        trainer_class: Trainer class

    Returns:
        Objective function for Optuna
    """
    def objective(config: DictConfig, trial: Trial) -> float:
        """Train model and return validation metric."""
        # Create model
        model = model_class(config)

        # Create data module
        data_module = data_module_class(config)
        data_module.setup()

        # Create trainer
        trainer = trainer_class(
            model=model,
            config=config,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        # Train for limited epochs (for speed)
        max_epochs = min(config.train.trainer.max_epochs, 20)
        results = trainer.train(
            train_loader=data_module.train_dataloader(),
            val_loader=data_module.val_dataloader(),
            max_epochs=max_epochs,
        )

        # Return metric value
        return results["final_val_metrics"][config.optuna.metric]

    return objective
