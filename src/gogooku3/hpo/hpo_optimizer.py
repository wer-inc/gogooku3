"""
ATFT-GAT-FAN HPO Optimizer with GPU acceleration and multi-horizon objectives
Optuna-based hyperparameter optimization with financial metrics focus
"""

import logging
import os
import json
import time
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.trial import TrialState
import torch
import subprocess
import tempfile
import shutil

from .objectives import MultiHorizonObjective
from .metrics_extractor import MetricsExtractor, TrainingMetrics

logger = logging.getLogger(__name__)


class ATFTHPOOptimizer:
    """ATFT-GAT-FAN Hyperparameter Optimizer with Optuna integration"""

    def __init__(self,
                 study_name: str = "atft_hpo",
                 storage: Optional[str] = None,
                 n_trials: int = 100,
                 n_jobs: int = 1,
                 gpu_memory_fraction: float = 0.8,
                 timeout: Optional[float] = None,
                 base_config_path: Optional[str] = None,
                 load_if_exists: bool = True):
        """
        Initialize ATFT HPO optimizer

        Args:
            study_name: Optuna study name
            storage: Database URL for study persistence
            n_trials: Number of optimization trials
            n_jobs: Number of parallel jobs
            gpu_memory_fraction: GPU memory usage limit
            timeout: Optimization timeout in seconds
            base_config_path: Base configuration file path
            load_if_exists: Load existing study if available
        """
        self.study_name = study_name
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.gpu_memory_fraction = gpu_memory_fraction
        self.timeout = timeout
        self.base_config_path = base_config_path
        self.load_if_exists = load_if_exists

        # Setup storage with environment variable support
        self.storage = self._setup_storage(storage)

        # Initialize components
        self.objective_func = MultiHorizonObjective()
        self.metrics_extractor = MetricsExtractor()

        # Setup Optuna
        self.sampler = TPESampler(
            n_startup_trials=20,
            n_ei_candidates=24,
            multivariate=True,
            group=True,
            warn_independent_sampling=True
        )

        self.pruner = MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=5,
            interval_steps=1,
            n_min_trials=5
        )

        # GPU optimization settings
        self._setup_gpu_optimization()

        # Working directory for trials
        self.work_dir = Path("hpo_trials")
        self.work_dir.mkdir(exist_ok=True)

    def _setup_storage(self, storage: Optional[str]) -> Optional[str]:
        """
        Setup Optuna storage with environment variable support

        Args:
            storage: Explicit storage URL or None

        Returns:
            Storage URL or None for in-memory
        """
        try:
            # Priority: explicit parameter > environment variable > default
            if storage:
                storage_url = storage
            else:
                storage_url = os.getenv('OPTUNA_STORAGE_URL')

            if storage_url:
                # Ensure output directory exists for SQLite
                if storage_url.startswith('sqlite:///'):
                    db_path = Path(storage_url.replace('sqlite:///', ''))
                    db_path.parent.mkdir(parents=True, exist_ok=True)
                    logger.info(f"📊 Using SQLite storage: {db_path}")
                elif storage_url.startswith('postgresql://'):
                    logger.info(f"🐘 Using PostgreSQL storage: {storage_url.split('@')[-1]}")  # Hide credentials
                else:
                    logger.info(f"💾 Using custom storage: {storage_url}")

                return storage_url
            else:
                logger.info("🧠 Using in-memory storage (no persistence)")
                return None

        except Exception as e:
            logger.warning(f"Failed to setup storage: {e}")
            logger.info("Falling back to in-memory storage")
            return None

    def _setup_gpu_optimization(self):
        """Configure GPU settings for optimal performance"""
        try:
            if torch.cuda.is_available():
                # Set memory fraction
                torch.cuda.set_per_process_memory_fraction(self.gpu_memory_fraction)

                # Enable expandable segments
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

                # Enable mixed precision
                os.environ['ENABLE_MIXED_PRECISION'] = 'true'

                logger.info(f"🚀 GPU optimization configured: memory_fraction={self.gpu_memory_fraction}")
                logger.info(f"   Available GPUs: {torch.cuda.device_count()}")
                logger.info(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

        except Exception as e:
            logger.warning(f"Failed to setup GPU optimization: {e}")

    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for ATFT-GAT-FAN model

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of suggested hyperparameters
        """
        params = {}

        # Model architecture parameters
        params.update({
            # ATFT parameters
            'atft_d_model': trial.suggest_categorical('atft_d_model', [128, 256, 384, 512]),
            'atft_n_heads': trial.suggest_categorical('atft_n_heads', [4, 6, 8, 12, 16]),
            'atft_n_layers': trial.suggest_int('atft_n_layers', 2, 6),
            'atft_d_ff': trial.suggest_categorical('atft_d_ff', [256, 512, 1024, 2048]),
            'atft_dropout': trial.suggest_float('atft_dropout', 0.1, 0.5),

            # GAT parameters
            'gat_hidden_dim': trial.suggest_categorical('gat_hidden_dim', [64, 128, 256, 384]),
            'gat_n_heads': trial.suggest_categorical('gat_n_heads', [4, 6, 8, 12]),
            'gat_n_layers': trial.suggest_int('gat_n_layers', 2, 4),
            'gat_dropout': trial.suggest_float('gat_dropout', 0.1, 0.4),
            'gat_alpha': trial.suggest_float('gat_alpha', 0.01, 0.3, log=True),

            # FAN parameters
            'fan_n_frequencies': trial.suggest_categorical('fan_n_frequencies', [16, 32, 64, 128]),
            'fan_hidden_dim': trial.suggest_categorical('fan_hidden_dim', [64, 128, 256]),
            'fan_dropout': trial.suggest_float('fan_dropout', 0.1, 0.4),
        })

        # Training parameters
        params.update({
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512, 1024]),
            'gradient_clip_val': trial.suggest_float('gradient_clip_val', 0.5, 2.0),
            'warmup_steps': trial.suggest_int('warmup_steps', 100, 2000),
        })

        # Loss function parameters
        horizon_weights = self.objective_func.suggest_horizon_weights(trial)
        params['horizon_weights'] = horizon_weights

        # Optimizer parameters
        params.update({
            'optimizer': trial.suggest_categorical('optimizer', ['adamw', 'lamb', 'adafactor']),
            'scheduler': trial.suggest_categorical('scheduler', ['cosine', 'linear', 'polynomial']),
            'beta1': trial.suggest_float('beta1', 0.85, 0.95),
            'beta2': trial.suggest_float('beta2', 0.95, 0.999),
            'eps': trial.suggest_float('eps', 1e-9, 1e-7, log=True),
        })

        # Regularization parameters
        params.update({
            'label_smoothing': trial.suggest_float('label_smoothing', 0.0, 0.1),
            'mixup_alpha': trial.suggest_float('mixup_alpha', 0.0, 0.4),
            'cutmix_alpha': trial.suggest_float('cutmix_alpha', 0.0, 0.4),
        })

        return params

    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function

        Args:
            trial: Optuna trial object

        Returns:
            Objective score (higher is better)
        """
        try:
            # Suggest hyperparameters
            params = self.suggest_hyperparameters(trial)

            # Log trial start
            logger.info(f"🎯 Starting trial {trial.number}")
            logger.info(f"   Parameters: {len(params)} hyperparameters")

            # Create trial directory
            trial_dir = self.work_dir / f"trial_{trial.number}"
            trial_dir.mkdir(exist_ok=True)

            # Generate config file
            config_path = trial_dir / "trial_config.yaml"
            self._create_trial_config(params, config_path)

            # Run training
            metrics = self._run_training_trial(config_path, trial_dir, trial)

            if metrics is None:
                logger.warning(f"❌ Trial {trial.number} failed - no metrics extracted")
                return -1.0

            # Compute objective score
            score = self.objective_func.compute_score({
                'rank_ic': metrics.rank_ic,
                'sharpe': metrics.sharpe
            })

            # Report metrics to Optuna for pruning
            if hasattr(trial, 'report'):
                trial.report(score, step=metrics.epoch)

            # Log results
            logger.info(f"✅ Trial {trial.number} completed: score={score:.4f}")
            logger.info(f"   {self.metrics_extractor.format_metrics(metrics)}")

            # Cleanup trial directory
            if trial_dir.exists():
                shutil.rmtree(trial_dir)

            return score

        except optuna.TrialPruned:
            logger.info(f"✂️ Trial {trial.number} pruned")
            raise
        except Exception as e:
            logger.error(f"❌ Trial {trial.number} failed: {e}")
            return -1.0

    def _create_trial_config(self, params: Dict[str, Any], config_path: Path):
        """Create Hydra config file for trial"""
        try:
            # Load base config if provided
            if self.base_config_path and Path(self.base_config_path).exists():
                import yaml
                with open(self.base_config_path, 'r') as f:
                    base_config = yaml.safe_load(f)
            else:
                base_config = {}

            # Update with trial parameters
            config = base_config.copy()

            # Map parameters to config structure
            config.update({
                'model': {
                    'atft': {
                        'd_model': params['atft_d_model'],
                        'n_heads': params['atft_n_heads'],
                        'n_layers': params['atft_n_layers'],
                        'd_ff': params['atft_d_ff'],
                        'dropout': params['atft_dropout']
                    },
                    'gat': {
                        'hidden_dim': params['gat_hidden_dim'],
                        'n_heads': params['gat_n_heads'],
                        'n_layers': params['gat_n_layers'],
                        'dropout': params['gat_dropout'],
                        'alpha': params['gat_alpha']
                    },
                    'fan': {
                        'n_frequencies': params['fan_n_frequencies'],
                        'hidden_dim': params['fan_hidden_dim'],
                        'dropout': params['fan_dropout']
                    }
                },
                'train': {
                    'learning_rate': params['learning_rate'],
                    'weight_decay': params['weight_decay'],
                    'batch_size': params['batch_size'],
                    'gradient_clip_val': params['gradient_clip_val'],
                    'warmup_steps': params['warmup_steps'],
                    'optimizer': params['optimizer'],
                    'scheduler': params['scheduler'],
                    'max_epochs': 5,  # Reduced for HPO
                    'early_stopping_patience': 3
                },
                'loss': {
                    'horizon_weights': params['horizon_weights'],
                    'label_smoothing': params['label_smoothing']
                }
            })

            # Save config
            import yaml
            with open(config_path, 'w') as f:
                yaml.safe_dump(config, f, default_flow_style=False)

            logger.debug(f"Created trial config: {config_path}")

        except Exception as e:
            logger.error(f"Failed to create trial config: {e}")
            raise

    def _run_training_trial(self,
                          config_path: Path,
                          trial_dir: Path,
                          trial: optuna.Trial) -> Optional[TrainingMetrics]:
        """Run training for a single trial"""
        try:
            # Setup environment
            env = os.environ.copy()
            env.update({
                'CUDA_VISIBLE_DEVICES': '0',  # Use first GPU
                'PYTHONPATH': '/home/ubuntu/gogooku3-standalone',
                'HPO_TRIAL_NUMBER': str(trial.number),
                'HPO_TRIAL_DIR': str(trial_dir)
            })

            # Training command
            cmd = [
                'python', 'scripts/train_atft.py',
                '--config-path', str(config_path.parent),
                '--config-name', config_path.stem,
                '--output-dir', str(trial_dir),
                '--experiment-name', f'hpo_trial_{trial.number}'
            ]

            # Run training
            logger.debug(f"Running: {' '.join(cmd)}")
            start_time = time.time()

            result = subprocess.run(
                cmd,
                cwd='/home/ubuntu/gogooku3-standalone',
                env=env,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout per trial
            )

            training_time = time.time() - start_time

            # Check if training succeeded
            if result.returncode != 0:
                logger.error(f"Training failed for trial {trial.number}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                return None

            # Extract metrics from logs
            metrics = self._extract_trial_metrics(trial_dir, result.stdout, training_time)

            return metrics

        except subprocess.TimeoutExpired:
            logger.warning(f"Trial {trial.number} timed out")
            return None
        except Exception as e:
            logger.error(f"Failed to run training trial: {e}")
            return None

    def _extract_trial_metrics(self,
                             trial_dir: Path,
                             stdout: str,
                             training_time: float) -> Optional[TrainingMetrics]:
        """Extract metrics from trial results"""
        try:
            # Try multiple extraction methods

            # 1. Extract from log files
            log_files = list(trial_dir.glob("*.log"))
            if log_files:
                metrics = self.metrics_extractor.extract_from_log_file(log_files[0])
                if metrics:
                    metrics.training_time = training_time
                    return metrics

            # 2. Extract from stdout
            if stdout:
                metrics = self.metrics_extractor._parse_log_content(stdout)
                if metrics:
                    metrics.training_time = training_time
                    return metrics

            # 3. Extract from TensorBoard logs
            tb_dirs = list(trial_dir.glob("**/tensorboard*"))
            if tb_dirs:
                metrics = self.metrics_extractor.extract_from_tensorboard(tb_dirs[0])
                if metrics:
                    metrics.training_time = training_time
                    return metrics

            logger.warning(f"No metrics extracted for trial {trial_dir}")
            return None

        except Exception as e:
            logger.error(f"Failed to extract trial metrics: {e}")
            return None

    def optimize(self) -> optuna.Study:
        """
        Run hyperparameter optimization

        Returns:
            Completed Optuna study
        """
        try:
            logger.info(f"🚀 Starting ATFT HPO optimization")
            logger.info(f"   Study: {self.study_name}")
            logger.info(f"   Trials: {self.n_trials}")
            logger.info(f"   Jobs: {self.n_jobs}")
            logger.info(f"   Timeout: {self.timeout}s" if self.timeout else "   Timeout: None")

            # Create or load study
            study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage,
                direction='maximize',
                sampler=self.sampler,
                pruner=self.pruner,
                load_if_exists=self.load_if_exists
            )

            # Log study information
            if self.storage and self.load_if_exists:
                completed_trials = len([t for t in study.trials if t.state == TrialState.COMPLETE])
                if completed_trials > 0:
                    logger.info(f"📈 Loaded existing study with {completed_trials} completed trials")
                    logger.info(f"   Current best score: {study.best_value:.4f}")
                else:
                    logger.info("📋 Created new study (no previous trials found)")
            else:
                logger.info("📋 Created new in-memory study")

            # Run optimization
            study.optimize(
                self.objective,
                n_trials=self.n_trials,
                n_jobs=self.n_jobs,
                timeout=self.timeout
            )

            # Log results
            logger.info(f"✅ HPO optimization completed")
            logger.info(f"   Best score: {study.best_value:.4f}")
            logger.info(f"   Total trials: {len(study.trials)}")
            logger.info(f"   Completed trials: {len([t for t in study.trials if t.state == optuna.TrialState.COMPLETE])}")

            # Format best parameters
            best_params_formatted = self.objective_func.format_best_params(study.best_params)
            logger.info(f"\n{best_params_formatted}")

            return study

        except Exception as e:
            logger.error(f"HPO optimization failed: {e}")
            raise

    def get_best_config(self, study: optuna.Study) -> Dict[str, Any]:
        """
        Generate best configuration from study

        Args:
            study: Completed Optuna study

        Returns:
            Best hyperparameter configuration
        """
        best_params = study.best_params.copy()

        # Convert to structured config
        config = {
            'model': {
                'atft': {
                    'd_model': best_params['atft_d_model'],
                    'n_heads': best_params['atft_n_heads'],
                    'n_layers': best_params['atft_n_layers'],
                    'd_ff': best_params['atft_d_ff'],
                    'dropout': best_params['atft_dropout']
                },
                'gat': {
                    'hidden_dim': best_params['gat_hidden_dim'],
                    'n_heads': best_params['gat_n_heads'],
                    'n_layers': best_params['gat_n_layers'],
                    'dropout': best_params['gat_dropout'],
                    'alpha': best_params['gat_alpha']
                },
                'fan': {
                    'n_frequencies': best_params['fan_n_frequencies'],
                    'hidden_dim': best_params['fan_hidden_dim'],
                    'dropout': best_params['fan_dropout']
                }
            },
            'train': {
                'learning_rate': best_params['learning_rate'],
                'weight_decay': best_params['weight_decay'],
                'batch_size': best_params['batch_size'],
                'gradient_clip_val': best_params['gradient_clip_val'],
                'warmup_steps': best_params['warmup_steps'],
                'optimizer': best_params['optimizer'],
                'scheduler': best_params['scheduler']
            },
            'loss': {
                'horizon_weights': {
                    '1d': best_params.get('weight_1d', 1.0),
                    '5d': best_params.get('weight_5d', 1.0),
                    '10d': best_params.get('weight_10d', 1.0),
                    '20d': best_params.get('weight_20d', 1.0)
                },
                'label_smoothing': best_params['label_smoothing']
            },
            'hpo_metadata': {
                'study_name': study.study_name,
                'best_value': study.best_value,
                'n_trials': len(study.trials),
                'optimization_time': sum(
                    (t.datetime_complete - t.datetime_start).total_seconds()
                    for t in study.trials
                    if t.datetime_complete and t.datetime_start
                )
            }
        }

        return config

    def save_best_config(self, study: optuna.Study, output_path: Union[str, Path]):
        """Save best configuration to file"""
        try:
            config = self.get_best_config(study)

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            import yaml
            with open(output_path, 'w') as f:
                yaml.safe_dump(config, f, default_flow_style=False)

            logger.info(f"💾 Best config saved to: {output_path}")

        except Exception as e:
            logger.error(f"Failed to save best config: {e}")
            raise

    def get_study_status(self) -> Dict[str, Any]:
        """
        Get current study status and statistics

        Returns:
            Dictionary with study status information
        """
        try:
            if not self.storage:
                logger.warning("Cannot get study status: no persistent storage configured")
                return {"error": "No persistent storage"}

            # Load existing study
            study = optuna.load_study(
                study_name=self.study_name,
                storage=self.storage
            )

            # Collect statistics
            all_trials = study.trials
            completed_trials = [t for t in all_trials if t.state == TrialState.COMPLETE]
            failed_trials = [t for t in all_trials if t.state == TrialState.FAIL]
            pruned_trials = [t for t in all_trials if t.state == TrialState.PRUNED]

            status = {
                "study_name": self.study_name,
                "storage": self.storage,
                "total_trials": len(all_trials),
                "completed_trials": len(completed_trials),
                "failed_trials": len(failed_trials),
                "pruned_trials": len(pruned_trials),
                "success_rate": len(completed_trials) / len(all_trials) * 100 if all_trials else 0,
                "best_value": study.best_value if completed_trials else None,
                "best_params": study.best_params if completed_trials else None
            }

            # Add timing information
            if completed_trials:
                durations = []
                for trial in completed_trials:
                    if trial.datetime_start and trial.datetime_complete:
                        duration = (trial.datetime_complete - trial.datetime_start).total_seconds()
                        durations.append(duration)

                if durations:
                    status["avg_trial_duration"] = np.mean(durations)
                    status["total_optimization_time"] = sum(durations)

            return status

        except Exception as e:
            logger.error(f"Failed to get study status: {e}")
            return {"error": str(e)}

    def resume_study(self, additional_trials: Optional[int] = None) -> optuna.Study:
        """
        Resume optimization of an existing study

        Args:
            additional_trials: Additional trials to run (if None, uses remaining trials)

        Returns:
            Updated study object
        """
        try:
            if not self.storage:
                raise ValueError("Cannot resume study: no persistent storage configured")

            logger.info(f"🔄 Resuming optimization for study: {self.study_name}")

            # Load existing study
            study = optuna.load_study(
                study_name=self.study_name,
                storage=self.storage
            )

            # Get current status
            status = self.get_study_status()
            logger.info(f"   Current status: {status['completed_trials']}/{status['total_trials']} trials completed")
            if status['best_value']:
                logger.info(f"   Current best score: {status['best_value']:.4f}")

            # Calculate remaining trials
            if additional_trials:
                remaining_trials = additional_trials
            else:
                remaining_trials = max(0, self.n_trials - status['completed_trials'])

            if remaining_trials <= 0:
                logger.info("✅ Optimization already completed (no remaining trials)")
                return study

            logger.info(f"🎯 Running {remaining_trials} additional trials")

            # Continue optimization
            study.optimize(
                self.objective,
                n_trials=remaining_trials,
                n_jobs=self.n_jobs,
                timeout=self.timeout
            )

            # Log final results
            final_status = self.get_study_status()
            logger.info(f"✅ Resume completed")
            logger.info(f"   Final: {final_status['completed_trials']}/{final_status['total_trials']} trials")
            logger.info(f"   Best score: {final_status['best_value']:.4f}")

            return study

        except Exception as e:
            logger.error(f"Failed to resume study: {e}")
            raise