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

        # Suggest hyperparameters matching Hydra config structure
        lr = trial.suggest_float(
            "lr", 5e-6, 5e-3, log=True
        )  # Wider range for better exploration
        # A100 80GB optimized ranges - EXPANDED for better GPU utilization
        batch_size = trial.suggest_categorical(
            "batch_size", [4096, 6144, 8192, 12288]
        )  # Larger batches
        hidden_size = trial.suggest_categorical(
            "hidden_size", [256, 384, 512, 768]
        )  # Larger models
        gat_dropout = trial.suggest_float("gat_dropout", 0.1, 0.4)
        gat_layers = trial.suggest_int("gat_layers", 2, 4)
        # NEW: Gradient accumulation for even larger effective batch sizes
        grad_accum = trial.suggest_categorical(
            "grad_accum", [1, 2, 4]
        )  # Effective batch up to 49152

        # Create HPO metrics output path
        trial_dir = self.output_dir / f"trial_{trial.number}"
        trial_dir.mkdir(exist_ok=True)
        hpo_metrics_path = trial_dir / "metrics.json"

        # ✅ FIX: Dynamically generate GAT architecture lists matching num_layers
        # hidden_channels: all layers use hidden_size
        hidden_channels_str = f"[{','.join([str(hidden_size)] * gat_layers)}]"
        # heads: first layer uses 8 heads, subsequent layers use 4 heads
        heads_str = f"[{','.join(['8'] + ['4'] * (gat_layers - 1))}]"
        # concat: all layers concatenate except the last one
        concat_str = f"[{','.join(['true'] * (gat_layers - 1) + ['false'])}]"

        # Build command with Hydra overrides matching config structure
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "integrated_ml_training_pipeline.py"),
            "--data-path",
            str(self.data_path),
            f"--max-epochs={self.max_epochs_per_trial}",
            f"--batch-size={batch_size}",
            f"--lr={lr}",
            f"model.hidden_size={hidden_size}",
            f"train.trainer.accumulate_grad_batches={grad_accum}",  # Gradient accumulation
            f"model.gat.layer_config.dropout={gat_dropout}",
            # GAT architecture with all 4 parameters
            f"model.gat.architecture.num_layers={gat_layers}",
            f"model.gat.architecture.hidden_channels={hidden_channels_str}",
            f"model.gat.architecture.heads={heads_str}",
            f"model.gat.architecture.concat={concat_str}",
            f"+hpo.output_metrics_json={hpo_metrics_path}",  # Add new parameter with + prefix
        ]

        logger.info(
            f"Trial {trial.number}: lr={lr:.2e}, batch={batch_size}, hidden={hidden_size}, gat_dropout={gat_dropout:.3f}, gat_layers={gat_layers}, grad_accum={grad_accum}"
        )

        try:
            # Run training with A100 80GB + 2TiB RAM + 256-core CPU optimized environment
            import os

            env = os.environ.copy()

            # GPU Optimization (A100 80GB)
            env["CUDA_VISIBLE_DEVICES"] = "0"
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

            # DataLoader: ENHANCED parallel loading with spawn context (fixes CPU bottleneck)
            # spawn avoids fork() thread deadlock on 256-core systems
            # IMPORTANT: Respect FORCE_SINGLE_PROCESS flag for Safe Mode
            if env.get("FORCE_SINGLE_PROCESS") == "1":
                # Safe Mode: single-process DataLoader (stability over speed)
                env["ALLOW_UNSAFE_DATALOADER"] = env.get("ALLOW_UNSAFE_DATALOADER", "0")
                env["NUM_WORKERS"] = env.get("NUM_WORKERS", "0")
                env["PERSISTENT_WORKERS"] = env.get("PERSISTENT_WORKERS", "0")
                # Keep other settings from parent environment
            else:
                # Optimized Mode: multi-worker DataLoader for better GPU utilization
                env["ALLOW_UNSAFE_DATALOADER"] = env.get("ALLOW_UNSAFE_DATALOADER", "1")
                env["NUM_WORKERS"] = env.get("NUM_WORKERS", "4")  # INCREASED: 2→4
                env["MULTIPROCESSING_CONTEXT"] = env.get(
                    "MULTIPROCESSING_CONTEXT", "spawn"
                )
                env["PREFETCH_FACTOR"] = env.get(
                    "PREFETCH_FACTOR", "4"
                )  # INCREASED: 2→4
                env["PIN_MEMORY"] = env.get("PIN_MEMORY", "1")
                env["PIN_MEMORY_DEVICE"] = env.get("PIN_MEMORY_DEVICE", "cuda:0")
                env["PERSISTENT_WORKERS"] = env.get("PERSISTENT_WORKERS", "1")

            # Memory Optimization
            env["RMM_POOL_SIZE"] = "70GB"  # 70GB for A100 80GB (留余10GB)
            env[
                "PYTORCH_CUDA_ALLOC_CONF"
            ] = "max_split_size_mb:512,expandable_segments:True"

            # Thread Optimization: Moderate thread count to avoid contention
            env["OMP_NUM_THREADS"] = "8"  # Reduced from 24 to prevent thread explosion
            env["MKL_NUM_THREADS"] = "8"  # Reduced from 24 to prevent thread explosion
            env["OPENBLAS_NUM_THREADS"] = "1"

            # Mixed Precision (bf16 for A100) - ENHANCED
            env["USE_AMP"] = "1"
            env["AMP_DTYPE"] = "bf16"
            # cuDNN optimization
            env["TORCH_CUDNN_V8_API_ENABLED"] = "1"  # Enable cuDNN V8 API
            env["CUDA_MODULE_LOADING"] = "LAZY"  # Faster startup
            env["PYTORCH_NVFUSER_DISABLE"] = "0"  # Enable NVFuser compiler

            # Loss weights: Focus on RankIC/IC over Sharpe
            env["USE_RANKIC"] = "1"
            env["RANKIC_WEIGHT"] = "0.5"  # Strong RankIC focus
            env["CS_IC_WEIGHT"] = "0.3"  # Cross-sectional IC
            env["SHARPE_WEIGHT"] = "0.1"  # Reduced Sharpe weight

            # Phase training: Respect PHASE_MAX_BATCHES if set (for minimal testing)
            # Default to full dataset (0 = no limit) if not explicitly set
            if "PHASE_MAX_BATCHES" not in env or not env.get("PHASE_MAX_BATCHES"):
                env["PHASE_MAX_BATCHES"] = "0"  # No limit (full dataset)

            # Validation logging: Reduce verbosity
            env["VAL_DEBUG_LOGGING"] = "0"  # Disable per-batch VAL-DEBUG logs
            env["AMP_DTYPE"] = "bf16"

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout (should be faster with GPU optimization)
                env=env,
            )

            # Log subprocess output for debugging
            if result.returncode != 0:
                logger.error(
                    f"Trial {trial.number} failed with return code {result.returncode}"
                )
                logger.error(f"STDERR: {result.stderr[-500:]}")  # Last 500 chars
                logger.error(f"STDOUT (last 500): {result.stdout[-500:]}")

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
                logger.warning(
                    f"Trial {trial.number}: No metrics file found at {hpo_metrics_path}"
                )
                logger.warning(f"Subprocess return code: {result.returncode}")
                logger.warning(f"Subprocess STDERR (last 200): {result.stderr[-200:]}")
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
            trials_data.append(
                {
                    "number": trial.number,
                    "params": trial.params,
                    "value": trial.value,
                    "state": str(trial.state),
                }
            )
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
