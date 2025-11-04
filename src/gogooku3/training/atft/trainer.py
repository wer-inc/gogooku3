"""
ATFT Trainer Module
学習ループと検証を管理するモジュール
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ATFTTrainer:
    """ATFT model trainer with robust training loop."""

    def __init__(
        self,
        model: nn.Module,
        config: DictConfig,
        device: torch.device,
        output_dir: Path | None = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: Model to train
            config: Training configuration
            device: Device to train on
            output_dir: Directory for outputs
        """
        self.model = model
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir) if output_dir else Path("output/training")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = float("-inf")
        self.best_checkpoint_path = None

        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []

        # Setup components
        self._setup_training_components()

    def _setup_training_components(self) -> None:
        """Set up optimizer, scheduler, and other training components."""
        # Optimizer
        self.optimizer = self._create_optimizer()

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Loss function
        self.criterion = self._create_loss_function()

        # Mixed precision training
        self.use_amp = self.config.train.trainer.precision in ["16-mixed", "bf16-mixed"]
        if self.use_amp:
            self.scaler = GradScaler()
            self.amp_dtype = torch.float16 if "16" in self.config.train.trainer.precision else torch.bfloat16
            logger.info(f"✅ Mixed precision training enabled: {self.config.train.trainer.precision}")

        # Early stopping
        self.early_stopping_patience = self.config.train.trainer.early_stopping.patience
        self.early_stopping_counter = 0
        self.early_stopping_min_delta = self.config.train.trainer.early_stopping.min_delta

        # EMA model for stability
        if self.config.train.stability.use_ema_teacher:
            self.ema_model = self._create_ema_model()
            self.ema_decay = self.config.train.stability.ema_decay
        else:
            self.ema_model = None

    def _create_optimizer(self) -> Optimizer:
        """Create optimizer."""
        optimizer_config = self.config.train.optimizer

        # Parameter groups with different learning rates
        param_groups = self._get_parameter_groups()

        if optimizer_config.type == "adamw":
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=optimizer_config.lr,
                betas=optimizer_config.betas,
                eps=optimizer_config.eps,
                weight_decay=optimizer_config.weight_decay,
            )
        elif optimizer_config.type == "adam":
            optimizer = torch.optim.Adam(
                param_groups,
                lr=optimizer_config.lr,
                betas=optimizer_config.betas,
                eps=optimizer_config.eps,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_config.type}")

        return optimizer

    def _get_parameter_groups(self) -> list[dict]:
        """Get parameter groups for optimizer."""
        # Simple version - can be extended for layer-wise learning rates
        return [{"params": self.model.parameters()}]

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        scheduler_config = self.config.train.optimizer.scheduler

        if scheduler_config.type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.T_max,
                eta_min=scheduler_config.min_lr,
            )
        elif scheduler_config.type == "linear":
            from torch.optim.lr_scheduler import LinearLR
            scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=scheduler_config.min_lr / self.config.train.optimizer.lr,
                total_iters=self.config.train.trainer.max_epochs,
            )
        else:
            scheduler = None

        return scheduler

    def _create_loss_function(self) -> nn.Module:
        """Create loss function."""
        loss_config = self.config.train.loss

        if loss_config.type == "mse":
            criterion = nn.MSELoss()
        elif loss_config.type == "huber":
            criterion = nn.HuberLoss(delta=loss_config.huber_delta)
        elif loss_config.type == "robust_mse":
            criterion = RobustMSELoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_config.type}")

        return criterion

    def _create_ema_model(self) -> nn.Module:
        """Create EMA model for stability."""
        import copy
        ema_model = copy.deepcopy(self.model)
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad = False
        return ema_model

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        max_epochs: int | None = None,
    ) -> dict[str, Any]:
        """
        Main training loop.

        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            max_epochs: Maximum epochs to train

        Returns:
            Training results dictionary
        """
        max_epochs = max_epochs or self.config.train.trainer.max_epochs
        logger.info(f"🚀 Starting training for {max_epochs} epochs")

        for epoch in range(max_epochs):
            self.current_epoch = epoch

            # Training epoch
            train_metrics = self.train_epoch(train_loader)
            self.train_metrics.append(train_metrics)

            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                self.val_metrics.append(val_metrics)

                # Check for improvement
                metric_name = self.config.train.trainer.early_stopping.monitor
                current_metric = val_metrics.get(metric_name, 0)

                if self._is_better(current_metric):
                    self.best_val_metric = current_metric
                    self.early_stopping_counter = 0
                    self._save_checkpoint(is_best=True)
                else:
                    self.early_stopping_counter += 1

                # Early stopping check
                if self.early_stopping_counter >= self.early_stopping_patience:
                    logger.info(f"⚠️ Early stopping triggered at epoch {epoch}")
                    break

            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()

            # Periodic checkpoint
            if (epoch + 1) % self.config.experiment.checkpoint_interval == 0:
                self._save_checkpoint(is_best=False)

            # Logging
            self._log_epoch_summary(epoch, train_metrics, val_metrics)

        # Final summary
        return self._create_training_summary()

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        epoch_metrics = {}

        progress_bar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}", disable=not self.config.experiment.log_interval)

        for batch_idx, (features, targets) in enumerate(progress_bar):
            # Move to device
            features = features.to(self.device)
            targets = targets.to(self.device)

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast(device_type="cuda", dtype=self.amp_dtype):
                    outputs = self.model(features)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)

            # Check for NaN
            if torch.isnan(loss):
                logger.warning(f"NaN loss detected at step {self.global_step}")
                continue

            # Backward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config.train.trainer.gradient_clip_val > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.train.trainer.gradient_clip_val
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                # Gradient clipping
                if self.config.train.trainer.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.train.trainer.gradient_clip_val
                    )

                self.optimizer.step()

            # Update EMA model
            if self.ema_model is not None:
                self._update_ema_model()

            # Track metrics
            epoch_losses.append(loss.item())
            self.global_step += 1

            # Update progress bar
            if batch_idx % 10 == 0:
                progress_bar.set_postfix({
                    "loss": np.mean(epoch_losses[-100:]) if epoch_losses else 0,
                    "lr": self.optimizer.param_groups[0]["lr"],
                })

        # Compute epoch metrics
        epoch_metrics["loss"] = np.mean(epoch_losses)
        epoch_metrics["lr"] = self.optimizer.param_groups[0]["lr"]

        return epoch_metrics

    def validate(self, dataloader: DataLoader) -> dict[str, float]:
        """Validate the model."""
        self.model.eval()
        val_losses = []
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for features, targets in tqdm(dataloader, desc="Validation", disable=True):
                features = features.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                if self.use_amp:
                    with autocast(device_type="cuda", dtype=self.amp_dtype):
                        outputs = self.model(features)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(features)
                    loss = self.criterion(outputs, targets)

                val_losses.append(loss.item())
                val_predictions.append(outputs.cpu())
                val_targets.append(targets.cpu())

        # Compute metrics
        val_predictions = torch.cat(val_predictions)
        val_targets = torch.cat(val_targets)

        metrics = {
            "loss": np.mean(val_losses),
            "ic": self._compute_ic(val_predictions, val_targets),
            "rank_ic": self._compute_rank_ic(val_predictions, val_targets),
            "sharpe": self._compute_sharpe(val_predictions, val_targets),
        }

        return metrics

    def _update_ema_model(self) -> None:
        """Update EMA model parameters."""
        if self.ema_model is None:
            return

        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def _is_better(self, current_metric: float) -> bool:
        """Check if current metric is better than best."""
        mode = self.config.train.trainer.early_stopping.mode
        min_delta = self.early_stopping_min_delta

        if mode == "max":
            return current_metric > self.best_val_metric + min_delta
        else:
            return current_metric < self.best_val_metric - min_delta

    def _save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_metric": self.best_val_metric,
            "config": self.config,
        }

        # Add EMA model if available
        if self.ema_model is not None:
            checkpoint["ema_model_state_dict"] = self.ema_model.state_dict()

        # Save checkpoint
        if is_best:
            checkpoint_path = self.output_dir / "best_model.pt"
            self.best_checkpoint_path = checkpoint_path
        else:
            checkpoint_path = self.output_dir / f"checkpoint_epoch_{self.current_epoch}.pt"

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"💾 Saved checkpoint: {checkpoint_path}")

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints keeping only top-k."""
        keep_top_k = self.config.performance.save_top_k_checkpoints
        checkpoints = sorted(self.output_dir.glob("checkpoint_epoch_*.pt"))

        if len(checkpoints) > keep_top_k:
            for checkpoint in checkpoints[:-keep_top_k]:
                checkpoint.unlink()
                logger.info(f"🗑️ Removed old checkpoint: {checkpoint}")

    def _compute_ic(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute information coefficient."""
        # Simple correlation-based IC
        pred_flat = predictions.view(-1).numpy()
        target_flat = targets.view(-1).numpy()
        return np.corrcoef(pred_flat, target_flat)[0, 1]

    def _compute_rank_ic(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute rank information coefficient."""
        from scipy.stats import spearmanr
        pred_flat = predictions.view(-1).numpy()
        target_flat = targets.view(-1).numpy()
        return spearmanr(pred_flat, target_flat)[0]

    def _compute_sharpe(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute Sharpe ratio."""
        # Simple Sharpe calculation
        returns = predictions.mean(dim=1).numpy()
        if len(returns) > 1:
            sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
        else:
            sharpe = 0
        return sharpe

    def _log_epoch_summary(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ) -> None:
        """Log epoch summary."""
        summary = f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}"

        if val_metrics:
            summary += f", Val Loss={val_metrics['loss']:.4f}"
            summary += f", IC={val_metrics.get('ic', 0):.3f}"
            summary += f", RankIC={val_metrics.get('rank_ic', 0):.3f}"
            summary += f", Sharpe={val_metrics.get('sharpe', 0):.3f}"

        logger.info(summary)

    def _create_training_summary(self) -> dict[str, Any]:
        """Create training summary."""
        summary = {
            "total_epochs": self.current_epoch + 1,
            "total_steps": self.global_step,
            "best_val_metric": self.best_val_metric,
            "best_checkpoint": str(self.best_checkpoint_path) if self.best_checkpoint_path else None,
            "final_train_loss": self.train_metrics[-1]["loss"] if self.train_metrics else None,
            "final_val_metrics": self.val_metrics[-1] if self.val_metrics else None,
            "training_time": time.time(),  # Would need to track start time
        }

        # Save metrics history
        metrics_path = self.output_dir / "training_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({
                "train_metrics": self.train_metrics,
                "val_metrics": self.val_metrics,
                "summary": summary,
            }, f, indent=2)

        return summary


class RobustMSELoss(nn.Module):
    """Robust MSE loss with outlier handling."""

    def __init__(self, outlier_threshold: float = 3.0):
        super().__init__()
        self.outlier_threshold = outlier_threshold

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute robust MSE loss."""
        diff = pred - target
        mse = diff.pow(2)

        # Clip outliers
        std = diff.std()
        mean = diff.mean()
        threshold = self.outlier_threshold * std

        mask = (diff - mean).abs() < threshold
        robust_mse = mse[mask].mean() if mask.any() else mse.mean()

        return robust_mse
