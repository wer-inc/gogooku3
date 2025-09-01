"""
Robust Trainer with EMA, GradScaler, and Advanced Scheduling
ATFT-GAT-FAN向け最適化トレーニング
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
import numpy as np
from scipy.stats import spearmanr
import time

from ..utils.settings import get_settings
from ..losses.multi_horizon_loss import ComprehensiveLoss

logger = logging.getLogger(__name__)
config = get_settings()


class ModelEMA:
    """Exponential Moving Average for Model Parameters"""

    def __init__(self, model: nn.Module, decay: float = 0.999, device: Optional[str] = None):
        """
        Args:
            model: ターゲットモデル
            decay: EMA減衰率
            device: デバイス
        """
        self.model = model
        self.decay = decay
        self.device = device or next(model.parameters()).device
        self.shadow = {}
        self.backup = {}

        # 初期化
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.clone().detach()

        logger.info(f"ModelEMA initialized with decay={decay}")

    @torch.no_grad()
    def update(self, model: nn.Module):
        """EMA更新"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param + self.decay * self.shadow[name]
                self.shadow[name].copy_(new_average)

    def apply_shadow(self):
        """シャドウパラメータをモデルに適用"""
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """バックアップからパラメータを復元"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        """EMAパラメータを指定モデルにコピー"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])


class AdvancedScheduler:
    """Warmup + Cosine Annealing + Plateau Detection"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 1500,
        max_steps: int = 10000,
        base_lr: float = 5e-4,
        min_lr: float = 1e-6,
        plateau_patience: int = 5,
        plateau_factor: float = 0.5
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.plateau_patience = plateau_patience
        self.plateau_factor = plateau_factor

        self.current_step = 0
        self.plateau_count = 0
        self.best_metric = float('inf')
        self.in_plateau_mode = False

        # Cosineスケジューラ
        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_steps - warmup_steps, eta_min=min_lr
        )

        logger.info(f"AdvancedScheduler initialized: warmup={warmup_steps}, max_steps={max_steps}")

    def step(self, metric: Optional[float] = None):
        """スケジューラステップ"""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Warmupフェーズ
            progress = self.current_step / self.warmup_steps
            lr = self.base_lr * progress

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        else:
            # Plateau検知
            if metric is not None:
                if metric < self.best_metric:
                    self.best_metric = metric
                    self.plateau_count = 0
                else:
                    self.plateau_count += 1

                # Plateauモードに移行
                if self.plateau_count >= self.plateau_patience and not self.in_plateau_mode:
                    self.in_plateau_mode = True
                    logger.info(f"Entering plateau mode at step {self.current_step}")

                    # 学習率を半減
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= self.plateau_factor

                    logger.info(f"Reduced LR to {param_group['lr']}")

            # Cosineスケジューリング（Plateauモードでない場合）
            if not self.in_plateau_mode:
                self.cosine_scheduler.step()

    def get_lr(self) -> float:
        """現在の学習率を取得"""
        return self.optimizer.param_groups[0]['lr']


class OptimizedOptimizer:
    """ParamGroup最適化オプティマイザー"""

    def __init__(self, model: nn.Module, config: Any):
        self.model = model
        self.config = config

        # デバッグ: config属性を確認
        logger.info(f"Config attributes: {[attr for attr in dir(config) if not attr.startswith('_')]}")
        if hasattr(config, 'train') and hasattr(config.train, 'optimizer'):
            logger.info(f"train.optimizer attributes: {[attr for attr in dir(config.train.optimizer) if not attr.startswith('_')]}")
            logger.info(f"base_lr: {getattr(config.train.optimizer, 'base_lr', 'NOT_FOUND')}")
            logger.info(f"gat_lr_multiplier: {getattr(config.train.optimizer, 'gat_lr_multiplier', 'NOT_FOUND')}")
        else:
            logger.info("config.train.optimizer not found")

        self.param_groups = self._build_param_groups()

        # AdamWオプティマイザ
        opt_config = config.train.optimizer
        self.optimizer = AdamW(
            self.param_groups,
            lr=opt_config.base_lr,
            weight_decay=opt_config.base_weight_decay,
            betas=tuple(opt_config.betas),
            eps=opt_config.eps
        )

        logger.info(f"OptimizedOptimizer initialized with {len(self.param_groups)} param groups")

    def _build_param_groups(self) -> List[Dict[str, Any]]:
        """ParamGroupの構築"""
        decay_params = []
        no_decay_params = []
        fan_params = []
        gat_params = []

        opt_config = self.config.train.optimizer

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # FANパラメータ
            if 'fan' in name.lower() and ('gain' in name.lower() or 'band' in name.lower()):
                fan_params.append(param)
            # GATパラメータ
            elif 'gat' in name.lower() and ('attn' in name.lower() or 'alpha' in name.lower()):
                gat_params.append(param)
            # ノーマライゼーション/バイアス/埋め込み
            elif (param.dim() == 1 or
                  name.endswith('.bias') or
                  'norm' in name.lower() or
                  'bn' in name.lower() or
                  'embedding' in name.lower()):
                no_decay_params.append(param)
            # その他（decay適用）
            else:
                decay_params.append(param)

        param_groups = []

        # Decay適用グループ
        if decay_params:
            param_groups.append({
                'params': decay_params,
                'lr': opt_config.base_lr,
                'weight_decay': opt_config.base_weight_decay,
                'name': 'decay'
            })

        # No-decayグループ
        if no_decay_params:
            param_groups.append({
                'params': no_decay_params,
                'lr': opt_config.base_lr,
                'weight_decay': 0.0,
                'name': 'no_decay'
            })

        # FANグループ
        if fan_params:
            param_groups.append({
                'params': fan_params,
                'lr': opt_config.base_lr * opt_config.fan_lr_multiplier,
                'weight_decay': opt_config.base_weight_decay * opt_config.fan_wd_multiplier,
                'name': 'fan'
            })

        # GATグループ
        if gat_params:
            param_groups.append({
                'params': gat_params,
                'lr': opt_config.base_lr * opt_config.gat_lr_multiplier,
                'weight_decay': opt_config.base_weight_decay * opt_config.gat_wd_multiplier,
                'name': 'gat'
            })

        logger.info(f"Built param groups: {[g['name'] for g in param_groups]}")
        return param_groups


class RobustTrainer:
    """
    堅牢なトレーニングクラス
    EMA + GradScaler + Advanced Scheduling + 自動回復
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: Any,
        val_loader: Any,
        config: Any,
        checkpoint_dir: str = "./checkpoints"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        # オプティマイザとスケジューラ
        self.optimizer = OptimizedOptimizer(model, config).optimizer
        self.scheduler = AdvancedScheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_epochs * len(train_loader),
            base_lr=config.base_lr,
            min_lr=1e-6
        )

        # EMA
        self.ema = ModelEMA(model, decay=config.ema_decay) if config.ema_decay < 1.0 else None

        # GradScaler
        self.scaler = GradScaler(
            init_scale=2**16,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000,
            enabled=config.mixed_precision
        )

        # 損失関数
        self.criterion = ComprehensiveLoss(
            horizons=config.prediction_horizons,
            horizon_weights=config.horizon_weights,
            huber_delta=config.huber_delta,
            rankic_weight=config.rankic_weight,
            sharpe_weight=config.sharpe_weight
        )

        # トレーニング状態
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # メトリクスバッファ
        self.train_metrics = []
        self.val_metrics = []

        logger.info("RobustTrainer initialized")

    def train_epoch(self) -> Dict[str, float]:
        """1エポックトレーニング"""
        self.model.train()
        epoch_loss = 0.0
        epoch_metrics = {
            'loss': 0.0,
            'grad_norm': 0.0,
            'lr': 0.0
        }

        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            step_loss, step_metrics = self.training_step(batch, batch_idx)

            epoch_loss += step_loss
            for key, value in step_metrics.items():
                epoch_metrics[key] += value

            self.global_step += 1

            # ログ
            if batch_idx % 10 == 0:
                logger.info(f"Step {batch_idx}: loss={step_loss:.4f}, lr={self.scheduler.get_lr():.6f}")

        # エポック平均
        n_steps = len(self.train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= n_steps

        epoch_metrics['epoch_time'] = time.time() - start_time
        epoch_metrics['steps_per_sec'] = n_steps / epoch_metrics['epoch_time']

        return epoch_metrics

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Tuple[float, Dict[str, float]]:
        """トレーニングステップ"""
        features = batch['features']
        targets = batch['targets']

        # Forward pass with AMP
        with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
            outputs = self.model(features)
            loss = self.criterion(outputs, targets, self.model)

        # Backward pass
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()

        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.grad_clip_val
        )

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # EMA update
        if self.ema is not None:
            self.ema.update(self.model)

        # Scheduler step
        self.scheduler.step()

        step_metrics = {
            'loss': loss.item(),
            'grad_norm': grad_norm.item(),
            'lr': self.scheduler.get_lr()
        }

        return loss.item(), step_metrics

    def validate(self) -> Dict[str, float]:
        """検証"""
        if self.ema is not None:
            self.ema.apply_shadow()

        self.model.eval()
        val_loss = 0.0
        val_metrics = {
            'loss': 0.0,
            'rankic_h1': 0.0,
            'pred_std_h1': 0.0,
            'target_std_h1': 0.0
        }

        all_preds_h1 = []
        all_targets_h1 = []

        with torch.no_grad():
            for batch in self.val_loader:
                features = batch['features']
                targets = batch['targets']

                outputs = self.model(features)
                loss = self.criterion(outputs, targets)

                val_loss += loss.item()

                # h1のメトリクス収集
                if 'h1' in outputs and 'h1' in targets:
                    preds = outputs['h1'].cpu().numpy().flatten()
                    targets_h1 = targets['h1'].cpu().numpy().flatten()

                    all_preds_h1.extend(preds)
                    all_targets_h1.extend(targets_h1)

        # 検証メトリクス計算
        n_batches = len(self.val_loader)
        val_metrics['loss'] = val_loss / n_batches

        if all_preds_h1 and all_targets_h1:
            all_preds_h1 = np.array(all_preds_h1)
            all_targets_h1 = np.array(all_targets_h1)

            # RankIC
            if len(all_preds_h1) > 10:
                rankic = spearmanr(all_preds_h1, all_targets_h1).correlation
                val_metrics['rankic_h1'] = rankic if not np.isnan(rankic) else 0.0

            # Std比
            pred_std = np.std(all_preds_h1)
            target_std = np.std(all_targets_h1)
            val_metrics['pred_std_h1'] = pred_std
            val_metrics['target_std_h1'] = target_std

        # EMA復元
        if self.ema is not None:
            self.ema.restore()

        return val_metrics

    def save_checkpoint(self, val_metrics: Dict[str, float], is_best: bool = False):
        """チェックポイント保存"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.__dict__,
            'scaler_state_dict': self.scaler.state_dict(),
            'val_metrics': val_metrics,
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else self.config
        }

        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.shadow

        # 通常チェックポイント
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # 最良チェックポイント
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")

            # EMAモデルも保存
            if self.ema is not None:
                ema_path = self.checkpoint_dir / "best_model_ema.pth"
                ema_checkpoint = checkpoint.copy()
                ema_checkpoint['model_state_dict'] = self.ema.shadow
                torch.save(ema_checkpoint, ema_path)
                logger.info(f"Saved best EMA model: {ema_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """チェックポイント読み込み"""
        checkpoint = torch.load(checkpoint_path)

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        if 'ema_state_dict' in checkpoint and self.ema is not None:
            self.ema.shadow = checkpoint['ema_state_dict']

        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def fit(self, max_epochs: int = 100, early_stopping_patience: int = 10):
        """トレーニング実行"""
        logger.info("Starting training...")

        for epoch in range(self.current_epoch, max_epochs):
            self.current_epoch = epoch

            # トレーニング
            train_metrics = self.train_epoch()
            logger.info(f"Epoch {epoch}: {train_metrics}")

            # 検証
            val_metrics = self.validate()
            logger.info(f"Validation {epoch}: {val_metrics}")

            # 最良モデル判定
            current_val_loss = val_metrics['loss']
            is_best = current_val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = current_val_loss

            # チェックポイント保存
            self.save_checkpoint(val_metrics, is_best)

            # 早期停止判定
            if epoch - (self.best_val_loss_epoch if hasattr(self, 'best_val_loss_epoch') else 0) > early_stopping_patience:
                logger.info("Early stopping triggered")
                break

        logger.info("Training completed")


# 後方互換性のためのエイリアス
RobustTrainingPipeline = RobustTrainer
