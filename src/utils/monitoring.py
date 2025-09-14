"""
Monitoring System for ATFT-GAT-FAN
W&B + TensorBoard統合監視システム
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

logger = logging.getLogger(__name__)


class MetricsCollector:
    """メトリクス収集・集約クラス"""

    def __init__(self, buffer_size: int = 100):
        self.buffer_size = buffer_size
        self.metrics_buffer = []
        self.step_buffer = []

    def add_metrics(self, metrics: dict[str, Any], step: int):
        """メトリクスをバッファに追加"""
        self.metrics_buffer.append(metrics)
        self.step_buffer.append(step)

        # バッファサイズ超過時は古いものを削除
        if len(self.metrics_buffer) > self.buffer_size:
            self.metrics_buffer.pop(0)
            self.step_buffer.pop(0)

    def get_batch_metrics(self) -> list[dict[str, Any]]:
        """バッファされたメトリクスを取得"""
        return self.metrics_buffer.copy()

    def clear_buffer(self):
        """バッファをクリア"""
        self.metrics_buffer = []
        self.step_buffer = []

    def get_latest_metrics(self) -> dict[str, Any] | None:
        """最新のメトリクスを取得"""
        return self.metrics_buffer[-1] if self.metrics_buffer else None


class ComprehensiveLogger:
    """統合ロギングシステム（W&B + TensorBoard）"""

    def __init__(
        self,
        config: Any,
        experiment_name: str,
        log_dir: str = "./logs",
        use_wandb: bool = True,
        use_tensorboard: bool = True
    ):
        """
        Args:
            config: 設定オブジェクト
            experiment_name: 実験名
            log_dir: ログディレクトリ
            use_wandb: W&Bを使用
            use_tensorboard: TensorBoardを使用
        """
        self.config = config
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.use_tensorboard = use_tensorboard

        # ログディレクトリ作成
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        if self.use_tensorboard:
            self.tb_writer = SummaryWriter(str(self.log_dir))
            logger.info(f"TensorBoard logging to: {self.log_dir}")

        # W&B
        if self.use_wandb:
            try:
                self.wandb_run = wandb.init(
                    project=getattr(config, 'wandb_project', 'atft-gat-fan'),
                    name=experiment_name,
                    config=self._config_to_dict(config),
                    tags=["atft-gat-fan", "production"],
                    resume="allow"
                )
                logger.info(f"W&B logging enabled: {wandb.run.name}")
            except Exception as e:
                logger.warning(f"W&B initialization failed: {e}")
                self.use_wandb = False
        else:
            self.wandb_run = None
            if use_wandb and not WANDB_AVAILABLE:
                logger.warning("W&B requested but not available")

        # メトリクス収集器
        self.metrics_collector = MetricsCollector()

        # プロファイラー
        self.profiler_enabled = getattr(config, 'enable_profiler', False)
        if self.profiler_enabled:
            self._setup_profiler()

    def _config_to_dict(self, config) -> dict[str, Any]:
        """設定オブジェクトを辞書に変換"""
        if hasattr(config, '__dict__'):
            return config.__dict__
        elif hasattr(config, 'dict'):
            return config.dict()
        else:
            return dict(config) if hasattr(config, 'items') else {}

    def _setup_profiler(self):
        """プロファイラー設定"""
        try:
            from torch.profiler import (
                ProfilerActivity,
                profile,
                tensorboard_trace_handler,
            )

            self.profiler = profile(
                activities=[
                    ProfilerActivity.CPU,
                    ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=tensorboard_trace_handler(str(self.log_dir)),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
            logger.info("Profiler enabled")
        except Exception as e:
            logger.warning(f"Profiler setup failed: {e}")
            self.profiler_enabled = False

    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int,
        prefix: str = ""
    ):
        """メトリクスをログ"""
        # プレフィックス付与
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        # バッファに追加
        self.metrics_collector.add_metrics(metrics, step)

        # TensorBoard
        if self.use_tensorboard:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, step)

        # W&B
        if self.use_wandb:
            try:
                wandb.log(metrics, step=step)
            except Exception as e:
                logger.debug(f"W&B logging failed: {e}")

    def log_hyperparameters(self, params: dict[str, Any]):
        """ハイパーパラメータをログ"""
        # TensorBoard
        if self.use_tensorboard:
            self.tb_writer.add_hparams(params, {})

        # W&B
        if self.use_wandb:
            try:
                wandb.config.update(params)
            except Exception as e:
                logger.debug(f"W&B config update failed: {e}")

    def log_model_graph(self, model: torch.nn.Module, input_tensor: torch.Tensor):
        """モデルグラフをログ"""
        if self.use_tensorboard:
            try:
                self.tb_writer.add_graph(model, input_tensor)
            except Exception as e:
                logger.debug(f"Model graph logging failed: {e}")

    def log_model_stats(self, model: torch.nn.Module, step: int):
        """モデル統計をログ"""
        if not self.use_tensorboard:
            return

        try:
            # 重みの分布
            for name, param in model.named_parameters():
                if param.requires_grad and param.numel() > 0:
                    param_data = param.data.flatten().cpu().numpy()
                    self.tb_writer.add_histogram(
                        f"weights/{name}", param_data, step, bins='auto'
                    )

                    # 勾配が存在する場合
                    if param.grad is not None:
                        grad_data = param.grad.data.flatten().cpu().numpy()
                        self.tb_writer.add_histogram(
                            f"gradients/{name}", grad_data, step, bins='auto'
                        )

            # 活性化フック（簡易版）
            self._log_activations(model, step)

        except Exception as e:
            logger.debug(f"Model stats logging failed: {e}")

    def _log_activations(self, model: torch.nn.Module, step: int):
        """活性化をログ（一時的なフック）"""
        activations = {}

        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor) and output.numel() > 0:
                module_name = module.__class__.__name__
                if module_name not in activations:
                    activations[module_name] = output.flatten().cpu().numpy()

        # フック登録
        hooks = []
        for module in model.modules():
            if len(list(module.children())) == 0:  # リーフモジュール
                hooks.append(module.register_forward_hook(hook_fn))

        # ダミー入力で活性化を取得
        try:
            with torch.no_grad():
                dummy_input = torch.randn(1, 60, model.hidden_size or 256)
                if hasattr(model, 'device'):
                    dummy_input = dummy_input.to(model.device)
                model(dummy_input)
        except:
            pass

        # フック解除
        for hook in hooks:
            hook.remove()

        # ログ出力
        for module_name, activation_data in activations.items():
            if len(activation_data) > 1000:  # 大きすぎる場合はサンプリング
                activation_data = np.random.choice(activation_data, 1000, replace=False)
            self.tb_writer.add_histogram(
                f"activations/{module_name}", activation_data, step, bins='auto'
            )

    def log_batch_metrics(self, step: int):
        """バッファされたメトリクスをバッチログ"""
        batch_metrics = self.metrics_collector.get_batch_metrics()

        if not batch_metrics:
            return

        # W&Bにバッチログ
        if self.use_wandb:
            try:
                wandb.log({"batch_metrics": batch_metrics})
            except Exception as e:
                logger.debug(f"W&B batch logging failed: {e}")

        # バッファクリア
        self.metrics_collector.clear_buffer()

    def start_profiling(self):
        """プロファイリング開始"""
        if self.profiler_enabled and hasattr(self, 'profiler'):
            self.profiler.start()

    def stop_profiling(self):
        """プロファイリング停止"""
        if self.profiler_enabled and hasattr(self, 'profiler'):
            self.profiler.stop()

    def log_system_stats(self, step: int):
        """システム統計をログ"""
        try:
            import GPUtil
            import psutil

            # CPU使用率
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent

            system_metrics = {
                'system/cpu_percent': cpu_percent,
                'system/memory_percent': memory_percent,
            }

            # GPU統計
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    system_metrics.update({
                        'system/gpu_utilization': gpu.load * 100,
                        'system/gpu_memory_used': gpu.memoryUsed,
                        'system/gpu_memory_total': gpu.memoryTotal,
                        'system/gpu_temperature': gpu.temperature,
                    })
            except:
                pass

            self.log_metrics(system_metrics, step)

        except ImportError:
            logger.debug("psutil or GPUtil not available for system stats")
        except Exception as e:
            logger.debug(f"System stats logging failed: {e}")

    def finish(self):
        """ロギング終了処理"""
        if self.use_tensorboard:
            self.tb_writer.close()

        if self.use_wandb:
            try:
                wandb.finish()
            except:
                pass

        logger.info("Logging finished")


class TrainingMonitor:
    """トレーニング監視クラス"""

    def __init__(self, logger: ComprehensiveLogger, config: Any):
        self.logger = logger
        self.config = config
        self.start_time = datetime.now()

        # 監視対象メトリクス
        self.best_metrics = {
            'val_loss': float('inf'),
            'val_rankic_h1': -float('inf'),
            'train_loss': float('inf')
        }

        # 履歴
        self.metric_history = {
            'val_loss': [],
            'val_rankic_h1': [],
            'pred_std_ratio': [],
            'grad_norm': []
        }

    def update_best_metrics(self, metrics: dict[str, float], step: int):
        """最良メトリクス更新"""
        updated = False

        for key, value in metrics.items():
            if key in self.best_metrics:
                if 'loss' in key and value < self.best_metrics[key]:
                    self.best_metrics[key] = value
                    updated = True
                elif 'rankic' in key and value > self.best_metrics[key]:
                    self.best_metrics[key] = value
                    updated = True

            # 履歴更新
            if key in self.metric_history:
                self.metric_history[key].append((step, value))

        if updated:
            self.logger.log_metrics(
                {f"best_{k}": v for k, v in self.best_metrics.items()},
                step,
                prefix="best"
            )

    def check_training_stability(self, metrics: dict[str, float], step: int) -> dict[str, bool]:
        """トレーニング安定性をチェック"""
        warnings = {}

        # 損失がNaN/Infでないか
        if not np.isfinite(metrics.get('loss', 0)):
            warnings['loss_invalid'] = True

        # 勾配ノルムが異常でないか
        grad_norm = metrics.get('grad_norm', 0)
        if grad_norm > 10.0:
            warnings['grad_norm_high'] = True
        elif grad_norm < 1e-6:
            warnings['grad_norm_low'] = True

        # Pred.std / Target.std の適正性
        pred_std = metrics.get('pred_std_h1', 1.0)
        target_std = metrics.get('target_std_h1', 1.0)
        if target_std > 0:
            std_ratio = pred_std / target_std
            if std_ratio < 0.3 or std_ratio > 1.5:
                warnings['std_ratio_invalid'] = True

        # 警告をログ
        if warnings:
            warning_metrics = {f"warning_{k}": 1 for k in warnings.keys()}
            self.logger.log_metrics(warning_metrics, step, prefix="stability")

        return warnings

    def log_training_summary(self, final_metrics: dict[str, float]):
        """トレーニングサマリーをログ"""
        summary = {
            'training_duration_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
            'final_val_loss': final_metrics.get('val_loss', 0),
            'final_val_rankic_h1': final_metrics.get('val_rankic_h1', 0),
            'best_val_loss': self.best_metrics['val_loss'],
            'best_val_rankic_h1': self.best_metrics['val_rankic_h1'],
            'improvement_val_loss': self.best_metrics['train_loss'] - final_metrics.get('train_loss', 0),
        }

        self.logger.log_metrics(summary, 0, prefix="summary")
        logger.info(f"Training Summary: {summary}")


# 後方互換性のためのエイリアス
Logger = ComprehensiveLogger
Monitor = TrainingMonitor
