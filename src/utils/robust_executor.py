"""
Robust Executor for ATFT-GAT-FAN
本番環境向けエラーハンドリングと自動回復システム
"""

import os
import sys
import signal
import logging
import traceback
from typing import Any, Callable, Optional, Dict, List
from pathlib import Path
from contextlib import contextmanager
from functools import wraps
import time
import threading
import json

import torch
import torch.nn as nn
from torch.cuda import OutOfMemoryError

from .settings import get_settings

logger = logging.getLogger(__name__)
config = get_settings()


class CheckpointManager:
    """チェックポイント管理クラス"""

    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.emergency_file = self.checkpoint_dir / "emergency_checkpoint.pth"

    def save_emergency_checkpoint(self, model: Optional[nn.Module] = None,
                                optimizer: Optional[torch.optim.Optimizer] = None,
                                scaler: Optional[torch.cuda.amp.GradScaler] = None,
                                extra_data: Optional[Dict[str, Any]] = None):
        """緊急チェックポイント保存"""
        try:
            checkpoint = {
                'timestamp': time.time(),
                'emergency': True,
            }

            if model is not None:
                checkpoint['model_state_dict'] = model.state_dict()

            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()

            if scaler is not None:
                checkpoint['scaler_state_dict'] = scaler.state_dict()

            if extra_data:
                checkpoint.update(extra_data)

            torch.save(checkpoint, self.emergency_file)
            logger.info(f"Emergency checkpoint saved: {self.emergency_file}")

            # 通知（オプション）
            self._notify_emergency_save()

        except Exception as e:
            logger.error(f"Failed to save emergency checkpoint: {e}")

    def load_emergency_checkpoint(self) -> Optional[Dict[str, Any]]:
        """緊急チェックポイント読み込み"""
        if not self.emergency_file.exists():
            return None

        try:
            checkpoint = torch.load(self.emergency_file)
            logger.info(f"Emergency checkpoint loaded: {self.emergency_file}")
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load emergency checkpoint: {e}")
            return None

    def _notify_emergency_save(self):
        """緊急保存通知（Slack等への通知用フック）"""
        # ここに通知ロジックを実装
        # 例: Slack通知、メール送信など
        pass


class SignalHandler:
    """シグナルハンドラー"""

    def __init__(self, checkpoint_manager: CheckpointManager):
        self.checkpoint_manager = checkpoint_manager
        self.shutdown_event = threading.Event()
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """シグナルハンドラーの設定"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Windows以外の場合
        if hasattr(signal, 'SIGUSR1'):
            signal.signal(signal.SIGUSR1, self._checkpoint_handler)

    def _signal_handler(self, signum, frame):
        """メインシグナルハンドラー"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")

        # 緊急チェックポイント保存
        self.checkpoint_manager.save_emergency_checkpoint()

        # シャットダウンイベント設定
        self.shutdown_event.set()

        logger.info("Graceful shutdown initiated")

    def _checkpoint_handler(self, signum, frame):
        """チェックポイント保存シグナルハンドラー"""
        logger.info("Received checkpoint signal, saving checkpoint...")
        self.checkpoint_manager.save_emergency_checkpoint()

    def should_shutdown(self) -> bool:
        """シャットダウンが必要かチェック"""
        return self.shutdown_event.is_set()

    def wait_for_shutdown(self, timeout: Optional[float] = None):
        """シャットダウン待機"""
        self.shutdown_event.wait(timeout)


class RobustExecutor:
    """
    堅牢な実行クラス
    自動回復・エラーハンドリング・シグナル処理
    """

    def __init__(self, config: Any, checkpoint_manager: Optional[CheckpointManager] = None):
        self.config = config
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.signal_handler = SignalHandler(self.checkpoint_manager)

        # 実行統計
        self.execution_stats = {
            'start_time': None,
            'end_time': None,
            'attempts': 0,
            'successes': 0,
            'failures': 0,
            'oom_errors': 0,
            'other_errors': 0,
            'recovery_attempts': 0
        }

        logger.info("RobustExecutor initialized")

    @contextmanager
    def graceful_execution(self):
        """優雅な実行コンテキスト"""
        self.execution_stats['start_time'] = time.time()
        self.execution_stats['attempts'] += 1

        try:
            yield

            # 成功
            self.execution_stats['successes'] += 1
            logger.info("Execution completed successfully")

        except OutOfMemoryError as e:
            # OOMエラー
            self.execution_stats['oom_errors'] += 1
            logger.error(f"OOM error occurred: {e}")

            # メモリ解放
            self._handle_oom_error()

            # 再スロー
            raise

        except Exception as e:
            # その他のエラー
            self.execution_stats['failures'] += 1
            self.execution_stats['other_errors'] += 1
            logger.error(f"Execution failed: {e}")
            logger.error(traceback.format_exc())

            # 緊急チェックポイント保存
            self.checkpoint_manager.save_emergency_checkpoint()

            # 再スロー
            raise

        finally:
            self.execution_stats['end_time'] = time.time()

    def auto_recover(self, func: Callable, max_retries: int = 3, backoff_factor: float = 2.0) -> Callable:
        """自動回復デコレータ"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    # シャットダウンチェック
                    if self.signal_handler.should_shutdown():
                        logger.info("Shutdown requested, aborting execution")
                        return None

                    return func(*args, **kwargs)

                except OutOfMemoryError as e:
                    last_exception = e
                    self.execution_stats['recovery_attempts'] += 1

                    if attempt < max_retries:
                        logger.warning(f"OOM on attempt {attempt + 1}/{max_retries + 1}, attempting recovery...")

                        # OOM回復処理
                        self._handle_oom_error()

                        # 指数バックオフ
                        sleep_time = backoff_factor ** attempt
                        logger.info(f"Waiting {sleep_time:.1f} seconds before retry...")
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"OOM recovery failed after {max_retries + 1} attempts")
                        break

                except Exception as e:
                    last_exception = e

                    if attempt < max_retries:
                        logger.warning(f"Error on attempt {attempt + 1}/{max_retries + 1}: {e}")
                        # 指数バックオフ
                        sleep_time = backoff_factor ** attempt
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"All recovery attempts failed")
                        break

            # 最終的に失敗した場合
            if last_exception:
                raise last_exception

        return wrapper

    def _handle_oom_error(self):
        """OOMエラー処理"""
        try:
            # GPUメモリ解放
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Python GC実行
            import gc
            gc.collect()

            logger.info("OOM recovery: memory cleared")

        except Exception as e:
            logger.warning(f"Failed to clear memory: {e}")

    def adaptive_batch_size(self, current_batch_size: int, reduction_factor: float = 0.5) -> int:
        """適応バッチサイズ調整"""
        new_batch_size = max(1, int(current_batch_size * reduction_factor))

        if new_batch_size != current_batch_size:
            logger.info(f"Reduced batch size from {current_batch_size} to {new_batch_size}")

        return new_batch_size

    def get_execution_stats(self) -> Dict[str, Any]:
        """実行統計取得"""
        stats = self.execution_stats.copy()

        if stats['start_time'] and stats['end_time']:
            stats['total_duration'] = stats['end_time'] - stats['start_time']
            stats['success_rate'] = stats['successes'] / stats['attempts'] if stats['attempts'] > 0 else 0
            stats['failure_rate'] = stats['failures'] / stats['attempts'] if stats['attempts'] > 0 else 0

        return stats

    def save_execution_report(self, output_path: str):
        """実行レポート保存"""
        try:
            report = {
                'timestamp': time.time(),
                'execution_stats': self.get_execution_stats(),
                'config_summary': self._summarize_config(),
                'system_info': self._get_system_info()
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str, ensure_ascii=False)

            logger.info(f"Execution report saved: {output_path}")

        except Exception as e:
            logger.error(f"Failed to save execution report: {e}")

    def _summarize_config(self) -> Dict[str, Any]:
        """設定のサマリー作成"""
        if hasattr(self.config, '__dict__'):
            return {k: v for k, v in self.config.__dict__.items()
                   if not k.startswith('_') and not callable(v)}
        elif hasattr(self.config, 'dict'):
            return self.config.dict()
        else:
            return {}

    def _get_system_info(self) -> Dict[str, Any]:
        """システム情報取得"""
        import platform

        info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['gpu_count'] = torch.cuda.device_count()
            info['current_device'] = torch.cuda.current_device()

        return info


def with_robust_execution(config: Any = None):
    """堅牢な実行デコレータ"""
    def decorator(func: Callable) -> Callable:
        executor = RobustExecutor(config)

        @wraps(func)
        def wrapper(*args, **kwargs):
            with executor.graceful_execution():
                return executor.auto_recover(func)(*args, **kwargs)

        return wrapper
    return decorator


# 後方互換性のためのエイリアス
RobustTrainingExecutor = RobustExecutor
