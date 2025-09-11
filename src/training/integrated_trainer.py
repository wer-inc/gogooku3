"""
Integrated Trainer for ATFT-GAT-FAN
すべての改善を統合したトレーニングスクリプト
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# インポート
from ..utils.settings import get_settings, set_reproducibility
from ..utils.monitoring import ComprehensiveLogger, TrainingMonitor
from ..utils.robust_executor import RobustExecutor, CheckpointManager
from .robust_trainer import RobustTrainer
from ..data.loaders.streaming_dataset import StreamingParquetDataset, OptimizedDataLoader
from ..losses.multi_horizon_loss import ComprehensiveLoss
from ..atft_gat_fan.models.architectures.atft_gat_fan import ATFT_GAT_FAN

config = get_settings()


class IntegratedTrainer:
    """統合トレーニングクラス"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # 再現性設定
        set_reproducibility(cfg.seed, cfg.deterministic)

        # チェックポイントマネージャー
        self.checkpoint_manager = CheckpointManager(cfg.checkpoint_dir)

        # 堅牢実行クラス
        self.executor = RobustExecutor(cfg, self.checkpoint_manager)

        # ロガー設定
        self.logger = self._setup_logger()

        # モニター
        self.monitor = TrainingMonitor(self.logger, cfg)

        # データセットとローダー
        self.train_loader, self.val_loader = self._setup_data()

        # モデル
        self.model = self._setup_model()

        # トレーニングクラス
        self.trainer = RobustTrainer(
            self.model,
            self.train_loader,
            self.val_loader,
            cfg,
            cfg.checkpoint_dir
        )

        logger.info("IntegratedTrainer initialized")

    def _setup_logger(self) -> ComprehensiveLogger:
        """ロガー設定"""
        experiment_name = f"{self.cfg.project.name}_{self.cfg.mode}_{self.cfg.seed}"
        return ComprehensiveLogger(
            config=self.cfg,
            experiment_name=experiment_name,
            use_wandb=self.cfg.enable_wandb,
            use_tensorboard=self.cfg.enable_tensorboard
        )

    def _setup_data(self):
        """データセットとローダー設定"""
        # トレーニングデータ
        train_dataset = StreamingParquetDataset(
            parquet_files=self.cfg.data.train_files,
            feature_cols=self.cfg.data.feature_cols,
            target_cols=self.cfg.data.target_cols,
            sequence_length=self.cfg.data.sequence_length,
            prediction_horizons=self.cfg.data.prediction_horizons,
            batch_size=self.cfg.batch.train_batch_size,
            online_normalization=True,
            cache_stats=True
        )

        # 検証データ
        val_dataset = StreamingParquetDataset(
            parquet_files=self.cfg.data.val_files,
            feature_cols=self.cfg.data.feature_cols,
            target_cols=self.cfg.data.target_cols,
            sequence_length=self.cfg.data.sequence_length,
            prediction_horizons=self.cfg.data.prediction_horizons,
            batch_size=self.cfg.batch.val_batch_size,
            online_normalization=True,
            cache_stats=False  # 検証時はキャッシュ使用
        )

        # 最適化ローダー
        train_loader = OptimizedDataLoader(
            train_dataset,
            batch_size=self.cfg.batch.train_batch_size,
            num_workers=self.cfg.num_workers,
            prefetch_factor=self.cfg.prefetch_factor,
            pin_memory=self.cfg.pin_memory
        ).get_dataloader()

        val_loader = OptimizedDataLoader(
            val_dataset,
            batch_size=self.cfg.batch.val_batch_size,
            num_workers=self.cfg.num_workers,
            prefetch_factor=self.cfg.prefetch_factor,
            pin_memory=self.cfg.pin_memory
        ).get_dataloader()

        return train_loader, val_loader

    def _setup_model(self) -> ATFT_GAT_FAN:
        """モデル設定"""
        model = ATFT_GAT_FAN(self.cfg)

        # torch.compile適用（オプション）
        if self.cfg.compile_model:
            try:
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")

        return model

    def train(self):
        """トレーニング実行"""
        logger.info("Starting integrated training...")

        # ハイパーパラメータログ
        self.logger.log_hyperparameters(OmegaConf.to_container(self.cfg, resolve=True))

        # モデルグラフログ
        if hasattr(self.train_loader, 'dataset'):
            dummy_batch = next(iter(self.train_loader))
            if 'features' in dummy_batch:
                self.logger.log_model_graph(self.model, dummy_batch['features'])

        try:
            # トレーニング実行
            self.trainer.fit(
                max_epochs=self.cfg.max_epochs,
                early_stopping_patience=self.cfg.early_stopping_patience
            )

            # 最終評価
            final_metrics = self.trainer.validate()
            self.monitor.log_training_summary(final_metrics)

            logger.info("Training completed successfully")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

        finally:
            # 実行レポート保存
            execution_stats = self.executor.get_execution_stats()
            self.logger.log_metrics(execution_stats, 0, prefix="execution")

            report_path = Path(self.cfg.output_dir) / "execution_report.json"
            self.executor.save_execution_report(str(report_path))

            # ロガー終了
            self.logger.finish()

    def resume_training(self, checkpoint_path: str):
        """チェックポイントからの再開"""
        logger.info(f"Resuming training from {checkpoint_path}")

        # チェックポイント読み込み
        self.trainer.load_checkpoint(checkpoint_path)

        # トレーニング再開
        self.train()


@hydra.main(config_path="../../configs/atft", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """メイン関数"""
    # 設定マージ
    cfg = OmegaConf.merge(config.__dict__ if hasattr(config, '__dict__') else {}, cfg)

    # 出力ディレクトリ作成
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # 統合トレーナー初期化
    trainer = IntegratedTrainer(cfg)

    # トレーニング実行
    if cfg.resume_from:
        trainer.resume_training(cfg.resume_from)
    else:
        trainer.train()


if __name__ == "__main__":
    try:
        from omegaconf import DictConfig
        test_cfg = DictConfig({})
        main(test_cfg)
    except Exception as e:
        print(f"Training test failed: {e}")
        import traceback
        traceback.print_exc()
