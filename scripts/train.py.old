#!/usr/bin/env python3
"""
ATFT-GAT-FAN 学習スクリプト
設計書v3.2に基づく高度な学習パイプライン
"""
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger, WandbLogger
import torch
import logging
from pathlib import Path

from src.atft_gat_fan.data.loaders.parquet_loader import ParquetDataModule
from src.atft_gat_fan.models.architectures.atft_gat_fan import ATFT_GAT_FAN
from src.atft_gat_fan.training.phases.phase_trainer import PhaseTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(config: DictConfig):
    """学習メイン関数"""

    logger.info(f"🚀 Starting ATFT-GAT-FAN training with config: {config.mode}")

    # 設定の検証
    logger.info("📋 Configuration validation:")
    logger.info(f"  Mode: {config.mode}")
    logger.info(f"  Model: {config.model.name}")
    logger.info(f"  Data: {config.data.name}")
    logger.info(f"  Train: {config.train.name}")

    # デバッグモードの場合
    if config.debug.enabled:
        logger.info("🐛 Debug mode enabled")
        config.trainer.fast_dev_run = config.debug.fast_dev_run

    # データモジュールの準備
    logger.info("📊 Preparing data module...")
    data_module = ParquetDataModule(config)

    # Phase Trainingの実行
    if config.train.phase_training.enabled:
        logger.info("🎯 Phase training enabled")
        phase_trainer = PhaseTrainer(config)
        final_model = phase_trainer.train_all_phases(data_module)
    else:
        # 通常の学習
        logger.info("🔥 Single phase training")
        final_model = train_single_phase(config, data_module)

    # 最終モデルの保存
    save_final_model(final_model, config)

    logger.info("✅ Training completed successfully!")
    return final_model


def train_single_phase(config: DictConfig, data_module):
    """単一フェーズの学習"""

    # モデル初期化
    model = ATFT_GAT_FAN(config)

    # コールバック設定
    callbacks = setup_callbacks(config)

    # ロガー設定
    loggers = setup_loggers(config)

    # トレーナー設定
    trainer_config = OmegaConf.to_container(config.trainer, resolve=True)
    trainer = pl.Trainer(
        **trainer_config,
        callbacks=callbacks,
        logger=loggers,
        enable_progress_bar=True,
    )

    # 学習実行
    logger.info("🚀 Starting training...")
    trainer.fit(model, data_module)

    # テスト実行
    logger.info("🧪 Running evaluation...")
    trainer.test(model, data_module)

    return model


def setup_callbacks(config: DictConfig):
    """コールバック設定"""
    callbacks = []

    # Model Checkpoint
    checkpoint_config = OmegaConf.to_container(config.train.checkpoint, resolve=True)
    checkpoint_callback = ModelCheckpoint(**checkpoint_config)
    callbacks.append(checkpoint_callback)

    # Early Stopping
    if config.train.early_stopping:
        early_stopping_config = OmegaConf.to_container(config.train.early_stopping, resolve=True)
        early_stopping = EarlyStopping(**early_stopping_config)
        callbacks.append(early_stopping)

    return callbacks


def setup_loggers(config: DictConfig):
    """ロガー設定"""
    loggers = []

    # MLFlow Logger
    if config.logging.mlflow:
        mlflow_config = OmegaConf.to_container(config.logging.mlflow, resolve=True)
        mlflow_logger = MLFlowLogger(**mlflow_config)
        loggers.append(mlflow_logger)

    # Wandb Logger
    if config.logging.wandb.enabled:
        wandb_config = OmegaConf.to_container(config.logging.wandb, resolve=True)
        wandb_logger = WandbLogger(**wandb_config)
        loggers.append(wandb_logger)

    return loggers


def save_final_model(model, config: DictConfig):
    """最終モデルの保存"""
    save_path = Path(config.paths.models) / f"{config.model.name}_final.ckpt"

    # PyTorch Lightningのモデル保存
    trainer = pl.Trainer()
    trainer.save_checkpoint(save_path)

    logger.info(f"💾 Model saved to: {save_path}")

    # 設定も保存
    config_path = save_path.parent / f"{config.model.name}_config.yaml"
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)

    logger.info(f"📋 Config saved to: {config_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main():
    """メイン実行関数"""
    train()


if __name__ == "__main__":
    main()
