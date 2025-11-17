"""Training entrypoint for ATFT-GAT-FAN within gogooku5.

This script is intentionally self-contained so that the
``gogooku5/models/atft_gat_fan`` package can run without importing
training utilities from gogooku3. It provides a minimal but functional
training loop using PyTorch Lightning and the shared ML dataset produced
by ``gogooku5/data``.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from atft_gat_fan.config.training_config import TrainingConfig
from atft_gat_fan.training.data import build_dataloaders, load_dataset
from atft_gat_fan.training.module import ATFTGATFANLightningModule

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
_logger = logging.getLogger("atft_gat_fan.train")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training script."""

    parser = argparse.ArgumentParser(description="Train the ATFT-GAT-FAN model")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the training configuration file.",
    )
    parser.add_argument(
        "--dataset",
        default="../../data/output/ml_dataset_latest.parquet",
        help="Path to the pre-built dataset parquet file.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override the maximum number of epochs.",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point used by the Makefile and CLI."""

    args = parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        _logger.error("Config file not found: %s", config_path)
        return 1

    cfg = TrainingConfig.from_yaml(config_path)

    # Override dataset path if explicitly provided.
    if args.dataset:
        cfg.dataset.path = Path(args.dataset).resolve()

    if args.max_epochs is not None:
        cfg.train.max_epochs = int(args.max_epochs)

    _logger.info("Using dataset: %s", cfg.dataset.path)
    df, spec = load_dataset(cfg.dataset)
    _logger.info(
        "Loaded dataset: %d rows, %d feature columns, %d targets",
        len(df),
        len(spec.feature_columns),
        len(spec.target_columns),
    )

    train_loader, val_loader = build_dataloaders(df, spec, cfg.dataset, cfg.train)
    in_features = len(spec.feature_columns)

    model = ATFTGATFANLightningModule(
        in_features=in_features,
        n_targets=len(spec.target_columns),
        train_cfg=cfg.train,
        amp_cfg=cfg.amp,
    )

    precision = "bf16-mixed" if cfg.amp.enabled and cfg.amp.dtype == "bfloat16" else "16-mixed"
    callbacks = []
    if cfg.logging.enable_checkpointing:
        cfg.logging.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_cb = ModelCheckpoint(
            dirpath=str(cfg.logging.checkpoint_dir),
            filename=cfg.logging.checkpoint_name,
            save_top_k=1,
            monitor="val_loss",
            mode="min",
        )
        callbacks.append(ckpt_cb)

    accelerator = "gpu" if pl.utilities.device_parser.num_cuda_devices() > 0 else "cpu"

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=accelerator,
        precision=precision if cfg.amp.enabled else "32-true",
        log_every_n_steps=cfg.logging.log_every_n_steps,
        callbacks=callbacks,
    )

    _logger.info(
        "Starting training: epochs=%d, batch_size=%d, params=%d",
        cfg.train.max_epochs,
        cfg.train.batch_size,
        model.parameter_count,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    _logger.info("Training finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
