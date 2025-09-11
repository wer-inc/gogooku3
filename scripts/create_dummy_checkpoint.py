#!/usr/bin/env python3
"""
Create a valid dummy checkpoint for evaluation of ATFT-GAT-FAN.

This script instantiates the actual model class with a minimal config
compatible with src/models/architectures/atft_gat_fan.py, then saves
an untrained state_dict checkpoint to output/checkpoints/dummy_model.pth
"""

from __future__ import annotations

import sys
from pathlib import Path
import logging

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("create_dummy_checkpoint")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def build_minimal_model_config() -> "OmegaConf":
    """Build a minimal DictConfig the model can consume.

    The evaluation script loads configs/model/atft_gat_fan_v1.yaml which lacks
    the "data" section the class expects. Here we create a compact config with
    the same structure as the runtime model, including data.features.input_dim
    and data.time_series.prediction_horizons, so the instantiated model's
    state_dict keys will match at evaluation time.
    """
    # Base model config (lightweight)
    cfg = OmegaConf.create(
        {
            "project": {"name": "ATFT-GAT-FAN"},
            "data": {
                "features": {"input_dim": 64},
                "time_series": {"prediction_horizons": [1, 2, 3, 5, 10]},
            },
            "model": {
                "hidden_size": 64,
                "input_projection": {"use_layer_norm": True, "dropout": 0.1},
                "adaptive_normalization": {
                    "fan": {
                        "enabled": False,
                        "window_sizes": [5, 10, 20],
                        "aggregation": "weighted_mean",
                        "learn_weights": True,
                    },
                    "san": {
                        "enabled": False,
                        "num_slices": 1,
                        "overlap": 0.0,
                        "slice_aggregation": "mean",
                    },
                },
                "tft": {
                    "variable_selection": {
                        "dropout": 0.1,
                        "use_sigmoid": True,
                        "sparsity_coefficient": 0.0,
                    },
                    "attention": {"heads": 2},
                    "lstm": {"layers": 1, "dropout": 0.1},
                    "temporal": {"use_positional_encoding": True, "max_sequence_length": 20},
                },
                "gat": {
                    "enabled": False,
                    "architecture": {
                        "num_layers": 1,
                        "hidden_channels": [64],
                        "heads": [2],
                        "concat": [False],
                    },
                    "layer_config": {"dropout": 0.0, "edge_dropout": 0.0},
                    "edge_features": {"use_edge_attr": False, "edge_dim": None},
                    "regularization": {"edge_weight_penalty": 0.0, "attention_entropy_penalty": 0.0},
                },
                "prediction_head": {
                    "architecture": {"hidden_layers": [], "dropout": 0.0},
                    "output": {
                        "point_prediction": True,
                        "quantile_prediction": {"enabled": False, "quantiles": [0.1, 0.5, 0.9]},
                    },
                },
            },
        }
    )
    return cfg


def main() -> int:
    logger.info("Creating dummy checkpoint for evaluation...")

    # Lazy import after sys.path patch
    from src.models.architectures.atft_gat_fan import ATFT_GAT_FAN

    cfg = build_minimal_model_config()
    model = ATFT_GAT_FAN(cfg)

    ckpt = {
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "train_loss": 0.0,
        "val_loss": 0.0,
    }

    out_dir = ROOT / "output" / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "dummy_model.pth"
    torch.save(ckpt, ckpt_path)
    logger.info(f"✅ Dummy checkpoint saved: {ckpt_path}")
    logger.info(f"Size: {ckpt_path.stat().st_size / 1024:.2f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

