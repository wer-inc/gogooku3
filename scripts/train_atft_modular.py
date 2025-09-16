#!/usr/bin/env python3
"""
Modular ATFT Training Script
モジュール化されたATFT学習スクリプト - 7000行から大幅削減

PDFで提案された改善: train_atft.pyを機能別モジュールに分割
"""

import argparse
import logging
import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

# Add paths
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

# Import modular components
from gogooku3.training.atft.environment import ATFTEnvironment
from gogooku3.training.atft.data_module import ProductionDataModuleV2
from gogooku3.training.atft.trainer import ATFTTrainer

# Import model (assuming it exists)
try:
    from gogooku3.models.atft_gat_fan import ATFT_GAT_FAN
except ImportError:
    # Fallback import
    from src.models.architectures.atft_gat_fan import ATFT_GAT_FAN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs/atft", config_name="unified_config")
def train(cfg: DictConfig) -> None:
    """
    Main training function using modular components.

    Args:
        cfg: Hydra configuration
    """
    logger.info("=" * 80)
    logger.info("🚀 ATFT Modular Training Started")
    logger.info("=" * 80)

    # Step 1: Environment setup
    logger.info("🔧 Setting up environment...")
    env = ATFTEnvironment(profile="production")
    env.setup()

    # Apply environment overrides to config
    cfg = env.apply_env_overrides(cfg)

    # Fix random seed
    env.fix_seed(seed=cfg.experiment.seed, deterministic=False)

    # Get device
    device = env.get_device()

    # Log hardware info
    hardware_info = env.get_hardware_info()
    logger.info(f"📊 Hardware: {hardware_info}")

    # Step 2: Data preparation
    logger.info("📂 Preparing data...")
    data_module = ProductionDataModuleV2(cfg)
    data_module.setup()

    # Get data info
    data_info = data_module.get_data_info()
    logger.info(f"📊 Data info: {data_info}")

    # Step 3: Model initialization
    logger.info("🏗️ Building model...")
    model = build_model(cfg, data_info)
    model = model.to(device)

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Step 4: Trainer setup
    logger.info("🎯 Setting up trainer...")
    output_dir = Path(cfg.experiment.output_dir) / cfg.experiment.name
    trainer = ATFTTrainer(
        model=model,
        config=cfg,
        device=device,
        output_dir=output_dir,
    )

    # Step 5: Training
    logger.info("🚄 Starting training...")
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=cfg.train.trainer.max_epochs,
    )

    # Step 6: Final evaluation
    logger.info("📈 Training completed!")
    logger.info(f"✅ Best validation metric: {results['best_val_metric']:.4f}")
    logger.info(f"📁 Best checkpoint: {results['best_checkpoint']}")

    # Step 7: Cleanup
    env.restore()

    return results


def build_model(cfg: DictConfig, data_info: dict) -> torch.nn.Module:
    """
    Build the ATFT-GAT-FAN model.

    Args:
        cfg: Model configuration
        data_info: Data information

    Returns:
        Initialized model
    """
    model_config = {
        "input_dim": data_info["num_features"],
        "hidden_dim": cfg.model.hidden_dim,
        "num_heads": cfg.model.num_heads,
        "num_layers": cfg.model.num_layers,
        "dropout": cfg.model.dropout,
        "sequence_length": data_info["sequence_length"],
        "prediction_horizons": cfg.data.time_series.prediction_horizons,
    }

    # Add graph configuration if enabled
    if cfg.model.graph.enable:
        model_config.update({
            "use_graph": True,
            "num_nodes": cfg.model.graph.num_nodes,
            "edge_threshold": cfg.model.graph.edge_threshold,
            "graph_alpha": cfg.model.graph.alpha_init,
        })

    # Add FAN configuration if enabled
    if cfg.model.fan.enable:
        model_config.update({
            "use_fan": True,
            "num_frequencies": cfg.model.fan.num_frequencies,
            "temperature": cfg.model.fan.temperature,
        })

    # Create model
    model = ATFT_GAT_FAN(**model_config)

    # Apply stability settings
    if cfg.train.stability.use_layerscale:
        apply_layerscale(model, init_value=cfg.train.stability.layerscale_init)

    return model


def apply_layerscale(model: torch.nn.Module, init_value: float = 0.1) -> None:
    """Apply LayerScale to model for training stability."""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            # Add LayerScale after attention
            if hasattr(module, "out_proj"):
                original_weight = module.out_proj.weight.data.clone()
                with torch.no_grad():
                    module.out_proj.weight.data = original_weight * init_value

        elif isinstance(module, torch.nn.Linear) and "mlp" in name:
            # Add LayerScale after MLP
            with torch.no_grad():
                module.weight.data *= init_value


def main():
    """Main entry point for direct script execution."""
    # Parse additional arguments if needed
    parser = argparse.ArgumentParser(description="Modular ATFT Training")
    parser.add_argument(
        "--config-path",
        type=str,
        default="../configs/atft",
        help="Path to config directory",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="unified_config",
        help="Config file name",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in Hydra format",
    )

    args = parser.parse_args()

    # Initialize Hydra and run training
    hydra.initialize(version_base=None, config_path=args.config_path)
    cfg = hydra.compose(config_name=args.config_name, overrides=args.overrides)

    # Run training
    results = train(cfg)

    # Exit with appropriate code
    sys.exit(0 if results.get("best_val_metric", 0) > 0 else 1)


if __name__ == "__main__":
    # Check if running with Hydra or standalone
    if "--help" in sys.argv or "--hydra-help" in sys.argv:
        # Let Hydra handle help
        train()
    else:
        # Use Hydra decorator
        train()