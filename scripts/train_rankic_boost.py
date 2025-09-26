#!/usr/bin/env python
"""
RankIC Boost Training Script - Fundamental solution for maximum RankIC improvement

This script provides a clean, structured approach to training with RankIC optimization
by properly setting environment variables and using dedicated Hydra configuration files.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Execute RankIC-boosted training with proper configuration."""

    # Get project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Check if ATFT data exists
    atft_data_path = project_root / "output/atft_data"
    if not atft_data_path.exists():
        logger.error(f"❌ ATFT data not found at {atft_data_path}")
        logger.error("Please run data conversion first:")
        logger.error("  python scripts/integrated_ml_training_pipeline.py --only-convert")
        return 1

    # Check train/val/test splits
    required_dirs = ["train", "val", "test"]
    for dir_name in required_dirs:
        if not (atft_data_path / dir_name).exists():
            logger.error(f"❌ Missing {dir_name} directory in {atft_data_path}")
            return 1

    # Setup environment variables for RankIC optimization
    env = os.environ.copy()

    # Core optimization settings
    rankic_env = {
        # RankIC and financial metrics optimization
        "USE_RANKIC": "1",
        "RANKIC_WEIGHT": "0.5",      # Maximum RankIC weight
        "SHARPE_WEIGHT": "0.3",      # Strong Sharpe focus
        "CS_IC_WEIGHT": "0.2",       # Cross-sectional IC
        "USE_CS_IC": "1",

        # Additional loss components
        "USE_HUBER": "1",
        "HUBER_WEIGHT": "0.1",
        "USE_DIR_AUX": "0",          # Disable auxiliary tasks for focus
        "DIR_AUX_WEIGHT": "0.0",

        # Data augmentation and regularization
        "OUTPUT_NOISE_STD": "0.02",
        "FEATURE_CLIP_VALUE": "10.0",
        "ENABLE_FEATURE_NORM": "1",

        # DataLoader optimization
        "ALLOW_UNSAFE_DATALOADER": "1",
        "NUM_WORKERS": "8",
        "PERSISTENT_WORKERS": "1",
        "PREFETCH_FACTOR": "4",
        "PIN_MEMORY": "1",

        # GPU optimization
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_BACKENDS_CUDNN_BENCHMARK": "1",
        "TF32_ENABLED": "1",
        "CUDA_LAUNCH_BLOCKING": "0",

        # Model configuration
        "HIDDEN_SIZE": "256",
        "MODEL_HIDDEN_SIZE": "256",

        # Torch compile
        "ENABLE_TORCH_COMPILE": "1",
        "TORCH_COMPILE_MODE": "max-autotune",

        # Safety features
        "DEGENERACY_GUARD": "1",
        "DEGENERACY_ABORT": "0",
        "USE_SAFE_AMP": "1",

        # Phase-aware training (optional, can be disabled)
        "USE_PHASE_TRAINING": "0",  # Disable for now, let config control

        # Logging
        "LOG_LEVEL": "INFO",
    }

    env.update(rankic_env)

    # Build command for train_atft.py with our custom config
    cmd = [
        sys.executable,
        str(project_root / "scripts/train_atft.py"),
        "--config-path", "../configs/atft",
        "--config-name", "config_rankic_boost",  # Use our dedicated config
        "data.source.data_dir=output/atft_data",  # Explicitly override data path
    ]

    # Add any command line arguments passed to this script
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])

    # Log configuration
    logger.info("=" * 80)
    logger.info("🚀 RANKIC BOOST TRAINING - FUNDAMENTAL SOLUTION")
    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info(f"  Config: configs/atft/config_rankic_boost.yaml")
    logger.info(f"  Train Config: configs/atft/train/rankic_boost.yaml")
    logger.info(f"  Data: {atft_data_path}")
    logger.info(f"  Hidden Size: 256")
    logger.info(f"  Batch Size: 2048")
    logger.info(f"  Learning Rate: 5e-4")
    logger.info(f"  Max Epochs: 120")
    logger.info("Optimization:")
    logger.info(f"  RankIC Weight: 0.5 (maximum)")
    logger.info(f"  Sharpe Weight: 0.3")
    logger.info(f"  CS-IC Weight: 0.2")
    logger.info(f"  Workers: 8 (full parallelization)")
    logger.info(f"  Torch Compile: Enabled")
    logger.info("=" * 80)

    # Execute training
    try:
        logger.info("Starting training process...")
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            text=True,
            bufsize=1,  # Line buffered
        )

        logger.info("✅ Training completed successfully!")
        logger.info("Check logs/ directory for detailed results")
        logger.info("Checkpoints saved based on best val/rank_ic_5d")
        return 0

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Training failed with exit code {e.returncode}")
        if "CUDA out of memory" in str(e):
            logger.error("💡 Try reducing batch size in configs/atft/train/rankic_boost.yaml")
        return e.returncode

    except KeyboardInterrupt:
        logger.warning("⚠️ Training interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())