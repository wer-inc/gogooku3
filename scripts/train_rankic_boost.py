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

    # Check if ATFT data exists; if not, try to auto-convert from latest ML dataset
    atft_data_path = project_root / "output/atft_data"
    if not atft_data_path.exists():
        logger.warning(f"⚠️ ATFT data not found at {atft_data_path}; attempting auto-conversion from latest dataset…")
        latest = project_root / "output/ml_dataset_latest_full.parquet"
        if not latest.exists():
            # fallback: pick the newest ml_dataset_*.parquet
            try:
                import glob
                cands = sorted(glob.glob(str(project_root / "output/ml_dataset_*_full.parquet")))
                if not cands:
                    cands = sorted(glob.glob(str(project_root / "output/ml_dataset_*.parquet")))
                latest = Path(cands[-1]) if cands else latest
            except Exception:
                pass

        if latest and latest.exists():
            try:
                logger.info(f"🔄 Converting ML dataset to ATFT format: {latest}")
                subprocess.run([
                    sys.executable,
                    str(project_root / "scripts/models/unified_feature_converter.py"),
                    "--input", str(latest),
                    "--output", "output/atft_data",
                ], check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Auto-conversion failed (exit={e.returncode}). Please convert manually.")
                return e.returncode
        else:
            logger.error("❌ No ML dataset found under output/. Run: make dataset-full-gpu START=… END=…")
            return 1

        if not atft_data_path.exists():
            logger.error("❌ ATFT data still missing after conversion. Aborting.")
            return 1

    # Check train/val/test splits
    required_dirs = ["train", "val", "test"]
    for dir_name in required_dirs:
        if not (atft_data_path / dir_name).exists():
            logger.error(f"❌ Missing {dir_name} directory in {atft_data_path}")
            return 1

    # Optional: quick dataset sanity check (targets, id columns, duplicates)
    try:
        sanity_script = project_root / "scripts/ci/dataset_sanity.py"
        if sanity_script.exists():
            logger.info("[preflight] Running dataset sanity checks (scripts/ci/dataset_sanity.py)")
            subprocess.run([sys.executable, str(sanity_script)], check=False)
        else:
            logger.info("[preflight] dataset_sanity.py not found; skipping")
    except Exception as e:
        logger.warning(f"[preflight] dataset sanity check skipped: {e}")

    # Setup environment variables for RankIC optimization
    env = os.environ.copy()

    # Core optimization settings
    rankic_env = {
        # RankIC and financial metrics optimization
        "USE_RANKIC": "1",
        "RANKIC_WEIGHT": "0.5",      # Maximum RankIC weight
        # Pairwise rank learning (RankNet-style)
        "USE_PAIRWISE_RANK": "1",
        "PAIRWISE_RANK_WEIGHT": "0.2",
        "PAIRWISE_SAMPLE_RATIO": "0.25",
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

        # Batch size optimization - CRITICAL for GPU utilization
        "BATCH_SIZE": "2048",        # Explicitly set batch size (fixes default 64 issue)
        "VAL_BATCH_SIZE": "4096",    # Larger validation batch for speed
        "MAX_BATCH_SIZE": "4096",    # Maximum allowed batch size

        # DataLoader optimization - Enable full parallelization
        "ALLOW_UNSAFE_DATALOADER": "1",  # Enable multi-process dataloader
        "NUM_WORKERS": "8",          # Use 8 workers for data loading
        "PERSISTENT_WORKERS": "1",   # Keep workers alive between epochs
        "PREFETCH_FACTOR": "4",      # Prefetch more batches
        "PIN_MEMORY": "1",           # Keep pinned memory for GPU transfer

        # GPU optimization
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_BACKENDS_CUDNN_BENCHMARK": "1",
        "TF32_ENABLED": "1",
        "CUDA_LAUNCH_BLOCKING": "0",

        # Mixed Precision Training - CRITICAL for A100 performance
        "USE_BF16": "1",             # Use BFloat16 for A100 optimization
        "AMP_ENABLED": "1",          # Enable automatic mixed precision
        "USE_SAFE_AMP": "1",         # Safe AMP mode
        "GRADIENT_CHECKPOINTING": "0",  # Disable for now (can enable if OOM)

        # Model configuration
        "HIDDEN_SIZE": "256",
        "MODEL_HIDDEN_SIZE": "256",

        # Torch compile
        "ENABLE_TORCH_COMPILE": "1",
        "TORCH_COMPILE_MODE": "max-autotune",

        # Safety features
        "DEGENERACY_GUARD": "1",
        "DEGENERACY_ABORT": "0",

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
    logger.info(f"  Batch Size: 2048 (Optimized for A100)")
    logger.info(f"  Val Batch Size: 4096")
    logger.info(f"  Learning Rate: 5e-4")
    logger.info(f"  Max Epochs: 120")
    logger.info("GPU Optimization:")
    logger.info(f"  Mixed Precision: BF16 enabled")
    logger.info(f"  Workers: 8 (parallel data loading)")
    logger.info(f"  Persistent Workers: Yes")
    logger.info(f"  Torch Compile: max-autotune")
    logger.info("Loss Weights:")
    logger.info(f"  RankIC Weight: 0.5 (maximum)")
    logger.info(f"  Sharpe Weight: 0.3")
    logger.info(f"  CS-IC Weight: 0.2")
    logger.info("=" * 80)

    # Execute training
    try:
        logger.info("Starting training process...")
        subprocess.run(
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

        # Check for specific error types and provide solutions
        if "CUDA out of memory" in str(e):
            logger.error("💡 Solution: Reduce batch size in configs/atft/train/rankic_boost.yaml")
            logger.error("   Current batch size: 2048")
            logger.error("   Try: 1024 or 512")

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
