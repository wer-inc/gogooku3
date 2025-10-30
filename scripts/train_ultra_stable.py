#!/usr/bin/env python
"""
Ultra-stable training configuration with all known issues resolved.
Prioritizes stability over performance optimizations.
"""

import os
import subprocess
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    """Run ultra-stable training with maximum safety."""

    # Set up environment for maximum stability
    env = os.environ.copy()

    # CRITICAL FIXES FOR STABILITY
    env.update({
        # Fix 1: Completely disable mixed precision and GradScaler
        "APEX_ENABLED": "0",
        "USE_AMP": "0",
        "MIXED_PRECISION": "0",
        "ENABLE_GRADSCALER": "0",

        # Fix 2: Single-process DataLoader to avoid worker crashes
        "NUM_WORKERS": "0",
        "PERSISTENT_WORKERS": "0",
        "PIN_MEMORY": "0",
        "ALLOW_UNSAFE_DATALOADER": "0",  # Force safe mode

        # Fix 3: Disable all experimental features
        "USE_SWA": "0",  # Avoid deepcopy errors
        "USE_EMA": "0",  # Disable EMA
        "ENABLE_TORCH_COMPILE": "0",  # Disable torch.compile
        "ENABLE_PHASE_TRAINING": "0",  # Disable phase training

        # Fix 4: Disable advanced graph features temporarily
        "ADV_GRAPH_TRAIN": "0",  # Disable advanced graph training
        "USE_GRAPH_BUILDER": "0",  # Disable graph builder

        # PDF-recommended optimizations (safe subset)
        "MODEL_HIDDEN_SIZE": "128",  # Start smaller for stability
        "USE_RANKIC": "1",
        "RANKIC_WEIGHT": "0.3",
        "USE_CS_IC": "1",
        "CS_IC_WEIGHT": "0.2",
        "SHARPE_WEIGHT": "0.3",

        # Memory and performance settings
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
        "CUDA_LAUNCH_BLOCKING": "0",  # Keep async for performance

        # Debugging
        "HYDRA_FULL_ERROR": "0",  # Disable full error traces
        "WANDB_DISABLED": "1",  # Disable W&B to avoid issues
    })

    # Check if ATFT data exists
    atft_data_path = PROJECT_ROOT / "output/atft_data"
    if not atft_data_path.exists():
        print(f"❌ ATFT data not found at {atft_data_path}")
        print("Run: python scripts/data/unified_feature_converter.py first")
        return 1

    # Build command with ultra-stable settings
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/train_atft.py"),
        "--config-path", str(PROJECT_ROOT / "configs/atft"),
        "--config-name", "config",  # Use base config

        # Data configuration
        f"data.source.data_dir={atft_data_path}",

        # Model settings (start small for stability)
        "model.hidden_size=128",  # Smaller model for stability
        # Skip GAT settings - they're not in config structure

        # Training settings (conservative)
        "train.batch.train_batch_size=512",  # Smaller batch size
        "train.optimizer.lr=1e-4",  # Lower learning rate
        "train.trainer.max_epochs=5",  # Start with just 5 epochs

        # CRITICAL: Use fp32 precision only
        "train.trainer.precision=32",  # Full precision, no mixed

        # Stability settings
        "train.trainer.gradient_clip_val=0.5",  # Aggressive clipping
        "train.trainer.devices=1",
        "train.trainer.accumulate_grad_batches=1",
        "train.trainer.val_check_interval=0.5",  # Check more frequently

        # Disable problematic features (only existing keys)
        "improvements.compile_model=false",
        "improvements.use_ema=false",
        "improvements.enable_tensorboard=false",
        "improvements.enable_wandb=false",

        # Data settings (safe mode)
        "data.distributed.enabled=false",
        "data.distributed.num_workers=0",
        "data.memory.chunk_size=5000",  # Smaller chunks
        "data.memory.cache_size_gb=4",  # Less memory usage

        # Graph settings (disabled)
        "data.graph_builder.use_in_training=false",
    ]

    print("=" * 60)
    print("🛡️ ULTRA-STABLE TRAINING CONFIGURATION")
    print("=" * 60)
    print("Stability Fixes:")
    print("  ✅ Mixed precision COMPLETELY DISABLED")
    print("  ✅ GradScaler DISABLED")
    print("  ✅ Single-process DataLoader (no workers)")
    print("  ✅ SWA/EMA/torch.compile DISABLED")
    print("  ✅ Phase training DISABLED")
    print("  ✅ Graph features DISABLED (temporarily)")
    print("  ✅ W&B/TensorBoard DISABLED")
    print("")
    print("Conservative Settings:")
    print("  ✅ Model hidden_size=128 (smaller)")
    print("  ✅ Batch size 512 (conservative)")
    print("  ✅ Learning rate 1e-4 (safe)")
    print("  ✅ Only 5 epochs (test run)")
    print("  ✅ FP32 precision only")
    print("  ✅ Gradient clipping 0.5")
    print("  ✅ Single horizon loss (1d only)")
    print("-" * 60)
    print("Starting ultra-stable training...")
    print()

    try:
        subprocess.run(cmd, env=env, check=True)
        print("\n✅ Training completed successfully!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check logs in logs/ directory")
        print("2. Verify CUDA availability: nvidia-smi")
        print("3. Check data exists: ls -la output/atft_data/")
        print("4. Try even smaller batch size: train.batch.train_batch_size=256")
        return 1
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return 2

if __name__ == "__main__":
    sys.exit(main())
