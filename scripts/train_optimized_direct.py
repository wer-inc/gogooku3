#!/usr/bin/env python
"""
Direct optimized training script that bypasses the integrated pipeline.
Uses train_atft.py directly with the optimized configuration.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    """Run direct optimized training."""

    # Set up environment
    env = os.environ.copy()

    # Critical optimizations from PDF (with safer worker settings)
    env.update({
        "ALLOW_UNSAFE_DATALOADER": "1",
        "NUM_WORKERS": "2",  # Reduced from 8 to avoid crashes
        "PERSISTENT_WORKERS": "0",  # Disable to avoid worker issues
        "PREFETCH_FACTOR": "2",  # Reduced from 4
        "PIN_MEMORY": "1",
        "USE_RANKIC": "1",
        "RANKIC_WEIGHT": "0.2",
        "USE_CS_IC": "1",
        "CS_IC_WEIGHT": "0.15",
        "SHARPE_WEIGHT": "0.3",
        "MODEL_HIDDEN_SIZE": "256",
        "ENABLE_TORCH_COMPILE": "1",
        "TORCH_COMPILE_MODE": "max-autotune",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_BACKENDS_CUDNN_BENCHMARK": "1",
        "TF32_ENABLED": "1",
        "CUDA_LAUNCH_BLOCKING": "0",
    })

    # Check if ATFT data exists
    atft_data_path = PROJECT_ROOT / "output/atft_data"
    if not atft_data_path.exists():
        print(f"❌ ATFT data not found at {atft_data_path}")
        print("Run: python scripts/data/unified_feature_converter.py first")
        return 1

    # Build command for direct train_atft.py execution
    # Hydra overrides must come AFTER config arguments
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/train_atft.py"),
        "--config-path", str(PROJECT_ROOT / "configs/atft"),
        "--config-name", "config_production",  # Use working config
        # Add overrides AFTER config arguments
        f"data.source.data_dir={atft_data_path}",
        "model.hidden_size=256",
        "improvements.compile_model=true",
        "train.batch.train_batch_size=2048",  # Correct path
        "train.optimizer.lr=5e-4",
        "train.trainer.max_epochs=120",
    ]

    print("=" * 60)
    print("🚀 DIRECT OPTIMIZED TRAINING")
    print("=" * 60)
    print("Command:", " ".join(cmd))
    print("-" * 60)

    try:
        result = subprocess.run(cmd, env=env, check=True)
        print("\n✅ Training completed successfully!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return 2

if __name__ == "__main__":
    sys.exit(main())