#!/usr/bin/env python
"""
Safe optimized training script with conservative settings to avoid crashes.
Uses single-worker DataLoader to ensure stability.
"""

import os
import subprocess
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    """Run safe optimized training."""

    # Set up environment with conservative settings
    env = os.environ.copy()

    # Safe settings to avoid crashes
    env.update({
        # DataLoader settings (conservative)
        "NUM_WORKERS": "0",  # Single process to avoid crashes
        "PERSISTENT_WORKERS": "0",
        "PREFETCH_FACTOR": "2",
        "PIN_MEMORY": "0",  # Disable to reduce memory issues

        # Critical optimizations from PDF (keep these)
        "USE_RANKIC": "1",
        "RANKIC_WEIGHT": "0.2",
        "USE_CS_IC": "1",
        "CS_IC_WEIGHT": "0.15",
        "SHARPE_WEIGHT": "0.3",
        "MODEL_HIDDEN_SIZE": "256",
        "ENABLE_TORCH_COMPILE": "0",  # Disable compile for stability
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
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/train_atft.py"),
        "--config-path", str(PROJECT_ROOT / "configs/atft"),
        "--config-name", "config_production",
        # Add overrides AFTER config arguments
        f"data.source.data_dir={atft_data_path}",
        "model.hidden_size=256",
        "improvements.compile_model=false",  # Disable for stability
        "train.batch.train_batch_size=1024",  # Reduced batch size
        "train.optimizer.lr=5e-4",
        "train.trainer.max_epochs=10",  # Start with fewer epochs for testing
        "train.trainer.gradient_clip_val=1.0",
        # Disable multi-worker explicitly
        "data.distributed.enabled=false",
        "data.distributed.num_workers=0",
    ]

    print("=" * 60)
    print("🛡️ SAFE OPTIMIZED TRAINING (Conservative Settings)")
    print("=" * 60)
    print("Settings:")
    print("  - Single-worker DataLoader (no crashes)")
    print("  - Reduced batch size (1024)")
    print("  - Disabled torch.compile (stability)")
    print("  - 10 epochs for initial testing")
    print("  - Model hidden_size=256 (PDF recommendation)")
    print("  - RankIC/Sharpe optimization enabled")
    print("-" * 60)
    print("Command:", " ".join(cmd))
    print("-" * 60)

    try:
        subprocess.run(cmd, env=env, check=True)
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
