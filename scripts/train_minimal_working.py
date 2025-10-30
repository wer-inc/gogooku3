#!/usr/bin/env python
"""
Minimal working training script with all issues fixed.
Focuses on stability over optimization.
"""

import os
import subprocess
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    """Run minimal working training configuration."""

    # Set up environment with minimal, stable settings
    env = os.environ.copy()

    # Minimal stable settings
    env.update({
        # Disable all problematic features
        "USE_MIXED_PRECISION": "0",  # Disable to avoid GradScaler issues
        "ENABLE_TORCH_COMPILE": "0",  # Disable torch.compile
        "USE_SWA": "0",  # Disable SWA to avoid deepcopy issues

        # Safe DataLoader settings
        "NUM_WORKERS": "0",
        "PERSISTENT_WORKERS": "0",
        "PIN_MEMORY": "0",

        # Disable all fancy features
        "USE_RANKIC": "0",
        "USE_CS_IC": "0",
        "USE_PHASE_TRAINING": "0",
        "USE_EMA": "0",

        # Basic model settings (keep PDF recommendation)
        "MODEL_HIDDEN_SIZE": "256",

        # Memory settings
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",

        # Debug settings
        "HYDRA_FULL_ERROR": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
    })

    # Check if ATFT data exists
    atft_data_path = PROJECT_ROOT / "output/atft_data"
    if not atft_data_path.exists():
        print(f"❌ ATFT data not found at {atft_data_path}")
        print("Run: python scripts/data/unified_feature_converter.py first")
        return 1

    # Build minimal command
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/train_atft.py"),
        "--config-path", str(PROJECT_ROOT / "configs/atft"),
        "--config-name", "config_production",

        # Basic overrides only
        f"data.source.data_dir={atft_data_path}",

        # Model settings
        "model.hidden_size=256",

        # Disable all optimizations (use correct paths)
        "improvements.compile_model=false",
        "improvements.use_ema=false",

        # Training settings
        "train.batch.train_batch_size=512",  # Smaller batch size
        "train.optimizer.lr=1e-4",  # Lower learning rate
        "train.trainer.max_epochs=5",  # Just test 5 epochs
        "train.trainer.precision=32",  # Disable mixed precision
        "train.trainer.gradient_clip_val=1.0",

        # Disable distributed
        "data.distributed.enabled=false",
        "data.distributed.num_workers=0",

        # Ensure single GPU
        "train.trainer.devices=1",
        "train.trainer.strategy=auto",
    ]

    print("=" * 60)
    print("🔧 MINIMAL WORKING TRAINING (Debug Mode)")
    print("=" * 60)
    print("Settings:")
    print("  ⚠️  All optimizations disabled for stability")
    print("  ✅ Single-process DataLoader")
    print("  ✅ No mixed precision (fp32 only)")
    print("  ✅ No torch.compile")
    print("  ✅ Small batch size (512)")
    print("  ✅ 5 epochs only (testing)")
    print("  ✅ Model hidden_size=256")
    print("-" * 60)
    print("Command:", " ".join(cmd[:5]) + "...")
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
