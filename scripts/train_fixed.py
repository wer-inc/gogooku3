#!/usr/bin/env python
"""
Fixed training script with all critical issues resolved.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    """Run fixed training configuration."""

    # Set up environment with all fixes
    env = os.environ.copy()

    # Critical fixes
    env.update({
        # Fix 1: Disable mixed precision to avoid GradScaler errors
        "APEX_ENABLED": "0",
        "USE_AMP": "0",
        "MIXED_PRECISION": "0",

        # Fix 2: Single-process DataLoader
        "NUM_WORKERS": "0",
        "PERSISTENT_WORKERS": "0",
        "PIN_MEMORY": "0",

        # Fix 3: Disable problematic features
        "USE_SWA": "0",  # Avoid deepcopy errors
        "USE_EMA": "0",
        "ENABLE_TORCH_COMPILE": "0",

        # PDF optimizations (keep these)
        "MODEL_HIDDEN_SIZE": "256",
        "USE_RANKIC": "1",
        "RANKIC_WEIGHT": "0.2",
        "USE_CS_IC": "1",
        "CS_IC_WEIGHT": "0.15",
        "SHARPE_WEIGHT": "0.3",

        # Memory settings
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",

        # Debug
        "HYDRA_FULL_ERROR": "0",
    })

    # Check if ATFT data exists
    atft_data_path = PROJECT_ROOT / "output/atft_data"
    if not atft_data_path.exists():
        print(f"❌ ATFT data not found at {atft_data_path}")
        print("Run: python scripts/data/unified_feature_converter.py first")
        return 1

    # Build command with fixes
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/train_atft.py"),
        "--config-path", str(PROJECT_ROOT / "configs/atft"),
        "--config-name", "config",  # Use base config

        # Data configuration
        f"data.source.data_dir={atft_data_path}",

        # Model settings (PDF optimizations)
        "model.hidden_size=256",

        # Training settings
        "train.batch.train_batch_size=1024",
        "train.optimizer.lr=2e-4",
        "train.trainer.max_epochs=10",  # Start with 10 epochs

        # CRITICAL FIX: Disable mixed precision
        "train.trainer.precision=32",  # Use fp32, not 16 or bf16-mixed

        # Other fixes
        "train.trainer.gradient_clip_val=1.0",
        "train.trainer.devices=1",
        "train.trainer.accumulate_grad_batches=1",

        # Disable problematic features
        "improvements.compile_model=false",
        "improvements.use_ema=false",

        # Data settings
        "data.distributed.enabled=false",
        "data.distributed.num_workers=0",
    ]

    print("=" * 60)
    print("🔧 FIXED TRAINING CONFIGURATION")
    print("=" * 60)
    print("Fixes Applied:")
    print("  ✅ Mixed precision disabled (fp32 only)")
    print("  ✅ Single-process DataLoader")
    print("  ✅ SWA/EMA disabled")
    print("  ✅ torch.compile disabled")
    print("")
    print("PDF Optimizations:")
    print("  ✅ Model hidden_size=256")
    print("  ✅ RankIC/Sharpe optimization")
    print("  ✅ Batch size 1024")
    print("  ✅ Learning rate 2e-4")
    print("  ✅ 10 epochs")
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