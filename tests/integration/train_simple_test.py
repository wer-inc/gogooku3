#!/usr/bin/env python
"""
Simplest possible test configuration - direct execution without subprocess.
This helps identify the exact issue.
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set minimal environment
os.environ.update({
    "CUDA_VISIBLE_DEVICES": "0",
    "NUM_WORKERS": "0",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "ALLOW_UNSAFE_DATALOADER": "0",
})

# Direct import and run
if __name__ == "__main__":
    # Set Hydra override arguments
    sys.argv = [
        "train_atft.py",
        "--config-path", str(PROJECT_ROOT / "configs/atft"),
        "--config-name", "config",  # Use base config
        # Minimal overrides
        f"data.source.data_dir={PROJECT_ROOT / 'output/atft_data'}",
        "model.hidden_size=64",  # Use smaller model for testing
        "improvements.compile_model=false",
        "train.batch.train_batch_size=256",  # Small batch
        "train.optimizer.lr=1e-4",
        "train.trainer.max_epochs=1",  # Just 1 epoch
        "train.trainer.devices=1",
        "data.distributed.enabled=false",
        "data.distributed.num_workers=0",
    ]

    print("=" * 60)
    print("🧪 SIMPLE TEST TRAINING")
    print("=" * 60)
    print("Configuration:")
    print("  - 1 epoch only")
    print("  - Small batch size (256)")
    print("  - Small model (hidden_size=64)")
    print("  - No optimizations")
    print("-" * 60)

    # Import and run
    try:
        # Change to script directory for proper execution
        os.chdir(PROJECT_ROOT / "scripts")

        # Import the training script
        import train_atft

        # Run training
        train_atft.train()

        print("\n✅ Training test completed!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)