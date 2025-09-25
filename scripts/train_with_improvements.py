#!/usr/bin/env python
"""
Performance-improved training script for ATFT-GAT-FAN model.

Improvements based on performance analysis:
1. Enable multi-worker DataLoader
2. Increase model capacity (hidden_size)
3. Optimize loss function weights (IC/RankIC)
4. Add PyTorch 2.x compilation
5. Improve learning rate scheduling
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup_improved_environment() -> Dict[str, str]:
    """Set up environment variables for improved performance."""

    env_vars = {
        # 1. Enable multi-worker DataLoader (CRITICAL for GPU utilization)
        "ALLOW_UNSAFE_DATALOADER": "1",
        "NUM_WORKERS": "8",  # Adjust based on CPU cores
        "PERSISTENT_WORKERS": "1",
        "PREFETCH_FACTOR": "4",
        "PIN_MEMORY": "1",

        # 2. Loss function optimization (focus on IC/RankIC)
        "USE_CS_IC": "1",
        "CS_IC_WEIGHT": "0.2",  # Increased from default 0.05
        "USE_RANKIC": "1",
        "SHARPE_WEIGHT": "0.3",  # If available

        # 3. PyTorch optimizations
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "CUDA_LAUNCH_BLOCKING": "0",
        "TORCH_BACKENDS_CUDNN_BENCHMARK": "1",
        "TF32_ENABLED": "1",

        # 4. Training settings
        "USE_SWA": "1",
        "SNAPSHOT_ENS": "1",

        # 5. Performance monitoring
        "MONITOR_GPU": "1",
        "LOG_LEVEL": "INFO"
    }

    # Preserve existing environment variables
    full_env = os.environ.copy()
    full_env.update(env_vars)

    return full_env


def create_improved_config():
    """Create improved configuration file with optimized settings."""

    improved_config = """# Performance-optimized ATFT-GAT-FAN configuration
model:
  name: atft_gat_fan
  # CRITICAL: Increase model capacity (was 64, now 256)
  hidden_size: 256  # Increased from 64 for better learning capacity

  # TFT settings with larger dimensions
  tft:
    hidden_size: ${model.hidden_size}
    n_head: 8
    dropout: 0.1

  # GAT settings with more capacity
  gat:
    enabled: true
    hidden_size: ${model.hidden_size}
    num_heads: 8  # Increased from 4
    dropout: 0.1

  # FAN settings
  fan:
    hidden_size: ${model.hidden_size}
    enabled: true

train:
  # Batch settings optimized for A100
  batch:
    batch_size: 2048  # Can increase with larger model
    accumulate_grad_batches: 2  # Effective batch = 4096
    num_workers: 8  # Multi-worker enabled
    persistent_workers: true
    prefetch_factor: 4
    pin_memory: true

  # Optimizer settings
  optimizer:
    lr: 5e-4  # Slightly higher for larger model
    weight_decay: 1e-5

  # Improved scheduler (Plateau instead of Cosine)
  scheduler:
    type: plateau  # Change from cosine to plateau
    patience: 5
    factor: 0.5
    min_lr: 1e-6

  # Training duration
  trainer:
    max_epochs: 100  # Increased from 75 for better convergence
    precision: bf16-mixed
    gradient_clip_val: 1.0

  # Loss weights optimized for financial metrics
  loss:
    quantile_weight: 0.5  # Reduced from 1.0
    sharpe_weight: 0.3    # Increased
    ic_weight: 0.2        # Increased
    rankic_weight: 0.1    # Added

  # Phase training adjustments
  phases:
    - name: baseline
      epochs: 10  # Increased from 5
      use_gat: false
      use_fan: false
    - name: gat
      epochs: 20  # Increased from 15
      use_gat: true
      use_fan: false
    - name: full
      epochs: 40  # Main training phase
      use_gat: true
      use_fan: true
    - name: finetune
      epochs: 30  # Extended fine-tuning
      use_gat: true
      use_fan: true
      lr_multiplier: 0.1

data:
  # Data loading optimized
  batch_sampler:
    enabled: true
    shuffle: true

  # Feature configuration
  features:
    # These should be populated based on actual dataset columns
    basic: []  # Will be filled from dataset analysis
    technical: []
    ma_derived: []
"""

    config_path = PROJECT_ROOT / "configs" / "atft" / "train" / "production_improved.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(improved_config)
    print(f"✅ Created improved config at: {config_path}")
    return config_path


def run_improved_training(config_path: Path = None):
    """Run training with all improvements enabled."""

    print("\n" + "="*60)
    print("🚀 PERFORMANCE-IMPROVED TRAINING PIPELINE")
    print("="*60)

    # Set up improved environment
    env = setup_improved_environment()

    print("\n📝 Environment improvements applied:")
    for key, value in sorted(env.items()):
        if key in ["ALLOW_UNSAFE_DATALOADER", "NUM_WORKERS", "CS_IC_WEIGHT", "USE_RANKIC"]:
            print(f"  - {key}={value}")

    # Create improved config if not provided
    if config_path is None:
        config_path = create_improved_config()

    # Prepare training command
    train_script = PROJECT_ROOT / "scripts" / "train_atft.py"

    cmd = [
        sys.executable,
        str(train_script),
        "--config-path", str(config_path.parent),
        "--config-name", config_path.stem,
        # Hydra overrides for additional improvements
        f"train.batch.num_workers={env['NUM_WORKERS']}",
        "train.scheduler.type=plateau",  # Force plateau scheduler
        "model.hidden_size=256",  # Force larger model
        "+compile.enabled=true",  # Enable torch.compile if supported
        "+compile.mode=max-autotune"
    ]

    print(f"\n🔧 Running command:")
    print(" ".join(cmd))
    print("\n" + "-"*60 + "\n")

    # Run training with improved settings
    try:
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            text=True,
            capture_output=False  # Show output in real-time
        )
        print("\n✅ Training completed successfully!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return 2


def validate_improvements():
    """Quick validation of improvements before full training."""

    print("\n🔍 Validating improvements with smoke test...")

    env = setup_improved_environment()
    env["MAX_EPOCHS"] = "1"  # Quick test

    smoke_script = PROJECT_ROOT / "scripts" / "smoke_test.py"

    cmd = [sys.executable, str(smoke_script)]

    try:
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout for smoke test
        )

        # Check if multi-worker is active
        if "num_workers=0" in result.stdout:
            print("⚠️ Warning: Multi-worker DataLoader may not be active")
        else:
            print("✅ Multi-worker DataLoader confirmed")

        # Check model size
        if "~2.7M" in result.stdout or "2.7M" in result.stdout:
            print("⚠️ Warning: Model size still small (2.7M params)")
        elif "20M" in result.stdout or "19M" in result.stdout:
            print("✅ Model size increased (~20M params)")

        return True

    except subprocess.TimeoutExpired:
        print("❌ Smoke test timeout - may indicate performance issues")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Smoke test failed: {e}")
        return False


def main():
    """Main entry point for improved training."""

    import argparse
    parser = argparse.ArgumentParser(description="Run improved ATFT-GAT-FAN training")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only run validation smoke test")
    parser.add_argument("--config", type=Path,
                       help="Path to custom config file")
    parser.add_argument("--skip-validation", action="store_true",
                       help="Skip validation and run full training")

    args = parser.parse_args()

    if args.validate_only:
        success = validate_improvements()
        return 0 if success else 1

    if not args.skip_validation:
        print("📋 Running pre-training validation...")
        if not validate_improvements():
            print("\n⚠️ Validation failed. Continue anyway? [y/N]: ", end="")
            response = input().strip().lower()
            if response != 'y':
                print("Aborted.")
                return 1

    # Run full improved training
    return run_improved_training(args.config)


if __name__ == "__main__":
    sys.exit(main())