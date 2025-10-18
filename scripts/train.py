#!/usr/bin/env python
"""
Unified Training Entry Point for ATFT-GAT-FAN Model

This script provides a single interface for training with different modes.
It wraps integrated_ml_training_pipeline.py with sensible defaults.

Usage:
    # Quick validation (3 epochs)
    python scripts/train.py --epochs 3

    # Full training (optimized)
    python scripts/train.py --epochs 120 --batch-size 2048

    # Safe mode (stable, single-worker)
    python scripts/train.py --epochs 120 --mode safe

    # Via Makefile (recommended)
    make train
    make train-quick
    make train-safe
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified training entry point")

    parser.add_argument(
        "--data-path",
        type=str,
        default="output/ml_dataset_latest_full.parquet",
        help="Path to ML dataset",
    )
    parser.add_argument("--epochs", type=int, default=120, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--hidden-size", type=int, default=256, help="Model hidden size"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["optimized", "safe", "standard"],
        default="optimized",
        help="Training mode",
    )
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument(
        "--validate-only", action="store_true", help="Only validate configuration"
    )
    parser.add_argument("--background", action="store_true", help="Run in background")
    parser.add_argument(
        "--no-background", action="store_true", help="Run in foreground"
    )

    args = parser.parse_args()

    # Setup environment based on mode
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["USE_AMP"] = "1"
    env["AMP_DTYPE"] = "bf16"
    env["ENABLE_FEATURE_NORM"] = "1"
    env["FEATURE_CLIP_VALUE"] = "10.0"

    if args.mode == "optimized":
        env["ALLOW_UNSAFE_DATALOADER"] = "1"
        env["NUM_WORKERS"] = str(args.num_workers)
        env["PERSISTENT_WORKERS"] = "1"
        env["PREFETCH_FACTOR"] = "4"
        env["USE_RANKIC"] = "1"
        env["RANKIC_WEIGHT"] = "0.5"  # Increased from 0.2 for stronger RankIC focus
        env["CS_IC_WEIGHT"] = "0.3"  # Increased from 0.15 for better IC learning
        env["SHARPE_WEIGHT"] = "0.1"  # Reduced from 0.3 to prioritize RankIC/IC
    elif args.mode == "safe":
        env["ALLOW_UNSAFE_DATALOADER"] = "0"
        env["NUM_WORKERS"] = "0"
        env["FORCE_SINGLE_PROCESS"] = "1"
        env["USE_RANKIC"] = "1"
        env["RANKIC_WEIGHT"] = "0.5"  # Increased from 0.1 for stronger RankIC focus
        env["CS_IC_WEIGHT"] = "0.3"  # Added for better IC learning
        env["SHARPE_WEIGHT"] = "0.1"  # Added to complete loss function config

    # Validation only
    if args.validate_only:
        print("🔍 Validating configuration...")
        print(f"   Mode: {args.mode}")
        print(f"   Epochs: {args.epochs}")
        print(f"   Batch size: {args.batch_size}")
        print(f"   Dataset: {args.data_path}")

        if not Path(args.data_path).exists():
            print(f"❌ Dataset not found: {args.data_path}")
            return 1

        print("✅ Configuration valid")
        return 0

    # Build command for integrated_ml_training_pipeline.py
    script_path = Path(__file__).parent / "integrated_ml_training_pipeline.py"
    cmd = [
        "python",
        str(script_path),
        "--data-path",
        args.data_path,
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--max-epochs",
        str(args.epochs),
    ]

    if args.compile:
        cmd.append("--compile")

    print(f"\n🚀 Starting training ({args.mode} mode)...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print("")

    try:
        result = subprocess.run(cmd, env=env, check=False)  # noqa: S603
        return result.returncode
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted")
        return 130
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
