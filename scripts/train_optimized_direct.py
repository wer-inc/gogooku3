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

    # Critical optimizations from PDF (OPTIMIZED for A100)
    env.update({
        "ALLOW_UNSAFE_DATALOADER": "1",
        "NUM_WORKERS": "4",  # FIX: Reduced to 4 to prevent thread explosion (conservative start)
        "PERSISTENT_WORKERS": "1",  # OPTIMIZATION: Reuse workers for efficiency
        "PREFETCH_FACTOR": "2",  # FIX: Reduced to 2 to limit memory usage
        "PIN_MEMORY": "1",
        "USE_DAY_BATCH": "0",  # Disable day-batch sampling to keep GPU busy early
        "USE_GRAPH_IN_TRAINING": "1",  # Default: build correlation graphs during training
        "VAL_BATCH_SIZE": "1024",  # Larger val micro-batch for stable metrics
        "SHARPE_EPS": "1e-6",  # Avoid NaN Sharpe when std is tiny
        "PHASE_TRAINING": "1",  # Keep phased training but allow overrides
        "PHASE0_EPOCHS": "2",
        "PHASE1_EPOCHS": "6",
        "PHASE2_EPOCHS": "16",
        "PHASE3_EPOCHS": "8",
        "PHASE_WARMUP_EPOCHS": "4",
        "PHASE_MAX_BATCHES": "0",
        "FUSE_START_PHASE": "0",
        "USE_ADV_GRAPH_TRAIN": "1",  # Enable training-time graph builder optimizations
        "GRAPH_EDGE_THR": "0.20",  # Slightly denser graph for better context
        "GRAPH_K_DEFAULT": "20",  # Increase neighbors for GAT context
        "GRAPH_MIN_EDGES": "50",
        "BATCH_SIZE": "512",  # Ensure train_atft picks large micro-batch via env fallback
        "OMP_NUM_THREADS": "4",  # CRITICAL FIX: Limit OpenMP threads (4 workers × 4 threads = 16 total)
        "USE_RANKIC": "1",
        "RANKIC_WEIGHT": "0.1",
        "USE_CS_IC": "1",
        "CS_IC_WEIGHT": "0.15",
        "SHARPE_WEIGHT": "0.5",
        "MODEL_HIDDEN_SIZE": "256",
        "FEATURE_CLIP_VALUE": "8",  # FIX: Clip features to ±8 for numerical stability
        "ENABLE_TORCH_COMPILE": "0",  # TEMPORARY: Disable to test GPU usage (torch.compile may cause CPU fallback)
        "TORCH_COMPILE_MODE": "reduce-overhead",  # FIX: max-autotune causes CUDA misaligned address errors
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_BACKENDS_CUDNN_BENCHMARK": "1",
        "TF32_ENABLED": "1",
        "CUDA_LAUNCH_BLOCKING": "0",  # Set to 1 for debugging CUDA errors
        "OUTPUT_BASE": str(PROJECT_ROOT / "output"),  # FIX: Required by config interpolation
        "SCHEDULER": "warmup_cosine",
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
        "--config-name", "config_production_optimized",  # OPTIMIZATION: Use fully optimized config
        # Add overrides AFTER config arguments
        f"data.source.data_dir={atft_data_path}",
        "model.hidden_size=256",
        "model.optimization.compile.enabled=false",  # TEMPORARY: Disable torch.compile to test GPU usage
        # FIX: Conservative batch settings (Phase 1)
        "+train.batch.train_batch_size=512",
        "+train.batch.gradient_accumulation_steps=16",  # FIX: Effective batch = 512 × 16 = 8192 (large effective but more frequent updates)
        # FIX: DataLoader settings aligned with environment variables
        "train.batch.num_workers=4",  # FIX: Match NUM_WORKERS=4 to prevent thread explosion
        "+train.batch.val_batch_size=1024",
        "train.batch.prefetch_factor=2",  # FIX: Match PREFETCH_FACTOR=2
        "train.batch.persistent_workers=true",
        "train.batch.pin_memory=true",
        # FIX: Drop undersized daily batches (TODO: implement later)
        # "data.sampling.min_nodes_per_day=256",  # May need config schema update
        "train.optimizer.lr=5e-4",
        "train.trainer.max_epochs=120",
        f"data.graph_builder.cache_dir={PROJECT_ROOT / 'graph_cache'}",  # OPTIMIZATION: Enable graph caching
    ]

    use_graph_env = env.get("USE_GRAPH_IN_TRAINING", "1").strip().lower()
    if use_graph_env in ("0", "false", "off"):
        cmd.append("data.graph_builder.use_in_training=false")
    else:
        cmd.append("data.graph_builder.use_in_training=true")

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