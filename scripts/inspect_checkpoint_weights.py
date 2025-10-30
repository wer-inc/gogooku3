#!/usr/bin/env python3
"""
Tier 1.1: Inspect Checkpoint Weights
Diagnose model degeneracy by examining prediction head weights.
"""
from pathlib import Path

import torch


def inspect_checkpoint(checkpoint_path: str):
    """Inspect prediction head weights in checkpoint."""
    print("=" * 80)
    print(f"Inspecting Checkpoint: {checkpoint_path}")
    print("=" * 80)

    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        print("✅ Checkpoint loaded successfully\n")
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return

    # Extract state dict
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            print("📦 Checkpoint structure: {state_dict, ...}")
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            print("📦 Checkpoint structure: {model_state_dict, ...}")
        else:
            state_dict = checkpoint
            print("📦 Checkpoint structure: Direct state dict")
    else:
        print("❌ Unexpected checkpoint format")
        return

    print(f"📊 Total parameters: {len(state_dict)} tensors\n")

    # Find prediction head layers
    print("=" * 80)
    print("🔍 PREDICTION HEAD ANALYSIS")
    print("=" * 80)

    head_keys = [
        k for k in state_dict.keys() if "horizon" in k.lower() or "head" in k.lower()
    ]

    if not head_keys:
        print("⚠️  No prediction head keys found!")
        print(f"Available keys sample: {list(state_dict.keys())[:10]}")
        return

    print(f"Found {len(head_keys)} prediction head parameters:\n")

    for key in sorted(head_keys):
        tensor = state_dict[key]

        # Compute statistics
        mean = tensor.mean().item()
        std = tensor.std().item()
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        abs_mean = tensor.abs().mean().item()

        print(f"📌 {key}")
        print(f"   Shape: {tuple(tensor.shape)}")
        print(f"   Mean: {mean:+.6f}  |  Std: {std:.6f}")
        print(f"   Min: {min_val:+.6f}  |  Max: {max_val:+.6f}")
        print(f"   Abs Mean: {abs_mean:.6f}")

        # Diagnosis
        if abs_mean < 1e-6:
            print(f"   🚨 WARNING: Near-zero weights (abs_mean={abs_mean:.2e})")
        elif abs_mean < 1e-3:
            print(f"   ⚠️  Very small weights (abs_mean={abs_mean:.2e})")
        else:
            print("   ✅ Weights look reasonable")

        print()

    # Check for other suspicious layers
    print("=" * 80)
    print("🔍 OTHER LAYERS ANALYSIS")
    print("=" * 80)

    suspicious_patterns = ["final", "output", "fc", "linear"]
    suspicious_keys = []
    for pattern in suspicious_patterns:
        suspicious_keys.extend(
            [
                k
                for k in state_dict.keys()
                if pattern in k.lower() and k not in head_keys
            ]
        )

    if suspicious_keys:
        print(f"Found {len(suspicious_keys)} potentially final layer parameters:\n")
        for key in sorted(set(suspicious_keys))[:10]:  # Show first 10
            tensor = state_dict[key]
            abs_mean = tensor.abs().mean().item()
            print(f"📌 {key}")
            print(f"   Shape: {tuple(tensor.shape)}  |  Abs Mean: {abs_mean:.6f}")
            if abs_mean < 1e-6:
                print("   🚨 WARNING: Near-zero")
            print()

    # Summary statistics
    print("=" * 80)
    print("📊 SUMMARY STATISTICS")
    print("=" * 80)

    all_params_abs_mean = []
    for key, tensor in state_dict.items():
        if tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]:
            all_params_abs_mean.append(tensor.abs().mean().item())

    if all_params_abs_mean:
        print(f"Total trainable parameters: {len(all_params_abs_mean)}")
        print(
            f"Average abs weight across all layers: {sum(all_params_abs_mean)/len(all_params_abs_mean):.6f}"
        )
        print(f"Min abs weight: {min(all_params_abs_mean):.6f}")
        print(f"Max abs weight: {max(all_params_abs_mean):.6f}")

        near_zero_count = sum(1 for x in all_params_abs_mean if x < 1e-6)
        print(
            f"Layers with near-zero weights (<1e-6): {near_zero_count}/{len(all_params_abs_mean)}"
        )

    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    # Check Phase 3 trial checkpoint first
    phase3_ckpt = Path("models/checkpoints/best_main.pt")
    phase2_ckpt = Path("models/checkpoints/atft_gat_fan_final.pt")

    if phase3_ckpt.exists():
        print("\n🔬 Inspecting Phase 3 Trial Checkpoint\n")
        inspect_checkpoint(str(phase3_ckpt))

    if phase2_ckpt.exists():
        print("\n\n" + "=" * 80)
        print("🔬 Inspecting Phase 2 Baseline Checkpoint (for comparison)")
        print("=" * 80 + "\n")
        inspect_checkpoint(str(phase2_ckpt))

    if not phase3_ckpt.exists() and not phase2_ckpt.exists():
        print("❌ No checkpoints found!")
        print(f"   Looking for: {phase3_ckpt} or {phase2_ckpt}")
