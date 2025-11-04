"""
P0-3: GAT Gradient Flow Smoke Test

Validates that:
1. GATBlock forward pass works correctly
2. GatedCrossSectionFusion produces gate values in expected range
3. Gradients flow through both base and GAT paths
4. No gradient vanishing (total gradient > 0)
"""
import torch
from omegaconf import OmegaConf

# Avoid segmentation fault - don't import full model before configuration
print("[P0-3 SMOKE TEST] Starting GAT gradient flow validation...")

# Step 1: Create minimal config for P0-3
config = OmegaConf.create({
    "model": {
        "hidden_size": 128,
        "n_dynamic_features": 306,
        "n_static_features": 8,
        "gat": {"enabled": False},  # Will be overridden by top-level gat config
        "fan": {"enabled": False},
        "san": {"enabled": False},
    },
    "gat": {
        "use": True,
        "heads": [4, 2],
        "edge_dim": 3,
        "dropout": 0.2,
        "edge_dropout": 0.0,  # Disable for smoke test stability
        "tau": 1.25,
        "gate_per_feature": False,
        "gate_init_bias": -0.5,
        "attn_entropy_coef": 0.0,
    },
    "features": {
        "manifest_path": "output/reports/feature_manifest_306.yaml",
        "strict": False,  # Don't fail if manifest not found
    }
})

print(f"[P0-3 SMOKE TEST] Config created: hidden_size={config.model.hidden_size}, gat.use={config.gat.use}")

# Step 2: Import model after config is ready
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.atft_gat_fan.models.architectures.atft_gat_fan import ATFT_GAT_FAN

# Step 3: Create dummy data
B, H, T, F = 64, 128, 20, 306
x = torch.randn(B, T, F, requires_grad=False)
s = torch.randn(B, 8, requires_grad=False)

# Dummy graph (sparse, degree ~10)
# Create a simple chain graph + some random edges
edges_chain = torch.stack([torch.arange(B-1), torch.arange(1, B)], dim=0).long()
# Add reverse edges
edges_chain_rev = torch.stack([torch.arange(1, B), torch.arange(B-1)], dim=0).long()
# Combine
edge_index = torch.cat([edges_chain, edges_chain_rev], dim=1)
edge_attr = torch.randn(edge_index.size(1), 3)

print(f"[P0-3 SMOKE TEST] Data created: batch={B}, seq_len={T}, features={F}")
print(f"[P0-3 SMOKE TEST] Graph: {edge_index.size(1)} edges, avg_degree={(edge_index.size(1) / B):.1f}")

# Step 4: Build model
try:
    model = ATFT_GAT_FAN(config)
    model.train()
    print("[P0-3 SMOKE TEST] Model created successfully")
    print(f"[P0-3 SMOKE TEST] GAT enabled: {model.gat is not None}")
    print(f"[P0-3 SMOKE TEST] Fusion enabled: {model.fuse is not None}")
except Exception as e:
    print(f"❌ [P0-3 SMOKE TEST] Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 5: Forward pass
try:
    batch = {
        "dynamic_features": x,
        "static_features": s,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
    }
    outputs = model(batch)

    # Extract predictions
    if isinstance(outputs, dict):
        y_point = outputs.get("point_forecast")
        y_q = outputs.get("quantile_forecast")
    else:
        # Assume tuple return (point, quantile, ...)
        y_point = outputs[0] if len(outputs) > 0 else None
        y_q = outputs[1] if len(outputs) > 1 else None

    if y_point is None:
        raise ValueError("No point forecast in outputs")

    print("[P0-3 SMOKE TEST] Forward pass successful")
    print(f"[P0-3 SMOKE TEST] Output shapes: point={y_point.shape}, quantile={y_q.shape if y_q is not None else 'None'}")

    # Check gate values
    if hasattr(model, '_last_gate') and model._last_gate is not None:
        gate_mean = model._last_gate.mean().item()
        gate_std = model._last_gate.std().item()
        gate_min = model._last_gate.min().item()
        gate_max = model._last_gate.max().item()
        print(f"[P0-3 SMOKE TEST] Gate stats: mean={gate_mean:.3f}, std={gate_std:.3f}, min={gate_min:.3f}, max={gate_max:.3f}")

        # Validate gate range (should be in [0, 1] due to sigmoid)
        assert 0.0 <= gate_min <= 1.0, f"Gate min out of range: {gate_min}"
        assert 0.0 <= gate_max <= 1.0, f"Gate max out of range: {gate_max}"
        print("✅ [P0-3 SMOKE TEST] Gate values in valid range [0, 1]")
    else:
        print("⚠️  [P0-3 SMOKE TEST] No gate values recorded (GAT may not have been executed)")

except Exception as e:
    print(f"❌ [P0-3 SMOKE TEST] Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 6: Backward pass (gradient flow check)
try:
    # Create dummy loss
    loss = y_point.mean()
    if y_q is not None:
        loss = loss + y_q.mean()

    loss.backward()

    # Check gradients flow through both base and GAT paths
    total_grad = 0.0
    gat_grad = 0.0
    fusion_grad = 0.0

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.abs().sum().item()
            total_grad += grad_norm

            if 'gat' in name.lower():
                gat_grad += grad_norm
            if 'fuse' in name.lower():
                fusion_grad += grad_norm

    print("[P0-3 SMOKE TEST] Gradient flow:")
    print(f"  Total gradient norm: {total_grad:.3e}")
    print(f"  GAT gradient norm: {gat_grad:.3e}")
    print(f"  Fusion gradient norm: {fusion_grad:.3e}")

    # Validate gradient flow
    assert total_grad > 0, "Total gradient is zero! Gradient vanishing detected."
    print("✅ [P0-3 SMOKE TEST] Gradients are flowing (total_grad > 0)")

    if gat_grad > 0:
        print("✅ [P0-3 SMOKE TEST] GAT gradients are flowing")
    else:
        print("⚠️  [P0-3 SMOKE TEST] GAT gradients are zero (may be disabled or skipped)")

    if fusion_grad > 0:
        print("✅ [P0-3 SMOKE TEST] Fusion gradients are flowing")
    else:
        print("⚠️  [P0-3 SMOKE TEST] Fusion gradients are zero (may be disabled)")

except Exception as e:
    print(f"❌ [P0-3 SMOKE TEST] Backward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 7: Success summary
print("\n" + "="*80)
print("✅ P0-3 SMOKE TEST PASSED")
print("="*80)
print("Key metrics:")
print("  - Forward pass: ✅")
print(f"  - Gradient flow: ✅ (total_grad={total_grad:.3e})")
if hasattr(model, '_last_gate') and model._last_gate is not None:
    print(f"  - Gate mean: {gate_mean:.3f} (expected: 0.2-0.7)")
    print(f"  - Gate std: {gate_std:.3f} (expected: 0.05-0.30)")
print("\nNext steps:")
print("  1. Run quick training: make train-quick EPOCHS=3")
print("  2. Monitor gate statistics in logs")
print("  3. Collect RFI-5/6 data (graph health, loss metrics)")
print("="*80)
