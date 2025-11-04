"""
P0-3: Component-level smoke test (avoids full model import segfault)

Tests individual P0-3 components:
1. GATBlock
2. GatedCrossSectionFusion
3. Edge utilities
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

print("[P0-3 COMPONENT TEST] Starting component validation...")

# Test 1: GATBlock
print("\n[TEST 1] GATBlock")
try:
    from src.atft_gat_fan.models.components.gat_fuse import GATBlock

    B, H = 64, 128
    z = torch.randn(B, H, requires_grad=True)

    # Create simple chain graph
    edge_index = torch.stack([torch.arange(B-1), torch.arange(1, B)], dim=0).long()
    edge_attr = torch.randn(edge_index.size(1), 3)

    gat_block = GATBlock(
        in_dim=H,
        hidden_dim=H,
        heads=(4, 2),
        edge_dim=3,
        dropout=0.2
    )

    z_out = gat_block(z, edge_index, edge_attr)

    print(f"  Input shape: {z.shape}")
    print(f"  Output shape: {z_out.shape}")
    assert z_out.shape == z.shape, f"Shape mismatch: {z_out.shape} != {z.shape}"

    # Gradient check
    loss = z_out.mean()
    loss.backward()
    grad_norm = z.grad.abs().sum().item()
    print(f"  Gradient norm: {grad_norm:.3e}")
    assert grad_norm > 0, "Gradient is zero!"

    print("  ✅ GATBlock test passed")

except Exception as e:
    print(f"  ❌ GATBlock test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 2: GatedCrossSectionFusion
print("\n[TEST 2] GatedCrossSectionFusion")
try:
    from src.atft_gat_fan.models.components.gat_fuse import GatedCrossSectionFusion

    B, H = 64, 128
    z_base = torch.randn(B, H, requires_grad=True)
    z_gat = torch.randn(B, H, requires_grad=True)

    fusion = GatedCrossSectionFusion(
        hidden=H,
        gate_per_feature=False,
        tau=1.25,
        init_bias=-0.5
    )

    z_fused, gate = fusion(z_base, z_gat)

    print(f"  Base shape: {z_base.shape}")
    print(f"  GAT shape: {z_gat.shape}")
    print(f"  Fused shape: {z_fused.shape}")
    print(f"  Gate shape: {gate.shape}")
    print(f"  Gate stats: mean={gate.mean().item():.3f}, std={gate.std().item():.3f}, min={gate.min().item():.3f}, max={gate.max().item():.3f}")

    # Check gate range
    assert 0.0 <= gate.min().item() <= 1.0, f"Gate min out of range: {gate.min().item()}"
    assert 0.0 <= gate.max().item() <= 1.0, f"Gate max out of range: {gate.max().item()}"

    # Gradient check
    loss = z_fused.mean()
    loss.backward()

    grad_base = z_base.grad.abs().sum().item()
    grad_gat = z_gat.grad.abs().sum().item()

    print(f"  Base gradient: {grad_base:.3e}")
    print(f"  GAT gradient: {grad_gat:.3e}")

    assert grad_base > 0, "Base gradient is zero!"
    assert grad_gat > 0, "GAT gradient is zero!"

    print("  ✅ GatedCrossSectionFusion test passed")

except Exception as e:
    print(f"  ❌ GatedCrossSectionFusion test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Edge utilities
print("\n[TEST 3] Edge utilities")
try:
    from src.graph.graph_utils import apply_edge_dropout, standardize_edge_attr

    edge_index = torch.stack([torch.arange(63), torch.arange(1, 64)], dim=0).long()
    edge_attr = torch.randn(edge_index.size(1), 3)

    # Test standardization
    edge_attr_std = standardize_edge_attr(edge_attr)
    print(f"  Original edge_attr mean: {edge_attr.mean(dim=0).tolist()}")
    print(f"  Standardized edge_attr mean: {edge_attr_std.mean(dim=0).tolist()}")
    print(f"  Standardized edge_attr std: {edge_attr_std.std(dim=0).tolist()}")

    # Check standardization (mean ≈ 0, std ≈ 1)
    assert edge_attr_std.mean(dim=0).abs().max().item() < 1e-5, "Mean not close to 0"
    assert (edge_attr_std.std(dim=0) - 1.0).abs().max().item() < 1e-5, "Std not close to 1"

    # Test edge dropout
    edge_index_drop, edge_attr_drop = apply_edge_dropout(
        edge_index, edge_attr, p=0.2, training=True
    )

    print(f"  Original edges: {edge_index.size(1)}")
    print(f"  After dropout: {edge_index_drop.size(1)}")
    print(f"  Dropout rate: {1.0 - edge_index_drop.size(1) / edge_index.size(1):.1%}")

    # Check dropout worked (should have fewer edges)
    assert edge_index_drop.size(1) < edge_index.size(1), "Dropout didn't remove any edges"
    assert edge_index_drop.size(1) > 0, "Dropout removed all edges"

    print("  ✅ Edge utilities test passed")

except Exception as e:
    print(f"  ❌ Edge utilities test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Success summary
print("\n" + "="*80)
print("✅ ALL P0-3 COMPONENT TESTS PASSED")
print("="*80)
print("Components validated:")
print("  1. GATBlock: Forward + backward ✅")
print("  2. GatedCrossSectionFusion: Gating + gradients ✅")
print("  3. Edge utilities: Standardization + dropout ✅")
print("\nP0-3 core components are working correctly!")
print("="*80)
