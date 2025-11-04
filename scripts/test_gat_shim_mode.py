"""
P0-3: GraphConvShim 動作テスト

USE_GAT_SHIM=1モードでの動作確認。
torch_geometricを使わずにGATBlock/Fusionが動作することを検証。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os

os.environ["USE_GAT_SHIM"] = "1"  # 強制的にシムモード

import torch

print("="*80)
print("P0-3: GraphConvShim 動作テスト (USE_GAT_SHIM=1)")
print("="*80)

# Test 1: GraphConvShim 単体テスト
print("\n[TEST 1] GraphConvShim 単体")
try:
    from src.atft_gat_fan.models.components.gat_shim import GraphConvShim

    B, H = 64, 128
    z = torch.randn(B, H, requires_grad=True)

    # Create simple chain graph
    edge_index = torch.stack([torch.arange(B-1), torch.arange(1, B)], dim=0).long()
    edge_attr = torch.randn(edge_index.size(1), 3)

    conv = GraphConvShim(in_dim=H, hidden_dim=H, edge_dim=3, dropout=0.2)
    conv.train()

    z_out = conv(z, edge_index, edge_attr)

    print(f"  Input shape: {z.shape}")
    print(f"  Output shape: {z_out.shape}")
    assert z_out.shape == z.shape, f"Shape mismatch: {z_out.shape} != {z.shape}"

    # Gradient check
    loss = z_out.mean()
    loss.backward()
    grad_norm = z.grad.abs().sum().item()
    print(f"  Gradient norm: {grad_norm:.3e}")
    assert grad_norm > 0, "Gradient is zero!"

    print("  ✅ GraphConvShim test passed")

except Exception as e:
    print(f"  ❌ GraphConvShim test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: GATBlock (Shim mode)
print("\n[TEST 2] GATBlock (Shim mode)")
try:
    from src.atft_gat_fan.models.components.gat_fuse import GATBlock

    B, H = 64, 128
    z = torch.randn(B, H, requires_grad=True)

    edge_index = torch.stack([torch.arange(B-1), torch.arange(1, B)], dim=0).long()
    edge_attr = torch.randn(edge_index.size(1), 3)

    gat_block = GATBlock(
        in_dim=H,
        hidden_dim=H,
        heads=(4, 2),  # Ignored in shim mode
        edge_dim=3,
        dropout=0.2
    )
    gat_block.train()

    print(f"  GATBlock mode: {gat_block.mode}")
    assert gat_block.mode == "shim", f"Expected 'shim' mode, got '{gat_block.mode}'"

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

    print("  ✅ GATBlock (shim) test passed")

except Exception as e:
    print(f"  ❌ GATBlock (shim) test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: GatedCrossSectionFusion
print("\n[TEST 3] GatedCrossSectionFusion")
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
    print(f"  Gate stats: mean={gate.mean().item():.3f}, std={gate.std().item():.3f}, "
          f"min={gate.min().item():.3f}, max={gate.max().item():.3f}")

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
    print(f"  Gradient ratio: {grad_base / max(grad_gat, 1e-10):.2f}")

    assert grad_base > 0, "Base gradient is zero!"
    assert grad_gat > 0, "GAT gradient is zero!"

    print("  ✅ GatedCrossSectionFusion test passed")

except Exception as e:
    print(f"  ❌ GatedCrossSectionFusion test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Full integration (GATBlock + Fusion)
print("\n[TEST 4] Full integration (GATBlock + Fusion)")
try:
    from src.atft_gat_fan.models.components.gat_fuse import GATBlock, GatedCrossSectionFusion

    B, H = 64, 128
    z_base = torch.randn(B, H, requires_grad=True)

    edge_index = torch.stack([torch.arange(B-1), torch.arange(1, B)], dim=0).long()
    edge_attr = torch.randn(edge_index.size(1), 3)

    gat = GATBlock(in_dim=H, hidden_dim=H, edge_dim=3, dropout=0.2)
    fusion = GatedCrossSectionFusion(hidden=H, tau=1.25, init_bias=-0.5)

    gat.train()

    # GAT forward
    z_gat = gat(z_base, edge_index, edge_attr)

    # Fusion
    z_fused, gate = fusion(z_base, z_gat)

    print(f"  Base shape: {z_base.shape}")
    print(f"  GAT output shape: {z_gat.shape}")
    print(f"  Fused shape: {z_fused.shape}")
    print(f"  Gate mean: {gate.mean().item():.3f}")

    # Gradient check
    loss = z_fused.mean()
    loss.backward()

    total_grad = z_base.grad.abs().sum().item()
    gat_grad = sum(p.grad.abs().sum().item() for p in gat.parameters() if p.grad is not None)
    fusion_grad = sum(p.grad.abs().sum().item() for p in fusion.parameters() if p.grad is not None)

    print(f"  Total gradient: {total_grad:.3e}")
    print(f"  GAT gradient: {gat_grad:.3e}")
    print(f"  Fusion gradient: {fusion_grad:.3e}")

    assert total_grad > 0, "Total gradient is zero!"
    assert gat_grad > 0, "GAT gradient is zero!"
    assert fusion_grad > 0, "Fusion gradient is zero!"

    print("  ✅ Full integration test passed")

except Exception as e:
    print(f"  ❌ Full integration test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Success summary
print("\n" + "="*80)
print("✅ ALL SHIM MODE TESTS PASSED")
print("="*80)
print("Components validated (USE_GAT_SHIM=1):")
print("  1. GraphConvShim: Forward + backward ✅")
print("  2. GATBlock (shim mode): Forward + backward ✅")
print("  3. GatedCrossSectionFusion: Gating + gradients ✅")
print("  4. Full integration: GAT + Fusion ✅")
print("\nP0-3 shim mode is working correctly!")
print("Ready for training: USE_GAT_SHIM=1 make train-quick EPOCHS=3")
print("="*80)
