"""
Simple GAT gradient flow test
Test if gradients flow through GAT layer with simple MSE loss
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omegaconf import OmegaConf

from atft_gat_fan.models.architectures.atft_gat_fan import ATFT_GAT_FAN


def test_gat_gradient_flow():
    """Test if gradients flow to GAT layers with simple loss"""

    # Load Phase 1 config
    config_path = Path(__file__).parent.parent / "configs" / "atft" / "train" / "phase1_gat.yaml"
    cfg = OmegaConf.load(config_path)

    # Create model
    print("Creating model...")
    model = ATFT_GAT_FAN(cfg.model)
    model.train()

    # Create dummy data
    batch_size = 4
    seq_len = 20
    n_features = 82
    n_stocks = batch_size

    # Create features
    features = torch.randn(batch_size, seq_len, n_features)

    # Create simple edge_index (fully connected)
    edge_index = torch.stack([
        torch.arange(n_stocks).repeat_interleave(n_stocks),
        torch.arange(n_stocks).repeat(n_stocks)
    ], dim=0)

    # Create edge_attr
    edge_attr = torch.randn(edge_index.size(1), 3)

    # Create targets
    targets = {
        "horizon_1d": torch.randn(batch_size, 20),  # 20 quantiles
        "horizon_5d": torch.randn(batch_size, 20),
        "horizon_10d": torch.randn(batch_size, 20),
        "horizon_20d": torch.randn(batch_size, 20),
    }

    print(f"Input shapes: features={features.shape}, edge_index={edge_index.shape}")

    # Forward pass
    print("Running forward pass...")
    output = model(
        features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        return_attention=False  # Simpler
    )

    print(f"Output keys: {output.keys()}")
    predictions = output["predictions"]
    print(f"Predictions keys: {predictions.keys()}")

    # Simple MSE loss (instead of complex MultiHorizonLoss)
    print("\nComputing simple MSE loss...")
    loss = 0.0
    for horizon in ["horizon_1d", "horizon_5d", "horizon_10d", "horizon_20d"]:
        pred = predictions[horizon]
        target = targets[horizon]
        horizon_loss = nn.functional.mse_loss(pred, target)
        loss += horizon_loss
        print(f"  {horizon}: loss={horizon_loss.item():.6f}, pred_shape={pred.shape}")

    print(f"\nTotal loss: {loss.item():.6f}")

    # Backward pass
    print("\nRunning backward pass...")
    loss.backward()

    # Check gradients
    print("\n=== Gradient Check ===")

    # Check GAT layer gradients
    if hasattr(model, "gat") and model.gat is not None:
        print("\nGAT layer gradients:")
        gat_has_grad = False
        for name, param in model.gat.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"  {name}: grad_norm={grad_norm:.6e}")
                if grad_norm > 0:
                    gat_has_grad = True
            else:
                print(f"  {name}: grad is None")

        if gat_has_grad:
            print("\n✅ GAT layer HAS gradients!")
        else:
            print("\n❌ GAT layer has NO gradients!")

    # Check backbone_projection gradients
    print("\nBackbone projection gradients:")
    for name, param in model.backbone_projection.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"  {name}: grad_norm={grad_norm:.6e}")
        else:
            print(f"  {name}: grad is None")

    # Check gat_residual_gate gradient
    if hasattr(model, "gat_residual_gate"):
        if model.gat_residual_gate.grad is not None:
            gate_grad = model.gat_residual_gate.grad.item()
            print(f"\ngat_residual_gate: grad={gate_grad:.6e}")
        else:
            print("\ngat_residual_gate: grad is None")

    # Check TFT gradients
    print("\nTFT layer gradients (first layer as sample):")
    tft_params = list(model.tft.parameters())
    if tft_params:
        first_param = tft_params[0]
        if first_param.grad is not None:
            print(f"  TFT first param: grad_norm={first_param.grad.norm().item():.6e}")
        else:
            print("  TFT first param: grad is None")

if __name__ == "__main__":
    print("="*80)
    print("GAT Gradient Flow Simple Test")
    print("="*80)
    test_gat_gradient_flow()
