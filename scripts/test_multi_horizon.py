#!/usr/bin/env python3
"""
Multi-horizon output architecture test script.
Tests the new horizon-specific prediction heads implementation.
"""

import torch
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omegaconf import DictConfig
from atft_gat_fan.models.architectures.atft_gat_fan import MultiHorizonPredictionHeads


def test_multi_horizon_architecture():
    """Test the multi-horizon prediction heads architecture."""
    print("🧪 Testing Multi-horizon Output Architecture")

    # Create mock configuration
    config = DictConfig({
        'training': {
            'prediction': {
                'horizons': [1, 5, 10, 20],
                'horizon_weights': [2.0, 1.5, 1.0, 0.7]
            }
        },
        'architecture': {
            'dropout': 0.1
        },
        'output': {
            'quantile_prediction': {
                'quantiles': [0.1, 0.5, 0.9],
                'enable': True
            }
        }
    })

    # Test parameters
    batch_size = 64
    hidden_size = 256

    # Create multi-horizon prediction heads
    print(f"📊 Creating prediction heads for horizons: {config.training.prediction.horizons}")
    prediction_heads = MultiHorizonPredictionHeads(
        hidden_size=hidden_size,
        config=config
    )

    # Create mock input
    x = torch.randn(batch_size, hidden_size)
    print(f"🔢 Input shape: {x.shape}")

    # Test forward pass
    print("🚀 Testing forward pass...")
    with torch.no_grad():
        predictions = prediction_heads(x)

    print(f"✅ Multi-horizon predictions generated:")
    for horizon_key, pred in predictions.items():
        print(f"  - {horizon_key}: {pred.shape} (mean: {pred.mean():.4f}, std: {pred.std():.4f})")

    # Verify outputs
    expected_horizons = [f'horizon_{h}d' for h in config.training.prediction.horizons]
    actual_horizons = list(predictions.keys())

    assert set(actual_horizons) == set(expected_horizons), f"Expected {expected_horizons}, got {actual_horizons}"

    # Check output shapes (should be [batch_size, n_quantiles])
    expected_n_quantiles = len(config.output.quantile_prediction.quantiles)
    for pred in predictions.values():
        assert pred.shape == (batch_size, expected_n_quantiles), f"Expected shape ({batch_size}, {expected_n_quantiles}), got {pred.shape}"

    print("✅ All tests passed!")

    # Test architecture details
    print(f"\n📋 Architecture Details:")
    print(f"  - Number of horizons: {len(config.training.prediction.horizons)}")
    print(f"  - Horizon weights: {config.training.prediction.horizon_weights}")
    print(f"  - Model parameters: {sum(p.numel() for p in prediction_heads.parameters()):,}")

    # Show parameter distribution by horizon
    print(f"\n🔍 Parameter distribution by horizon:")
    for horizon in config.training.prediction.horizons:
        horizon_key = f'horizon_{horizon}d'
        if horizon_key in prediction_heads.horizon_heads:
            horizon_head = prediction_heads.horizon_heads[horizon_key]
            params = sum(p.numel() for p in horizon_head.parameters())
            print(f"  - {horizon}d horizon: {params:,} parameters")


if __name__ == "__main__":
    test_multi_horizon_architecture()