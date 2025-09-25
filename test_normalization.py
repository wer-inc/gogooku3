#!/usr/bin/env python
"""Test script to verify feature normalization is working correctly."""

import torch
import numpy as np
import os

# Set environment variables for testing
os.environ["ENABLE_FEATURE_NORM"] = "1"
os.environ["FEATURE_CLIP_VALUE"] = "10.0"

def test_normalization():
    """Test feature normalization logic."""

    # Create test data with different scales (similar to real data)
    batch_size = 32
    seq_len = 60
    n_features = 189

    # Create features with very different scales
    features = torch.randn(batch_size, seq_len, n_features)

    # Add different scales to different features (simulate real data)
    features[:, :, 0:10] *= 1000.0  # Price-like features
    features[:, :, 10:20] *= 0.01   # Return-like features
    features[:, :, 20:30] *= 10000.0  # Volume-like features

    print("Before normalization:")
    print(f"  Feature means (first 30): {features.mean(dim=(0,1))[:30]}")
    print(f"  Feature stds (first 30): {features.std(dim=(0,1))[:30]}")
    print(f"  Overall mean: {features.mean():.4f}, std: {features.std():.4f}")

    # Apply normalization (same logic as in train_atft.py)
    with torch.no_grad():
        # Compute batch statistics
        batch_mean = features.mean(dim=(0, 1), keepdim=True)
        batch_std = features.std(dim=(0, 1), keepdim=True)

        # Avoid division by zero
        batch_std = torch.clamp(batch_std, min=1e-6)

        # Z-score normalization
        features_norm = (features - batch_mean) / batch_std

        # Clip extreme values
        clip_value = float(os.environ.get("FEATURE_CLIP_VALUE", "10.0"))
        if clip_value > 0:
            features_norm = torch.clamp(features_norm, min=-clip_value, max=clip_value)

    print("\nAfter normalization:")
    print(f"  Feature means (first 30): {features_norm.mean(dim=(0,1))[:30]}")
    print(f"  Feature stds (first 30): {features_norm.std(dim=(0,1))[:30]}")
    print(f"  Overall mean: {features_norm.mean():.4f}, std: {features_norm.std():.4f}")
    print(f"  Min: {features_norm.min():.4f}, Max: {features_norm.max():.4f}")

    # Verify normalization worked
    assert abs(features_norm.mean()) < 0.1, "Mean should be close to 0"
    assert 0.8 < features_norm.std() < 1.2, "Std should be close to 1"
    assert features_norm.min() >= -clip_value, f"Min should be >= -{clip_value}"
    assert features_norm.max() <= clip_value, f"Max should be <= {clip_value}"

    print("\n✅ Normalization test passed!")
    return True

if __name__ == "__main__":
    test_normalization()