#!/usr/bin/env python
"""Test environment settings for optimized training."""

import os

# Test all environment variables
env_vars = {
    "ENABLE_FEATURE_NORM": "1",
    "FEATURE_CLIP_VALUE": "10.0",
    "USE_RANKIC": "1",
    "RANKIC_WEIGHT": "0.2",
    "USE_HUBER": "1",
    "HUBER_WEIGHT": "0.3",
    "USE_CS_IC": "1",
    "CS_IC_WEIGHT": "0.15",
    "USE_DIR_AUX": "1",
    "DIR_AUX_WEIGHT": "0.1",
    "SHARPE_WEIGHT": "0.3",
    "DYN_WEIGHT": "1",
    "ALLOW_UNSAFE_DATALOADER": "1",
    "NUM_WORKERS": "8",
    "PERSISTENT_WORKERS": "1",
    "PREFETCH_FACTOR": "4",
}

print("Environment Variable Settings Check:")
print("=" * 60)

for var, expected in env_vars.items():
    actual = os.environ.get(var, "NOT SET")
    status = "✅" if actual == expected else "❌"
    print(f"{status} {var}: {actual} (expected: {expected})")

# Check phase loss weights separately (it's complex)
phase_weights = os.environ.get("PHASE_LOSS_WEIGHTS", "NOT SET")
if "huber" in phase_weights and "quantile" in phase_weights and "sharpe" in phase_weights:
    print(f"✅ PHASE_LOSS_WEIGHTS: Set correctly (contains huber, quantile, sharpe)")
else:
    print(f"❌ PHASE_LOSS_WEIGHTS: {phase_weights[:50]}...")

print("=" * 60)
print("\nTo test these settings, they would be set by the Makefile command:")
print("make train-optimized-stable")
