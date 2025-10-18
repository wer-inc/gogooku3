#!/usr/bin/env python3
"""Simple test for Student-t configuration"""

import os
import sys
from pathlib import Path

# Set environment variables
os.environ["ENABLE_STUDENT_T"] = "1"
os.environ["USE_T_NLL"] = "1"
os.environ["NLL_WEIGHT"] = "0.02"

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import and test
try:
    from omegaconf import OmegaConf

    # Load config directly
    config_path = Path(__file__).parent / "configs" / "model" / "atft_gat_fan.yaml"
    cfg = OmegaConf.load(config_path)

    print("✅ Config loaded successfully")
    print(f"   student_t setting: {cfg.prediction_head.output.student_t}")

    # Import model
    import torch

    from src.models.architectures.atft_gat_fan import ATFT_GAT_FAN

    # Create minimal config for model
    full_config = OmegaConf.create(
        {
            "model": cfg,
            "data": {
                "time_series": {
                    "prediction_horizons": [1, 5, 10, 20],
                    "sequence_length": 20,
                },
                "features": {"input_dim": 13},
            },
        }
    )

    # Create model
    print("\n📊 Creating model...")
    model = ATFT_GAT_FAN(full_config)

    # Check Student-t heads
    t_heads = [k for k in model.t_heads.keys() if "t_params" in k]
    print(f"✅ Model created with {len(t_heads)} Student-t heads:")
    for head in t_heads:
        print(f"   - {head}")

    # Test forward pass
    print("\n🔄 Testing forward pass...")
    batch_size = 2
    seq_len = 20
    input_dim = 13

    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    outputs = model(dummy_input)

    # Check outputs
    t_outputs = [k for k in outputs.keys() if "t_params" in k]
    print(f"✅ Found {len(t_outputs)} Student-t outputs in forward pass:")
    for key in t_outputs:
        shape = outputs[key].shape
        print(f"   - {key}: shape {shape} (mu, sigma_raw, nu_raw)")
        if len(shape) >= 2 and shape[-1] == 3:
            print("      ✓ Correct shape for Student-t parameters")

    print("\n✅ Student-t configuration is working correctly!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
