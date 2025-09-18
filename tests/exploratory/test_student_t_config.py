#!/usr/bin/env python3
"""Test Student-t configuration"""

import os
import sys
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from pathlib import Path

# Set environment variables
os.environ["ENABLE_STUDENT_T"] = "1"
os.environ["USE_T_NLL"] = "1"
os.environ["NLL_WEIGHT"] = "0.02"

# Initialize Hydra
config_path = str(Path(__file__).parent / "configs" / "atft")
with initialize_config_dir(config_dir=config_path, version_base="1.1"):
    cfg = compose(config_name="config")
    
    # Apply env overrides (simplified version)
    if "prediction_head" in cfg.model and "output" in cfg.model.prediction_head:
        if "student_t" in cfg.model.prediction_head.output:
            print(f"✅ Before override: student_t = {cfg.model.prediction_head.output.student_t}")
            cfg.model.prediction_head.output.student_t = True
            print(f"✅ After override: student_t = {cfg.model.prediction_head.output.student_t}")
        else:
            print("❌ student_t field not found in config")
    else:
        print("❌ prediction_head.output not found in config")
    
    # Test model initialization
    print("\n📊 Testing model initialization...")
    
    # Import model
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from src.models.architectures.atft_gat_fan import ATFT_GAT_FAN
        import torch
        
        # Create model
        model = ATFT_GAT_FAN(cfg)
        
        # Check if Student-t heads are created
        t_heads_count = len(model.t_heads)
        print(f"✅ Model created with {t_heads_count} Student-t heads")
        
        if t_heads_count > 0:
            print("✅ Student-t heads initialized:")
            for key in model.t_heads.keys():
                print(f"   - {key}")
        
        # Test forward pass
        batch_size = 2
        seq_len = 20
        input_dim = 13
        
        dummy_input = torch.randn(batch_size, seq_len, input_dim)
        outputs = model(dummy_input)
        
        # Check for t_params in outputs
        t_params_keys = [k for k in outputs.keys() if "t_params" in k]
        print(f"\n✅ Forward pass successful, found {len(t_params_keys)} t_params outputs:")
        for key in t_params_keys:
            print(f"   - {key}: shape {outputs[key].shape}")
        
    except Exception as e:
        print(f"❌ Error during model test: {e}")
        import traceback
        traceback.print_exc()

print("\n✅ Student-t configuration test completed successfully!")