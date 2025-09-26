#!/usr/bin/env python3
"""Test data loading to diagnose zero loss issue"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append('/home/ubuntu/gogooku3-standalone')

# Simple test of data loading
from src.gogooku3.training.atft.data_module import ProductionDataModuleV2
from omegaconf import OmegaConf

# Create minimal config
config = OmegaConf.create({
    'data': {
        'source': {
            'data_dir': '/home/ubuntu/gogooku3-standalone/output/atft_data'
        },
        'time_series': {
            'sequence_length': 20,
            'prediction_horizons': [1, 5, 10, 20]
        }
    },
    'train': {
        'batch': {
            'batch_size': 32
        }
    }
})

print("Testing data loading...")
data_module = ProductionDataModuleV2(config, batch_size=32, num_workers=0)
data_module.setup()

train_loader = data_module.train_dataloader()

# Check first batch
for i, batch in enumerate(train_loader):
    print(f"\nBatch {i}:")
    print(f"  Keys: {batch.keys()}")
    
    if 'features' in batch:
        print(f"  Features shape: {batch['features'].shape}")
    
    if 'targets' in batch:
        targets = batch['targets']
        print(f"  Targets type: {type(targets)}")
        if isinstance(targets, dict):
            for k, v in targets.items():
                if torch.is_tensor(v):
                    nonzero = (v != 0).sum().item()
                    total = v.numel()
                    print(f"    {k}: shape={v.shape}, nonzero={nonzero}/{total} ({100*nonzero/total:.1f}%)")
                    print(f"      mean={v.mean():.6f}, std={v.std():.6f}")
                    print(f"      sample values: {v.flatten()[:5].tolist()}")
        else:
            print(f"  Targets shape: {targets.shape if hasattr(targets, 'shape') else 'N/A'}")
    
    if i >= 2:  # Check first 3 batches
        break

print("\n✅ Data loading test complete")
