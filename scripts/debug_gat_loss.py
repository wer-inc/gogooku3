#!/usr/bin/env python
"""GAT Loss計算の診断スクリプト - モデル初期化まで実行"""

import sys
from pathlib import Path
import torch
import hydra
from omegaconf import DictConfig

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

@hydra.main(version_base=None, config_path="../configs/atft", config_name="config_production_optimized")
def main(cfg: DictConfig):
    """モデルを初期化してGAT loss計算をテスト"""

    print("=" * 80)
    print("GAT Loss Calculation Diagnostic")
    print("=" * 80)

    # Import model
    try:
        from atft_gat_fan.models.architectures.atft_gat_fan import ATFT_GAT_FAN
        print("\n✅ ATFT_GAT_FAN imported successfully")
    except Exception as e:
        print(f"\n❌ Failed to import ATFT_GAT_FAN: {e}")
        return

    # Initialize model
    print("\n[1] Initializing model...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ATFT_GAT_FAN(cfg).to(device)
        print(f"✅ Model initialized on {device}")
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Check GAT configuration
    print("\n[2] Checking GAT configuration:")
    print(f"  model.gat is None: {model.gat is None}")
    print(f"  model.gat_entropy_weight: {getattr(model, 'gat_entropy_weight', 'N/A')}")
    print(f"  model.gat_edge_weight: {getattr(model, 'gat_edge_weight', 'N/A')}")
    print(f"  model.gat_output_dim: {getattr(model, 'gat_output_dim', 'N/A')}")

    # Create dummy input
    print("\n[3] Creating dummy input...")
    batch_size = 32
    seq_len = 20
    features = 200

    dummy_input = {
        "features": torch.randn(batch_size, seq_len, features, device=device),
        "edge_index": torch.randint(0, batch_size, (2, 100), device=device),
        "edge_attr": torch.randn(100, 3, device=device),
    }
    print(f"  features: {dummy_input['features'].shape}")
    print(f"  edge_index: {dummy_input['edge_index'].shape}")
    print(f"  edge_attr: {dummy_input['edge_attr'].shape}")

    # Forward pass
    print("\n[4] Running forward pass...")
    model.train()
    try:
        # ATFT_GAT_FAN.forward() takes a batch dict
        outputs = model(dummy_input)
        print(f"✅ Forward pass successful")
        print(f"  Output keys: {list(outputs.keys())}")
        for key, val in outputs.items():
            if torch.is_tensor(val):
                print(f"  {key}: shape={val.shape}, requires_grad={val.requires_grad}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Check GAT internal states
    print("\n[5] Checking GAT internal states:")
    print(f"  _gat_attention_entropy: {model._gat_attention_entropy}")
    print(f"    type: {type(model._gat_attention_entropy)}")
    if torch.is_tensor(model._gat_attention_entropy):
        print(f"    requires_grad: {model._gat_attention_entropy.requires_grad}")
        print(f"    grad_fn: {model._gat_attention_entropy.grad_fn}")
    print(f"  _gat_edge_reg_value: {model._gat_edge_reg_value}")
    print(f"    type: {type(model._gat_edge_reg_value)}")
    if torch.is_tensor(model._gat_edge_reg_value):
        print(f"    requires_grad: {model._gat_edge_reg_value.requires_grad}")
        print(f"    grad_fn: {model._gat_edge_reg_value.grad_fn}")

    # Check loss calculation conditions
    print("\n[6] Checking loss calculation conditions:")

    # Edge regularization condition
    edge_cond1 = model._gat_edge_reg_value is not None
    edge_cond2 = isinstance(model._gat_edge_reg_value, torch.Tensor) if edge_cond1 else False
    edge_cond3 = model.gat_edge_weight > 0
    print(f"  Edge regularization:")
    print(f"    - _gat_edge_reg_value is not None: {edge_cond1}")
    print(f"    - isinstance(Tensor): {edge_cond2}")
    print(f"    - gat_edge_weight > 0: {edge_cond3}")
    print(f"    - ALL CONDITIONS MET: {edge_cond1 and edge_cond2 and edge_cond3}")

    # Entropy regularization condition
    entropy_cond1 = model._gat_attention_entropy is not None
    entropy_cond2 = isinstance(model._gat_attention_entropy, torch.Tensor) if entropy_cond1 else False
    entropy_cond3 = model.gat_entropy_weight > 0
    print(f"  Entropy regularization:")
    print(f"    - _gat_attention_entropy is not None: {entropy_cond1}")
    print(f"    - isinstance(Tensor): {entropy_cond2}")
    print(f"    - gat_entropy_weight > 0: {entropy_cond3}")
    print(f"    - ALL CONDITIONS MET: {entropy_cond1 and entropy_cond2 and entropy_cond3}")

    # Calculate total loss if possible
    print("\n[7] Simulating loss calculation...")
    if edge_cond1 and edge_cond2 and edge_cond3:
        edge_reg = model.gat_edge_weight * model._gat_edge_reg_value
        print(f"  ✅ Edge regularization would be: {edge_reg.item():.6f}")
    else:
        print(f"  ❌ Edge regularization NOT calculated (conditions not met)")

    if entropy_cond1 and entropy_cond2 and entropy_cond3:
        entropy_reg = -model.gat_entropy_weight * model._gat_attention_entropy
        print(f"  ✅ Entropy regularization would be: {entropy_reg.item():.6f}")
    else:
        print(f"  ❌ Entropy regularization NOT calculated (conditions not met)")

    print("\n" + "=" * 80)
    print("Diagnostic Complete")
    print("=" * 80)

if __name__ == "__main__":
    main()
