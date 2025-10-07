#!/usr/bin/env python
"""GAT設定の読み込みを診断するスクリプト"""

import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@hydra.main(version_base=None, config_path="../configs/atft", config_name="config_production_optimized")
def main(cfg: DictConfig):
    """Hydra設定を読み込んでGAT関連の設定を確認"""

    print("=" * 80)
    print("GAT Configuration Diagnostic")
    print("=" * 80)

    # モデル設定全体を表示
    print("\n[1] Full model config:")
    print(OmegaConf.to_yaml(cfg.model))

    # GAT設定を抽出
    gat_cfg = getattr(cfg.model, "gat", None)
    print(f"\n[2] gat_cfg is None: {gat_cfg is None}")

    if gat_cfg is not None:
        print("\n[3] GAT enabled:", getattr(gat_cfg, "enabled", False))

        # regularization設定を確認
        reg_cfg = getattr(gat_cfg, "regularization", {})
        print(f"\n[4] regularization config type: {type(reg_cfg)}")
        print(f"[4] regularization config: {reg_cfg}")

        # 個別の値を抽出
        edge_penalty = float(getattr(reg_cfg, "edge_weight_penalty", 0.0))
        entropy_penalty = float(getattr(reg_cfg, "attention_entropy_penalty", 0.0))

        print(f"\n[5] Extracted values:")
        print(f"  edge_weight_penalty: {edge_penalty}")
        print(f"  attention_entropy_penalty: {entropy_penalty}")

        # モデル初期化と同じロジックでテスト
        print(f"\n[6] Test with model initialization logic:")
        print(f"  self.gat_edge_weight would be: {edge_penalty}")
        print(f"  self.gat_entropy_weight would be: {entropy_penalty}")
        print(f"  return_attention condition (gat_entropy_weight > 0): {entropy_penalty > 0}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
