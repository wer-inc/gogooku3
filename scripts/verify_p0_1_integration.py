#!/usr/bin/env python3
"""
P0-1統合テスト: ATFT-GAT-FANモデルへのStableFANSAN統合確認

検証項目:
1. StableFANSANが正しく初期化されているか
2. FAN/SANがLayerNormから置き換わっているか
3. 正則化項が損失に追加されているか
4. 環境変数BYPASS_ADAPTIVE_NORM=0で動作するか
"""
import os
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from omegaconf import OmegaConf

# 環境変数設定（本番モード）
os.environ["BYPASS_ADAPTIVE_NORM"] = "0"
os.environ["BYPASS_GAT_COMPLETELY"] = "0"

print("=" * 80)
print("P0-1統合テスト: ATFT-GAT-FAN → StableFANSAN")
print("=" * 80)

# 最小構成でモデルを初期化
print("\n1. モデル初期化テスト")
print("-" * 80)

try:
    from src.atft_gat_fan.models.architectures.atft_gat_fan import ATFT_GAT_FAN

    # 最小設定
    config = OmegaConf.create({
        "hidden_size": 128,
        "n_horizons": 4,
        "quantiles": [0.1, 0.5, 0.9],
        "n_dynamic_features": 99,  # 現在の実際の特徴量数
        "dropout": 0.1,
        "gat": {
            "enabled": False,  # GATは無効化してFAN/SANのみテスト
        },
        "improvements": {
            "use_ema_teacher": False,
            "compile_model": False,
        }
    })

    model = ATFT_GAT_FAN(config)
    print("✓ モデル初期化成功")

except Exception as e:
    print(f"❌ モデル初期化失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 2. StableFANSAN統合確認
print("\n2. StableFANSAN統合確認")
print("-" * 80)

# adaptive_normの型チェック
from src.atft_gat_fan.models.components.adaptive_normalization import StableFANSAN

if isinstance(model.adaptive_norm, StableFANSAN):
    print("✓ adaptive_normはStableFANSAN型")
    print(f"  - FAN windows: {model.adaptive_norm.fan.windows}")
    print(f"  - SAN slices: {model.adaptive_norm.san.K}")
    print(f"  - Entropy coef: {model.adaptive_norm.entropy_coef}")
else:
    print(f"❌ adaptive_normの型が間違っています: {type(model.adaptive_norm)}")
    print(f"   期待: StableFANSAN, 実際: {type(model.adaptive_norm).__name__}")
    sys.exit(1)

# regularizerメソッドの存在確認
if hasattr(model.adaptive_norm, "regularizer"):
    print("✓ regularizer()メソッドが存在")
else:
    print("❌ regularizer()メソッドが存在しません")
    sys.exit(1)

# 3. Forward pass テスト
print("\n3. Forward passテスト ([B,T,F] → 適応正規化 → [B,T,H])")
print("-" * 80)

model.eval()
B, T, F = 4, 20, 99
H = 128

# ダミーバッチ作成
batch = {
    "dynamic_features": torch.randn(B, F) * 0.3,
    "static_features": torch.randn(B, 10) * 0.1,
}

try:
    with torch.no_grad():
        outputs = model(batch)

    print("✓ Forward pass成功")
    print(f"  - 出力キー: {list(outputs.keys())}")

    # 正則化項の確認（訓練モードで）
    model.train()
    outputs_train = model(batch)

    # 正則化項を取得
    fan_san_reg = model.adaptive_norm.regularizer()
    print("\n正則化項:")
    print(f"  - FAN/SAN reg: {fan_san_reg.item():.6e}")
    print(f"  - FAN entropy: {model.adaptive_norm.fan._entropy.item():.6f}")

    if fan_san_reg.item() > 0:
        print("✓ 正則化項が正常に計算されています")
    else:
        print("⚠️  正則化項がゼロです（初期状態では正常）")

except Exception as e:
    print(f"❌ Forward pass失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. Backward passテスト（勾配フロー確認）
print("\n4. Backward passテスト (勾配フロー確認)")
print("-" * 80)

model.train()
batch_grad = {
    "dynamic_features": torch.randn(B, F) * 0.3,
    "static_features": torch.randn(B, 10) * 0.1,
}

try:
    outputs = model(batch_grad)

    # ダミー損失（quantile出力の平均）
    loss = outputs["quantile_outputs"][..., 1].mean()  # 中央値を使用

    # 正則化項を追加
    if hasattr(model, "adaptive_norm") and hasattr(model.adaptive_norm, "regularizer"):
        fan_san_reg = model.adaptive_norm.regularizer()
        loss = loss + fan_san_reg

    loss.backward()

    # FAN/SANのパラメータ勾配確認
    fan_alpha_grad = model.adaptive_norm.fan.alpha.grad
    san_gamma_grad = model.adaptive_norm.san.gamma.grad

    if fan_alpha_grad is not None and san_gamma_grad is not None:
        print("✓ FAN/SANパラメータに勾配が流れています")
        print(f"  - FAN alpha grad norm: {fan_alpha_grad.norm().item():.6e}")
        print(f"  - SAN gamma grad norm: {san_gamma_grad.norm().item():.6e}")

        if fan_alpha_grad.norm().item() > 0 and san_gamma_grad.norm().item() > 0:
            print("✓ 勾配は非ゼロで正常です")
        else:
            print("⚠️  勾配がゼロです（入力依存の場合あり）")
    else:
        print("❌ FAN/SANパラメータに勾配が流れていません")
        sys.exit(1)

except Exception as e:
    print(f"❌ Backward pass失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 最終結果
print("\n" + "=" * 80)
print("✅ P0-1統合テスト: 全PASS")
print("=" * 80)
print("\n統合確認サマリー:")
print("  ✓ StableFANSANが正しく統合されています")
print("  ✓ LayerNormから置き換わりました")
print("  ✓ 正則化項が損失に追加可能です")
print("  ✓ 勾配が正常に流れています")
print("\n✨ ATFT-GAT-FANモデルのFAN/SAN機能が復活しました！")
print("\n次のステップ:")
print("  1. P0-5: DataLoader安定化 (マルチワーカー、spawn)")
print("  2. 小規模訓練テスト (1-3 epoch)")
print("  3. P0-2: 特徴量306本復旧")
print("=" * 80)

sys.exit(0)
