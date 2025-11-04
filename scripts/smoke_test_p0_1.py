#!/usr/bin/env python3
"""
P0-1 スモークテスト: StableFANSAN の動作確認

目的:
1. NaN/Infが発生しないか
2. 勾配が非ゼロで流れているか
3. FAN/SANのbackwardが正しく通るか
4. 形状が [B,T,H] で正しく適用されているか
"""
import os
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

# 環境変数設定（デバッグモード無効化）
os.environ["BYPASS_ADAPTIVE_NORM"] = "0"
os.environ["BYPASS_GAT_COMPLETELY"] = "0"
os.environ["ENABLE_ENCODER_DIAGNOSTICS"] = "0"
os.environ["DEBUG_PREDICTIONS"] = "0"

# ランダムシード固定
torch.manual_seed(0)

print("=" * 80)
print("P0-1 スモークテスト: StableFANSAN")
print("=" * 80)

# テストパラメータ
B, T, F, H = 8, 20, 306, 128

print("\nテストパラメータ:")
print(f"  Batch size (B): {B}")
print(f"  Sequence length (T): {T}")
print(f"  Feature dim (F): {F}")
print(f"  Hidden dim (H): {H}")

# StableFANSAN単体テスト
print("\n" + "-" * 80)
print("1. StableFANSAN単体テスト")
print("-" * 80)

from src.atft_gat_fan.models.components.adaptive_normalization import StableFANSAN

# モデル初期化
model = StableFANSAN(
    feat_dim=H,
    windows=(5, 10, 20),
    num_slices=3,
    overlap=0.5,
    eps=1e-4,
    entropy_coef=1e-4
)
model.train()

print("\n✓ StableFANSAN初期化成功")
print(f"  - FAN windows: {model.fan.windows}")
print(f"  - SAN slices: {model.san.K}")
print(f"  - Entropy coef: {model.entropy_coef}")

# 入力テンソル（TFT入力を想定した [B, T, H]）
projected = torch.randn(B, T, H) * 0.3
projected.requires_grad_(True)

print("\n入力テンソル:")
print(f"  - Shape: {projected.shape}")
print(f"  - Mean: {projected.mean().item():.6f}")
print(f"  - Std: {projected.std().item():.6f}")
print(f"  - Min: {projected.min().item():.6f}")
print(f"  - Max: {projected.max().item():.6f}")

# Forward pass
proj_normed = model(projected)

print("\n出力テンソル:")
print(f"  - Shape: {proj_normed.shape}")
print(f"  - Mean: {proj_normed.mean().item():.6f}")
print(f"  - Std: {proj_normed.std().item():.6f}")
print(f"  - Min: {proj_normed.min().item():.6f}")
print(f"  - Max: {proj_normed.max().item():.6f}")

# 形状チェック
assert proj_normed.shape == (B, T, H), f"形状エラー: 期待 ({B}, {T}, {H}), 実際 {proj_normed.shape}"
print("\n✓ 形状チェック: PASS")

# NaN/Inf チェック
assert torch.isfinite(proj_normed).all(), "NaN/Inf検出: FAIL"
print("✓ NaN/Infチェック: PASS")

# Backward pass
head = torch.nn.Linear(H, 5)
out = head(proj_normed[:, -1, :])  # [B, 5]

# 正則化項も含める
reg = model.regularizer()
print("\n正則化項:")
print(f"  - Entropy reg: {reg.item():.6e}")
print(f"  - FAN entropy: {model.fan._entropy.item():.6f}")

loss = out.mean() + reg
loss.backward()

print("\nLoss backward完了:")
print(f"  - Loss: {loss.item():.6f}")

# 勾配チェック
g = projected.grad
assert g is not None, "勾配がNone: FAIL"
assert torch.isfinite(g).all(), "勾配にNaN/Inf: FAIL"
assert g.abs().mean() > 0, "勾配がゼロ: FAIL"

print("\n勾配統計:")
print(f"  - Mean: {g.mean().item():.6e}")
print(f"  - Std: {g.std().item():.6e}")
print(f"  - Norm: {g.norm().item():.6e}")
print(f"  - Max: {g.abs().max().item():.6e}")

print("\n✓ 勾配フローチェック: PASS")

# パラメータ勾配チェック
fan_alpha_grad = model.fan.alpha.grad
san_gamma_grad = model.san.gamma.grad

assert fan_alpha_grad is not None and torch.isfinite(fan_alpha_grad).all(), "FAN alpha勾配異常"
assert san_gamma_grad is not None and torch.isfinite(san_gamma_grad).all(), "SAN gamma勾配異常"

print("\nパラメータ勾配:")
print(f"  - FAN alpha grad norm: {fan_alpha_grad.norm().item():.6e}")
print(f"  - SAN gamma grad norm: {san_gamma_grad.norm().item():.6e}")
print(f"  - SAN beta grad norm: {model.san.beta.grad.norm().item():.6e}")

print("\n✓ パラメータ勾配チェック: PASS")

# 短いシーケンスでの降格テスト（T < max(windows)）
print("\n" + "-" * 80)
print("2. 短いシーケンステスト (T=3 < windows)")
print("-" * 80)

short_input = torch.randn(B, 3, H) * 0.3  # T=3
short_input.requires_grad_(True)
short_output = model(short_input)

assert short_output.shape == (B, 3, H), "短シーケンス形状エラー"
assert torch.isfinite(short_output).all(), "短シーケンスでNaN/Inf"

print(f"  - Input shape: {short_input.shape}")
print(f"  - Output shape: {short_output.shape}")
print(f"  - Output mean: {short_output.mean().item():.6f}")
print(f"  - Output std: {short_output.std().item():.6f}")

print("\n✓ 短いシーケンステスト: PASS")

# 最終結果
print("\n" + "=" * 80)
print("✅ P0-1 スモークテスト: 全PASS")
print("=" * 80)
print("\n結果サマリー:")
print("  ✓ StableFANSAN初期化")
print("  ✓ Forward pass (形状、NaN/Infなし)")
print("  ✓ Backward pass (勾配フロー確認)")
print("  ✓ 正則化項 (エントロピー)")
print("  ✓ パラメータ勾配 (FAN alpha, SAN gamma/beta)")
print("  ✓ 短いシーケンス対応 (降格動作)")
print("\nP0-1実装: 成功 ✅")
print("\n次のステップ:")
print("  1. モデル本体への統合 (atft_gat_fan.py)")
print("  2. P0-5: DataLoader安定化")
print("  3. 小規模訓練テスト (1 epoch)")
print("=" * 80)

sys.exit(0)
