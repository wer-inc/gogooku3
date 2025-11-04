"""
P0-3: GAT Fusion Block with Gated Residual
同次元化 + ゲート付き残差で勾配希釈を防止

フォールバック機能:
- PyG (GATv2Conv) 利用可能 → 高性能GAT実装
- PyG 未インストール or segfault → 安全シム実装
- 環境変数 USE_GAT_SHIM=1 → 強制的にシム使用
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

from .gat_shim import GATBlockShim

# CRITICAL: Check USE_GAT_SHIM *before* attempting PyG import
# PyG import can segfault on PyTorch 2.9.0+cu128, so skip import entirely if shim is forced
if os.getenv("USE_GAT_SHIM", "0") == "1":
    GATv2Conv = None
else:
    try:
        from torch_geometric.nn import GATv2Conv
    except Exception:
        GATv2Conv = None


class GATBlock(nn.Module):
    """
    2層の GATv2Conv（PyG実装）または GATBlockShim（フォールバック）。
    edge_attr を使用。出力は hidden_dim に合わせる。

    モード選択:
    1. GATv2Conv is None → 自動的にShim使用
    2. USE_GAT_SHIM=1 → 強制的にShim使用
    3. それ以外 → PyG実装使用
    """
    def __init__(self, in_dim: int, hidden_dim: int, heads=(4, 2), edge_dim=3, dropout=0.2):
        super().__init__()

        # フォールバック判定
        use_shim = GATv2Conv is None or bool(int(os.getenv("USE_GAT_SHIM", "0")))

        if use_shim:
            # フォールバック: 依存ゼロ版（GraphConvShim）
            self.impl = GATBlockShim(in_dim, hidden_dim, edge_dim=edge_dim, dropout=dropout)
            self.mode = "shim"
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"[P0-3 GAT-FALLBACK] Using GraphConvShim (PyG-free mode). "
                f"Reason: {'GATv2Conv unavailable' if GATv2Conv is None else 'USE_GAT_SHIM=1 set'}. "
                f"Performance: ~60-80% of PyG, suitable for RFI-5/6 collection."
            )
        else:
            # PyG 実装（2層 GATv2Conv）
            h1, h2 = heads
            self.g1 = GATv2Conv(
                in_dim, hidden_dim, heads=h1, edge_dim=edge_dim,
                dropout=dropout, add_self_loops=False
            )
            self.g2 = GATv2Conv(
                hidden_dim * h1, hidden_dim, heads=h2, edge_dim=edge_dim,
                dropout=dropout, add_self_loops=False, concat=False
            )
            self.drop = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(hidden_dim)
            self.mode = "pyg"
            import logging
            logger = logging.getLogger(__name__)
            logger.info("[P0-3 GAT-INIT] Using PyG GATv2Conv (full GAT implementation)")

    def forward(self, z, edge_index, edge_attr):
        if getattr(self, "mode", "pyg") == "shim":
            # Shim mode: GraphConvShim
            return self.impl(z, edge_index, edge_attr)
        else:
            # PyG mode: GATv2Conv
            h = torch.relu(self.g1(z, edge_index, edge_attr))
            h = self.drop(h)
            h = self.g2(h, edge_index, edge_attr)
            return self.norm(h)


class GatedCrossSectionFusion(nn.Module):
    """
    base(=TFT表現) と gat(=GAT表現) を同次元化し、ゲート付き残差で融合。
    ゲートは温度 τ により飽和を抑制。勾配ノルムを等方化して希釈を防ぐ。
    """
    def __init__(
        self,
        hidden: int,
        gate_per_feature: bool = False,
        tau: float = 1.25,
        init_bias: float = -0.5
    ):
        super().__init__()
        self.hidden = hidden
        self.tau = float(tau)
        # gate: [B,1] か [B,H]
        self.gate = nn.Linear(hidden * 2, hidden if gate_per_feature else 1)
        nn.init.constant_(self.gate.bias, init_bias)

    def forward(self, base: torch.Tensor, gat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        base/gat: [B,H] を想定。スケールを合わせ、gateで融合。
        戻り値: (fused [B,H], gate_value [B,1 or H])
        """
        # スケール合わせ（等方化）
        b_norm = base.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-6)
        g_norm = gat.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-6)
        gat = gat * (b_norm / g_norm)

        g_in = torch.cat([base, gat], dim=-1)
        g = torch.sigmoid(self.gate(g_in) / self.tau)  # 温度付き
        fused = base + g * gat
        return fused, g
