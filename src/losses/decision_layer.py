"""
Decision layer: differentiable position sizing + portfolio-aware losses.

Implements a minimal, robust setup that turns quantile predictions into
soft positions and optimizes a SoftSharpe-like objective with optional
regularizers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


def quantiles_to_ms(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert quantile outputs [B, n_quantiles] to expectation (m) and scale (s).

    Uses endpoints as an uncertainty proxy: m ≈ 0.5*(q_low+q_high), s ≈ 0.5*(q_high-q_low).
    Falls back to median if fewer than 2 quantiles available.
    """
    if q.dim() != 2 or q.size(1) == 0:
        raise ValueError("Expected [B, n_quantiles] with n_quantiles>=1")
    if q.size(1) == 1:
        m = q[:, 0]
        s = torch.zeros_like(m)
        return m, s
    q_sorted, _ = torch.sort(q, dim=1)
    q_low = q_sorted[:, 0]
    q_high = q_sorted[:, -1]
    m = 0.5 * (q_low + q_high)
    s = 0.5 * (q_high - q_low)
    return m, s


def soft_position(m: torch.Tensor, s: torch.Tensor, alpha: float = 2.0, eps: float = 1e-6,
                  method: str = "tanh") -> torch.Tensor:
    """Map expectation/uncertainty to soft position in [-1, 1].

    - method "tanh": pos = tanh(alpha * m / (s^2 + eps))
    """
    x = m / (s.pow(2) + eps)
    if method == "tanh":
        return torch.tanh(alpha * x)
    raise ValueError(f"Unsupported position method: {method}")


def soft_sharpe_loss(y: torch.Tensor, pos: torch.Tensor, span: int = 20, eps: float = 1e-8) -> torch.Tensor:
    """Negative SoftSharpe: - mean(r) / std(r), r = y * pos.

    Uses batch-wise mean/std as a simple differentiable proxy.
    """
    r = y * pos
    mu = r.mean()
    std = r.std()
    if std <= 0:
        return torch.tensor(0.0, device=r.device)
    return -(mu / (std + eps))


@dataclass
class DecisionLossConfig:
    alpha: float = 2.0
    method: str = "tanh"
    sharpe_weight: float = 0.1
    pos_l2: float = 1e-3
    fee_abs: float = 0.0  # Simple fee proportional to |pos|
    detach_signal: bool = True  # Stabilize early training by stopping grads into signal


class DecisionLayer(nn.Module):
    """Decision layer loss wrapper.

    Turn quantile predictions and targets into portfolio-aware loss components.
    """

    def __init__(self, cfg: DecisionLossConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, q: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            q: [B, n_quantiles] (sorted or not)
            y: [B]
        Returns:
            total_decision_loss, components dict
        """
        m, s = quantiles_to_ms(q)
        if self.cfg.detach_signal:
            m = m.detach()

        pos = soft_position(m, s, alpha=self.cfg.alpha, method=self.cfg.method)

        # Loss components
        l_sharpe = soft_sharpe_loss(y, pos) * self.cfg.sharpe_weight
        l_pos = (pos.pow(2).mean()) * self.cfg.pos_l2 if self.cfg.pos_l2 > 0 else q.new_tensor(0.0)
        l_fee = (pos.abs().mean()) * self.cfg.fee_abs if self.cfg.fee_abs > 0 else q.new_tensor(0.0)

        total = l_sharpe + l_pos + l_fee
        comps = {
            'decision_sharpe': l_sharpe.detach(),
            'decision_pos_l2': l_pos.detach(),
            'decision_fee': l_fee.detach(),
            'decision_pos_mean': pos.mean().detach(),
            'decision_pos_abs_mean': pos.abs().mean().detach(),
        }
        return total, comps

