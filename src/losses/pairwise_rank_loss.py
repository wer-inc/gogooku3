"""
Pairwise Rank loss utilities.

Implements a simple pairwise logistic loss (RankNet-style) with optional
hard-negative mining to cap compute to Top-K pairs within a batch.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PairwiseRankLoss(nn.Module):
    """Pairwise logistic rank loss with optional hard-negative mining.

    Args:
        s: scale for the margin in logistic loss.
        topk: if >0, select top-k samples by absolute error |y - z.detach()| to form pairs.

    Forward:
        z: [B] scores (higher means better rank)
        y: [B] targets
    """

    def __init__(self, s: float = 5.0, topk: int = 0):
        super().__init__()
        self.s = float(s)
        self.topk = int(topk)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if z.dim() != 1:
            z = z.view(-1)
        if y.dim() != 1:
            y = y.view(-1)

        n = z.size(0)
        if n <= 1:
            return torch.tensor(0.0, device=z.device)

        # Hard-negative mining subset selection (optional)
        if self.topk and self.topk < n:
            err = torch.abs(y - z.detach())
            idx = torch.topk(err, k=self.topk, largest=True).indices
            z = z[idx]
            y = y[idx]

        # Build pairwise differences
        diff = z.unsqueeze(1) - z.unsqueeze(0)  # [B, B]
        sign = torch.sign((y.unsqueeze(1) - y.unsqueeze(0)).clamp(min=-1, max=1))  # [-1,0,1]

        # Keep only strictly-ordered pairs to reduce noise
        mask = sign != 0
        if not mask.any():
            return torch.tensor(0.0, device=z.device)
        diff = diff[mask]
        sign = sign[mask]

        # Logistic loss log(1 + exp(-s * sign * diff))
        loss = torch.log1p(torch.exp(-self.s * sign * diff)).mean()
        return loss

