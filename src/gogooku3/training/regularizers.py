from __future__ import annotations

import torch
import torch.nn as nn


class L2Penalty(nn.Module):
    def __init__(self, weight: float = 1e-4) -> None:
        super().__init__()
        self.weight = float(weight)

    def forward(self, model: nn.Module) -> torch.Tensor:  # type: ignore[override]
        reg = torch.tensor(0.0)
        for p in model.parameters():
            if p.requires_grad:
                reg = reg + p.pow(2).sum()
        return self.weight * reg


__all__ = ["L2Penalty"]

