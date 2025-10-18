from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn


class FeatureGroupDropout(nn.Module):
    def __init__(
        self, groups: Mapping[str, Sequence[int]] | None, p: float = 0.2
    ) -> None:
        super().__init__()
        self.groups = dict(groups) if groups is not None else {}
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if not self.training or not self.groups or self.p <= 0.0:
            return x
        mask = torch.ones_like(x)
        # Drop whole groups with prob p
        for idxs in self.groups.values():
            if torch.rand(1).item() < self.p:
                mask[:, list(idxs)] = 0.0
        return x * mask


class MultiHeadRegressor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        *,
        hidden: int = 512,
        groups: Mapping[str, Sequence[int]] | None = None,
        out_heads: Sequence[int] = (1, 1, 1, 1, 1),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.fgdrop = FeatureGroupDropout(groups, p=0.2) if groups else nn.Identity()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, o) for o in out_heads])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:  # type: ignore[override]
        x = self.fgdrop(x)
        h = self.backbone(x)
        return [head(h) for head in self.heads]


__all__ = ["FeatureGroupDropout", "MultiHeadRegressor"]
