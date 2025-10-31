from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from .patchtst import PatchTSTEncoder


class MultiHorizonHead(nn.Module):
    """Linear heads for each forecasting horizon."""

    def __init__(self, d_model: int, horizons: Iterable[int]) -> None:
        super().__init__()
        self.horizons = [int(h) for h in horizons]
        self.heads = nn.ModuleDict(
            {self._key(h): nn.Linear(d_model, 1) for h in self.horizons}
        )

    @staticmethod
    def _key(horizon: int) -> str:
        return f"h{int(horizon)}"

    def forward(self, z: torch.Tensor) -> dict[int, torch.Tensor]:
        outputs: dict[int, torch.Tensor] = {}
        for h in self.horizons:
            head = self.heads[self._key(h)]
            outputs[h] = head(z).squeeze(-1)
        return outputs


class APEXRankerV0(nn.Module):
    """v0 baseline model (PatchTST + multi-horizon linear heads)."""

    def __init__(
        self,
        in_features: int,
        horizons: Iterable[int],
        *,
        d_model: int = 192,
        depth: int = 3,
        patch_len: int = 16,
        stride: int = 8,
        n_heads: int = 8,
        dropout: float = 0.1,
        loss_fn: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.horizons = [int(h) for h in horizons]

        self.encoder = PatchTSTEncoder(
            in_feats=in_features,
            d_model=d_model,
            depth=depth,
            patch_len=patch_len,
            stride=stride,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.head = MultiHorizonHead(d_model, self.horizons)
        self.loss_fn = loss_fn

    def forward(self, X: torch.Tensor) -> dict[int, torch.Tensor]:
        pooled, _ = self.encoder(X)
        return self.head(pooled)

    def compute_loss(
        self, scores: dict[int, torch.Tensor], targets: torch.Tensor
    ) -> torch.Tensor:
        if self.loss_fn is None:
            raise RuntimeError("loss_fn was not provided to APEXRankerV0")

        total = torch.tensor(0.0, device=targets.device)
        for idx, horizon in enumerate(self.horizons):
            if idx >= targets.shape[1]:
                break
            y = targets[:, idx]
            if torch.std(y) < 1e-6:
                continue
            total = total + self.loss_fn(scores[horizon], y)
        return total
