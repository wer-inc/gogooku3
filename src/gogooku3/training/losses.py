from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class HuberMultiHorizon(nn.Module):
    """Smooth L1 (Huber) loss per horizon with optional volatility scaling."""

    def __init__(
        self,
        *,
        deltas: Sequence[float] = (0.01, 0.015, 0.02, 0.025, 0.03),
        horizon_w: Sequence[float] = (1.0, 0.9, 0.8, 0.7, 0.6),
    ) -> None:
        super().__init__()
        self.deltas = list(deltas)
        self.horizon_w = list(horizon_w)
        self._huber = nn.SmoothL1Loss(reduction="none")

    def forward(
        self,
        preds_list: list[torch.Tensor],
        targets_list: list[torch.Tensor],
        *,
        vol20: torch.Tensor | None = None,
    ) -> torch.Tensor:  # type: ignore[override]
        assert len(preds_list) == len(targets_list), "mismatch heads/targets"
        total = torch.tensor(0.0, device=preds_list[0].device)
        for i, (p, t) in enumerate(zip(preds_list, targets_list)):
            delta = self.deltas[min(i, len(self.deltas) - 1)]
            # Temporarily adjust beta for different deltas via scaling (SmoothL1Loss has fixed beta=1)
            loss = self._huber(p.squeeze(-1) / delta, t / delta)
            if vol20 is not None:
                loss = loss / (vol20 + 1e-6)
            w = self.horizon_w[min(i, len(self.horizon_w) - 1)]
            total = total + w * loss.mean()
        return total


__all__ = ["HuberMultiHorizon"]

