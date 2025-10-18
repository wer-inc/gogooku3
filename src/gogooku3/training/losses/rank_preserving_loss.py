#!/usr/bin/env python3
"""
Rank-Preserving Regularizer for ATFT-GAT-FAN
Spearman correlation penalty to tighten RankIC without destabilizing IC

Purpose:
- Add rank correlation regularizer: λ_rank * (1 - SpearmanCorr(pred, target))
- Tighten RankIC without destabilizing raw IC
- Weight: λ_rank = 0.1 (tunable via HPO)

Usage:
    from gogooku3.training.losses import RankPreservingLoss

    loss_fn = RankPreservingLoss(rank_weight=0.1)
    total_loss = loss_fn(predictions, targets)
"""

import torch
import torch.nn as nn


class RankPreservingLoss(nn.Module):
    """
    Rank-preserving regularizer using Spearman correlation penalty.

    Loss = base_loss + λ_rank * (1 - SpearmanCorr(pred, target))

    Args:
        base_loss: Base loss function (e.g., MSE, Huber)
        rank_weight: Weight for rank correlation penalty (default: 0.1)
        eps: Small constant for numerical stability (default: 1e-8)
    """

    def __init__(
        self,
        base_loss: nn.Module | None = None,
        rank_weight: float = 0.1,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.base_loss = base_loss if base_loss is not None else nn.MSELoss()
        self.rank_weight = rank_weight
        self.eps = eps

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reduce: bool = True,
    ) -> torch.Tensor:
        """
        Compute loss with rank-preserving regularization.

        Args:
            predictions: Model predictions (N, *)
            targets: Ground truth targets (N, *)
            reduce: Whether to reduce to scalar (default: True)

        Returns:
            Total loss with rank penalty
        """
        # Base loss
        base_loss_val = self.base_loss(predictions, targets)

        # Flatten for rank computation
        pred_flat = predictions.reshape(-1)
        target_flat = targets.reshape(-1)

        # Remove NaN/Inf values
        valid_mask = torch.isfinite(pred_flat) & torch.isfinite(target_flat)
        if valid_mask.sum() < 2:
            # Not enough valid samples for correlation
            return base_loss_val

        pred_valid = pred_flat[valid_mask]
        target_valid = target_flat[valid_mask]

        # Compute Spearman correlation
        spearman_corr = self._spearman_correlation(pred_valid, target_valid)

        # Rank penalty: (1 - correlation)
        # Higher correlation → lower penalty
        rank_penalty = 1.0 - spearman_corr

        # Total loss
        total_loss = base_loss_val + self.rank_weight * rank_penalty

        return total_loss if reduce else (base_loss_val, rank_penalty)

    def _spearman_correlation(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Spearman's rank correlation coefficient.

        Spearman = Pearson(rank(x), rank(y))

        Args:
            x: First tensor (N,)
            y: Second tensor (N,)

        Returns:
            Spearman correlation coefficient
        """
        # Convert to ranks (differentiable via argsort)
        x_rank = self._rank_tensor(x)
        y_rank = self._rank_tensor(y)

        # Compute Pearson correlation on ranks
        return self._pearson_correlation(x_rank, y_rank)

    def _rank_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert tensor to ranks (differentiable).

        Args:
            x: Input tensor (N,)

        Returns:
            Ranks as float tensor (N,)
        """
        # argsort gives indices, argsort(argsort) gives ranks
        sorted_indices = torch.argsort(x)
        ranks = torch.empty_like(sorted_indices, dtype=torch.float32)
        ranks[sorted_indices] = torch.arange(
            len(x), dtype=torch.float32, device=x.device
        )
        return ranks

    def _pearson_correlation(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Pearson correlation coefficient.

        Args:
            x: First tensor (N,)
            y: Second tensor (N,)

        Returns:
            Pearson correlation coefficient
        """
        # Center variables
        x_centered = x - x.mean()
        y_centered = y - y.mean()

        # Compute correlation
        numerator = (x_centered * y_centered).sum()
        denominator = torch.sqrt(
            (x_centered**2).sum() * (y_centered**2).sum() + self.eps
        )

        return numerator / denominator


class MultiHorizonRankPreservingLoss(nn.Module):
    """
    Multi-horizon rank-preserving loss for ATFT-GAT-FAN.

    Applies rank regularization to each prediction horizon independently.

    Args:
        horizons: List of horizon names (e.g., ['horizon_1d', 'horizon_5d', ...])
        base_loss: Base loss function (default: MSE)
        rank_weight: Weight for rank penalty (default: 0.1)
        horizon_weights: Per-horizon weights (default: uniform)
    """

    def __init__(
        self,
        horizons: list[str] | None = None,
        base_loss: nn.Module | None = None,
        rank_weight: float = 0.1,
        horizon_weights: dict[str, float] | None = None,
    ):
        super().__init__()
        self.horizons = horizons or [
            "horizon_1d",
            "horizon_5d",
            "horizon_10d",
            "horizon_20d",
        ]
        self.rank_loss = RankPreservingLoss(
            base_loss=base_loss, rank_weight=rank_weight
        )

        # Default: uniform weights
        if horizon_weights is None:
            horizon_weights = {h: 1.0 for h in self.horizons}
        self.horizon_weights = horizon_weights

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute multi-horizon rank-preserving loss.

        Args:
            predictions: Dict of {horizon: predictions}
            targets: Dict of {horizon: targets}

        Returns:
            Weighted sum of per-horizon losses
        """
        total_loss = 0.0
        total_weight = 0.0

        for horizon in self.horizons:
            if horizon not in predictions or horizon not in targets:
                continue

            pred_h = predictions[horizon]
            target_h = targets[horizon]

            # Compute loss for this horizon
            loss_h = self.rank_loss(pred_h, target_h)

            # Weight and accumulate
            weight_h = self.horizon_weights.get(horizon, 1.0)
            total_loss += weight_h * loss_h
            total_weight += weight_h

        # Normalize by total weight
        if total_weight > 0:
            total_loss = total_loss / total_weight

        return total_loss
