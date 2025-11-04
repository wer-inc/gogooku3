"""
Transaction Cost Loss Module

Penalizes high portfolio turnover to reduce trading costs and improve net Sharpe ratio.

Key Features:
- Turnover-based cost modeling
- Configurable cost basis points
- Temporal consistency regularization
- Position change tracking

Usage:
    loss_fn = TransactionCostLoss(cost_bps=10, turnover_weight=0.5)
    cost_loss = loss_fn(current_positions, previous_positions)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TransactionCostLoss(nn.Module):
    """
    Transaction cost loss based on portfolio turnover.

    Penalizes position changes to reduce trading frequency and associated costs.

    Args:
        cost_bps: Transaction cost in basis points (default: 10 = 0.1%)
        turnover_weight: Weight for turnover penalty (default: 0.5)
        use_l1: Use L1 norm for turnover (vs L2). L1 is more robust. (default: True)
        normalize_by_volatility: Normalize turnover by position volatility (default: False)

    Example:
        >>> loss_fn = TransactionCostLoss(cost_bps=10, turnover_weight=0.5)
        >>> current_pos = model(batch)  # [batch_size, n_assets]
        >>> previous_pos = get_previous_positions()
        >>> cost = loss_fn(current_pos, previous_pos)
    """

    def __init__(
        self,
        cost_bps: float = 10.0,
        turnover_weight: float = 0.5,
        use_l1: bool = True,
        normalize_by_volatility: bool = False,
    ):
        super().__init__()
        self.cost_fraction = cost_bps / 10000.0  # Convert bps to fraction
        self.turnover_weight = turnover_weight
        self.use_l1 = use_l1
        self.normalize_by_volatility = normalize_by_volatility

    def forward(
        self,
        positions_current: torch.Tensor,
        positions_previous: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute transaction cost loss.

        Args:
            positions_current: Current predicted positions [batch_size, n_assets] or [batch_size]
            positions_previous: Previous positions (optional). If None, computed from batch.

        Returns:
            Transaction cost loss (scalar tensor)
        """
        # Handle None previous positions (first batch)
        if positions_previous is None:
            # No turnover penalty for first batch
            return torch.tensor(0.0, device=positions_current.device)

        # Ensure same device
        positions_previous = positions_previous.to(positions_current.device)

        # Compute position changes (turnover)
        position_change = positions_current - positions_previous

        # Compute turnover metric
        if self.use_l1:
            # L1: Sum of absolute position changes
            turnover = torch.abs(position_change).mean()
        else:
            # L2: RMS of position changes
            turnover = torch.sqrt((position_change ** 2).mean())

        # Optional: Normalize by volatility
        if self.normalize_by_volatility:
            # Compute rolling volatility of positions
            pos_std = positions_current.std() + 1e-8
            turnover = turnover / pos_std

        # Transaction cost = cost_fraction × turnover
        transaction_cost = self.cost_fraction * turnover

        # Apply weight
        weighted_cost = self.turnover_weight * transaction_cost

        return weighted_cost

    def extra_repr(self) -> str:
        return (
            f"cost_bps={self.cost_fraction * 10000:.1f}, "
            f"turnover_weight={self.turnover_weight}, "
            f"use_l1={self.use_l1}, "
            f"normalize_by_volatility={self.normalize_by_volatility}"
        )


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency regularization.

    Penalizes large prediction changes over time to reduce IC volatility.

    Args:
        weight: Weight for consistency penalty (default: 0.1)
        use_squared: Use squared difference (L2) vs absolute (L1) (default: False)

    Example:
        >>> consistency_fn = TemporalConsistencyLoss(weight=0.1)
        >>> pred_current = model(batch_t)
        >>> pred_previous = model(batch_t_minus_1)
        >>> consistency_loss = consistency_fn(pred_current, pred_previous)
    """

    def __init__(self, weight: float = 0.1, use_squared: bool = False):
        super().__init__()
        self.weight = weight
        self.use_squared = use_squared

    def forward(
        self,
        predictions_current: torch.Tensor,
        predictions_previous: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute temporal consistency loss.

        Args:
            predictions_current: Current predictions [batch_size, ...]
            predictions_previous: Previous predictions (optional)

        Returns:
            Consistency loss (scalar tensor)
        """
        if predictions_previous is None:
            return torch.tensor(0.0, device=predictions_current.device)

        predictions_previous = predictions_previous.to(predictions_current.device)

        # Compute prediction changes
        pred_change = predictions_current - predictions_previous

        # Compute penalty
        if self.use_squared:
            consistency_penalty = (pred_change ** 2).mean()
        else:
            consistency_penalty = torch.abs(pred_change).mean()

        return self.weight * consistency_penalty


class AdaptiveTurnoverLoss(nn.Module):
    """
    Adaptive turnover loss that adjusts penalty based on market regime.

    In high volatility periods, allows more trading.
    In low volatility periods, penalizes turnover more heavily.

    Args:
        base_cost_bps: Base transaction cost in bps (default: 10)
        turnover_weight: Weight for turnover penalty (default: 0.5)
        volatility_window: Window for computing volatility (default: 20)
        min_weight_multiplier: Minimum weight adjustment (default: 0.5)
        max_weight_multiplier: Maximum weight adjustment (default: 2.0)
    """

    def __init__(
        self,
        base_cost_bps: float = 10.0,
        turnover_weight: float = 0.5,
        volatility_window: int = 20,
        min_weight_multiplier: float = 0.5,
        max_weight_multiplier: float = 2.0,
    ):
        super().__init__()
        self.base_cost = base_cost_bps / 10000.0
        self.turnover_weight = turnover_weight
        self.volatility_window = volatility_window
        self.min_mult = min_weight_multiplier
        self.max_mult = max_weight_multiplier

        # Rolling volatility buffer
        self.register_buffer("volatility_buffer", torch.zeros(volatility_window))
        self.register_buffer("buffer_idx", torch.tensor(0))

    def _update_volatility(self, returns: torch.Tensor) -> torch.Tensor:
        """Update rolling volatility estimate"""
        # Compute current volatility
        current_vol = returns.std()

        # Update buffer
        idx = self.buffer_idx.item() % self.volatility_window
        self.volatility_buffer[idx] = current_vol
        self.buffer_idx += 1

        # Compute rolling volatility
        if self.buffer_idx < self.volatility_window:
            # Not enough data, use current
            rolling_vol = current_vol
        else:
            rolling_vol = self.volatility_buffer.mean()

        return rolling_vol

    def forward(
        self,
        positions_current: torch.Tensor,
        positions_previous: torch.Tensor | None = None,
        returns: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute adaptive turnover loss.

        Args:
            positions_current: Current positions
            positions_previous: Previous positions
            returns: Recent returns for volatility estimation (optional)

        Returns:
            Adaptive turnover cost
        """
        if positions_previous is None:
            return torch.tensor(0.0, device=positions_current.device)

        positions_previous = positions_previous.to(positions_current.device)

        # Compute turnover
        turnover = torch.abs(positions_current - positions_previous).mean()

        # Adjust weight based on volatility (if returns provided)
        if returns is not None:
            rolling_vol = self._update_volatility(returns)
            # Normalize volatility (assume typical vol ~ 0.02 daily)
            vol_normalized = rolling_vol / 0.02

            # High vol → lower penalty (allow more trading)
            # Low vol → higher penalty (reduce trading)
            weight_multiplier = torch.clamp(
                1.0 / vol_normalized,
                min=self.min_mult,
                max=self.max_mult,
            )
        else:
            weight_multiplier = 1.0

        # Transaction cost with adaptive weighting
        cost = self.base_cost * turnover * weight_multiplier * self.turnover_weight

        return cost


# Convenience function for easy integration
def create_turnover_loss(config: dict) -> nn.Module:
    """
    Factory function to create turnover loss from config.

    Args:
        config: Dictionary with keys:
            - type: 'transaction_cost' | 'temporal_consistency' | 'adaptive'
            - cost_bps: Transaction cost in basis points
            - weight: Loss weight
            - ... (other type-specific parameters)

    Returns:
        Configured loss module

    Example:
        >>> config = {'type': 'transaction_cost', 'cost_bps': 10, 'weight': 0.5}
        >>> loss_fn = create_turnover_loss(config)
    """
    loss_type = config.get("type", "transaction_cost")

    if loss_type == "transaction_cost":
        return TransactionCostLoss(
            cost_bps=config.get("cost_bps", 10.0),
            turnover_weight=config.get("weight", 0.5),
            use_l1=config.get("use_l1", True),
            normalize_by_volatility=config.get("normalize_by_volatility", False),
        )
    elif loss_type == "temporal_consistency":
        return TemporalConsistencyLoss(
            weight=config.get("weight", 0.1),
            use_squared=config.get("use_squared", False),
        )
    elif loss_type == "adaptive":
        return AdaptiveTurnoverLoss(
            base_cost_bps=config.get("cost_bps", 10.0),
            turnover_weight=config.get("weight", 0.5),
            volatility_window=config.get("volatility_window", 20),
            min_weight_multiplier=config.get("min_weight_mult", 0.5),
            max_weight_multiplier=config.get("max_weight_mult", 2.0),
        )
    else:
        raise ValueError(f"Unknown turnover loss type: {loss_type}")
