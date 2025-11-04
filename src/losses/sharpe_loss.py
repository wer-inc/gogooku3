"""
Advanced Sharpe Ratio Loss Module

Directly optimizes Sharpe ratio to improve risk-adjusted returns.

Key Features:
- Differentiable Sharpe ratio computation
- Batch-wise and rolling Sharpe estimation
- Sortino ratio variant (downside risk)
- Information ratio (vs benchmark)
- Robust to small batch sizes

Usage:
    sharpe_loss = AdvancedSharpeLoss(weight=0.3, method='batch')
    loss = sharpe_loss(predictions, targets)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AdvancedSharpeLoss(nn.Module):
    """
    Advanced Sharpe ratio loss with multiple computation methods.

    Args:
        weight: Loss weight (default: 0.3)
        method: Computation method - 'batch' | 'rolling' | 'sortino' (default: 'batch')
        min_periods: Minimum periods for valid computation (default: 20)
        risk_free_rate: Annual risk-free rate (default: 0.0)
        annualization_factor: Trading days per year (default: 252)
        epsilon: Numerical stability constant (default: 1e-8)

    Methods:
        - batch: Compute Sharpe over current batch
        - rolling: Use rolling buffer for temporal consistency
        - sortino: Penalize downside risk only
    """

    def __init__(
        self,
        weight: float = 0.3,
        method: str = "batch",
        min_periods: int = 20,
        risk_free_rate: float = 0.0,
        annualization_factor: float = 252.0,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.weight = weight
        self.method = method
        self.min_periods = min_periods
        self.risk_free_rate = risk_free_rate / annualization_factor  # Daily
        self.annualization_factor = annualization_factor
        self.epsilon = epsilon

        # Rolling buffer for rolling method
        if method == "rolling":
            self.register_buffer("returns_buffer", torch.zeros(min_periods * 2))
            self.register_buffer("buffer_idx", torch.tensor(0))

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute Sharpe loss.

        Args:
            predictions: Model predictions [batch_size, n_quantiles] or [batch_size]
            targets: Actual returns [batch_size]
            positions: Optional position weights [batch_size]. If None, use sign of predictions.

        Returns:
            Negative Sharpe ratio (scalar tensor for minimization)
        """
        # Handle quantile predictions - use median
        if predictions.dim() > 1 and predictions.size(-1) > 1:
            # Assume middle quantile is median
            median_idx = predictions.size(-1) // 2
            predictions = predictions[:, median_idx]

        # Ensure 1D
        predictions = predictions.squeeze()
        targets = targets.squeeze()

        # Check minimum size
        if len(predictions) < self.min_periods:
            return torch.tensor(0.0, device=predictions.device)

        # Compute positions if not provided
        if positions is None:
            # Simple strategy: long if pred > 0, short if pred < 0
            positions = torch.tanh(predictions)  # Smooth sign function

        # Compute strategy returns
        strategy_returns = positions * targets

        # Compute Sharpe based on method
        if self.method == "batch":
            sharpe = self._compute_batch_sharpe(strategy_returns)
        elif self.method == "rolling":
            sharpe = self._compute_rolling_sharpe(strategy_returns)
        elif self.method == "sortino":
            sharpe = self._compute_sortino_ratio(strategy_returns)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Return negative Sharpe for minimization
        # Apply weight
        return -self.weight * sharpe

    def _compute_batch_sharpe(self, returns: torch.Tensor) -> torch.Tensor:
        """Compute Sharpe ratio over current batch"""
        mean_return = returns.mean()
        std_return = returns.std() + self.epsilon

        # Excess return over risk-free rate
        excess_return = mean_return - self.risk_free_rate

        # Sharpe ratio (annualized)
        sharpe = (excess_return / std_return) * torch.sqrt(
            torch.tensor(self.annualization_factor, device=returns.device)
        )

        return sharpe

    def _compute_rolling_sharpe(self, returns: torch.Tensor) -> torch.Tensor:
        """Compute Sharpe using rolling buffer for temporal consistency"""
        # Update buffer with current returns
        batch_size = len(returns)
        buffer_size = self.returns_buffer.size(0)

        # Add returns to buffer (circular)
        for i in range(batch_size):
            idx = self.buffer_idx.item() % buffer_size
            self.returns_buffer[idx] = returns[i].item()
            self.buffer_idx += 1

        # Compute Sharpe over buffer
        if self.buffer_idx < self.min_periods:
            # Not enough data yet
            return self._compute_batch_sharpe(returns)

        # Use full buffer
        mean_return = self.returns_buffer.mean()
        std_return = self.returns_buffer.std() + self.epsilon

        excess_return = mean_return - self.risk_free_rate
        sharpe = (excess_return / std_return) * torch.sqrt(
            torch.tensor(self.annualization_factor, device=returns.device)
        )

        return sharpe

    def _compute_sortino_ratio(self, returns: torch.Tensor) -> torch.Tensor:
        """
        Compute Sortino ratio (penalize downside risk only).

        More appropriate for asymmetric return distributions.
        """
        mean_return = returns.mean()

        # Downside deviation (only negative returns)
        downside_returns = torch.where(
            returns < self.risk_free_rate,
            returns - self.risk_free_rate,
            torch.zeros_like(returns),
        )
        downside_std = torch.sqrt((downside_returns ** 2).mean()) + self.epsilon

        # Sortino ratio
        excess_return = mean_return - self.risk_free_rate
        sortino = (excess_return / downside_std) * torch.sqrt(
            torch.tensor(self.annualization_factor, device=returns.device)
        )

        return sortino


class InformationRatioLoss(nn.Module):
    """
    Information Ratio loss (Sharpe vs benchmark).

    Optimizes excess return over benchmark with minimum tracking error.

    Args:
        weight: Loss weight (default: 0.3)
        benchmark_returns: Benchmark returns tensor (optional, use market return if None)
        min_periods: Minimum periods for valid computation (default: 20)
        annualization_factor: Trading days per year (default: 252)
    """

    def __init__(
        self,
        weight: float = 0.3,
        benchmark_returns: torch.Tensor | None = None,
        min_periods: int = 20,
        annualization_factor: float = 252.0,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.weight = weight
        self.min_periods = min_periods
        self.annualization_factor = annualization_factor
        self.epsilon = epsilon

        if benchmark_returns is not None:
            self.register_buffer("benchmark_returns", benchmark_returns)
        else:
            self.benchmark_returns = None

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        positions: torch.Tensor | None = None,
        benchmark_returns: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute Information Ratio loss.

        Args:
            predictions: Model predictions
            targets: Actual returns
            positions: Optional position weights
            benchmark_returns: Optional batch benchmark returns

        Returns:
            Negative Information Ratio
        """
        # Handle predictions
        if predictions.dim() > 1 and predictions.size(-1) > 1:
            median_idx = predictions.size(-1) // 2
            predictions = predictions[:, median_idx]

        predictions = predictions.squeeze()
        targets = targets.squeeze()

        if len(predictions) < self.min_periods:
            return torch.tensor(0.0, device=predictions.device)

        # Compute positions
        if positions is None:
            positions = torch.tanh(predictions)

        # Strategy returns
        strategy_returns = positions * targets

        # Benchmark returns (use market average if not provided)
        if benchmark_returns is None:
            if self.benchmark_returns is not None:
                # Use stored benchmark
                benchmark_returns = self.benchmark_returns[: len(strategy_returns)]
            else:
                # Use equal-weighted market return
                benchmark_returns = targets.mean() * torch.ones_like(strategy_returns)

        # Active returns (vs benchmark)
        active_returns = strategy_returns - benchmark_returns

        # Information ratio = mean(active_return) / std(active_return)
        mean_active = active_returns.mean()
        std_active = active_returns.std() + self.epsilon

        ir = (mean_active / std_active) * torch.sqrt(
            torch.tensor(self.annualization_factor, device=predictions.device)
        )

        return -self.weight * ir


class CalmarRatioLoss(nn.Module):
    """
    Calmar Ratio loss (return / max drawdown).

    Optimizes return while minimizing maximum drawdown.

    Args:
        weight: Loss weight (default: 0.3)
        lookback_periods: Periods for max drawdown computation (default: 252)
        min_periods: Minimum periods for valid computation (default: 60)
        annualization_factor: Trading days per year (default: 252)
    """

    def __init__(
        self,
        weight: float = 0.3,
        lookback_periods: int = 252,
        min_periods: int = 60,
        annualization_factor: float = 252.0,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.weight = weight
        self.lookback_periods = lookback_periods
        self.min_periods = min_periods
        self.annualization_factor = annualization_factor
        self.epsilon = epsilon

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute Calmar Ratio loss.

        Args:
            predictions: Model predictions
            targets: Actual returns
            positions: Optional position weights

        Returns:
            Negative Calmar Ratio
        """
        # Handle predictions
        if predictions.dim() > 1 and predictions.size(-1) > 1:
            median_idx = predictions.size(-1) // 2
            predictions = predictions[:, median_idx]

        predictions = predictions.squeeze()
        targets = targets.squeeze()

        if len(predictions) < self.min_periods:
            return torch.tensor(0.0, device=predictions.device)

        # Compute positions
        if positions is None:
            positions = torch.tanh(predictions)

        # Strategy returns
        strategy_returns = positions * targets

        # Compute cumulative returns
        cumulative_returns = torch.cumsum(strategy_returns, dim=0)

        # Compute running maximum
        running_max = torch.maximum.accumulate(cumulative_returns)[0]

        # Drawdown at each point
        drawdown = running_max - cumulative_returns

        # Maximum drawdown
        max_drawdown = drawdown.max() + self.epsilon

        # Annualized return
        total_return = cumulative_returns[-1]
        periods = len(strategy_returns)
        annualized_return = (
            total_return * self.annualization_factor / periods
        )

        # Calmar ratio
        calmar = annualized_return / max_drawdown

        return -self.weight * calmar


class MultiObjectiveRiskLoss(nn.Module):
    """
    Multi-objective risk loss combining multiple risk-adjusted metrics.

    Combines Sharpe, Sortino, and Calmar ratios with configurable weights.

    Args:
        sharpe_weight: Weight for Sharpe ratio (default: 0.4)
        sortino_weight: Weight for Sortino ratio (default: 0.3)
        calmar_weight: Weight for Calmar ratio (default: 0.3)
        min_periods: Minimum periods for valid computation (default: 60)
    """

    def __init__(
        self,
        sharpe_weight: float = 0.4,
        sortino_weight: float = 0.3,
        calmar_weight: float = 0.3,
        min_periods: int = 60,
        annualization_factor: float = 252.0,
    ):
        super().__init__()

        # Normalize weights
        total_weight = sharpe_weight + sortino_weight + calmar_weight
        self.sharpe_weight = sharpe_weight / total_weight
        self.sortino_weight = sortino_weight / total_weight
        self.calmar_weight = calmar_weight / total_weight

        # Component losses
        self.sharpe_loss = AdvancedSharpeLoss(
            weight=1.0,  # Will multiply by component weight
            method="sortino" if sortino_weight > 0 else "batch",
            min_periods=min_periods,
            annualization_factor=annualization_factor,
        )

        self.calmar_loss = CalmarRatioLoss(
            weight=1.0,
            min_periods=min_periods,
            annualization_factor=annualization_factor,
        )

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute multi-objective risk loss.

        Args:
            predictions: Model predictions
            targets: Actual returns
            positions: Optional position weights

        Returns:
            Weighted combination of risk metrics
        """
        total_loss = torch.tensor(0.0, device=predictions.device)

        # Sharpe/Sortino component
        if self.sharpe_weight > 0 or self.sortino_weight > 0:
            sharpe_component = self.sharpe_loss(predictions, targets, positions)
            total_loss = total_loss + (
                self.sharpe_weight + self.sortino_weight
            ) * sharpe_component

        # Calmar component
        if self.calmar_weight > 0 and len(predictions) >= 60:
            calmar_component = self.calmar_loss(predictions, targets, positions)
            total_loss = total_loss + self.calmar_weight * calmar_component

        return total_loss


# Convenience factory function
def create_sharpe_loss(config: dict) -> nn.Module:
    """
    Factory function to create Sharpe loss from config.

    Args:
        config: Dictionary with keys:
            - type: 'sharpe' | 'sortino' | 'information' | 'calmar' | 'multi'
            - weight: Overall loss weight
            - ... (type-specific parameters)

    Returns:
        Configured Sharpe loss module

    Example:
        >>> config = {'type': 'sharpe', 'method': 'batch', 'weight': 0.3}
        >>> loss_fn = create_sharpe_loss(config)
    """
    loss_type = config.get("type", "sharpe")

    if loss_type == "sharpe":
        return AdvancedSharpeLoss(
            weight=config.get("weight", 0.3),
            method=config.get("method", "batch"),
            min_periods=config.get("min_periods", 20),
            risk_free_rate=config.get("risk_free_rate", 0.0),
            annualization_factor=config.get("annualization_factor", 252.0),
        )
    elif loss_type == "sortino":
        return AdvancedSharpeLoss(
            weight=config.get("weight", 0.3),
            method="sortino",
            min_periods=config.get("min_periods", 20),
            annualization_factor=config.get("annualization_factor", 252.0),
        )
    elif loss_type == "information":
        return InformationRatioLoss(
            weight=config.get("weight", 0.3),
            min_periods=config.get("min_periods", 20),
            annualization_factor=config.get("annualization_factor", 252.0),
        )
    elif loss_type == "calmar":
        return CalmarRatioLoss(
            weight=config.get("weight", 0.3),
            lookback_periods=config.get("lookback_periods", 252),
            min_periods=config.get("min_periods", 60),
            annualization_factor=config.get("annualization_factor", 252.0),
        )
    elif loss_type == "multi":
        return MultiObjectiveRiskLoss(
            sharpe_weight=config.get("sharpe_weight", 0.4),
            sortino_weight=config.get("sortino_weight", 0.3),
            calmar_weight=config.get("calmar_weight", 0.3),
            min_periods=config.get("min_periods", 60),
            annualization_factor=config.get("annualization_factor", 252.0),
        )
    else:
        raise ValueError(f"Unknown Sharpe loss type: {loss_type}")
