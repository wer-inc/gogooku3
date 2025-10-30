"""Backtesting infrastructure for APEX-Ranker."""
from .costs import (
    CostCalculator,
    CostConfig,
    calculate_round_trip_cost,
    calculate_total_cost,
)
from .portfolio import Portfolio, Position, Trade
from .rebalance import normalise_frequency, should_rebalance
from .splitter import Split
from .splitter import WalkForwardSplitter as LegacyWalkForwardSplitter
from .splitter import generate_splits_for_backtest
from .walk_forward import WalkForwardFold, WalkForwardSplitter
from .walk_forward_runner import WalkForwardRunConfig, run_walk_forward_backtest

__all__ = [
    # Portfolio
    "Portfolio",
    "Position",
    "Trade",
    # Rebalance helpers
    "normalise_frequency",
    "should_rebalance",
    # Costs
    "CostCalculator",
    "CostConfig",
    "calculate_total_cost",
    "calculate_round_trip_cost",
    # Splitter
    "Split",
    "WalkForwardFold",
    "WalkForwardSplitter",
    "LegacyWalkForwardSplitter",
    "generate_splits_for_backtest",
    # Walk-forward runner
    "WalkForwardRunConfig",
    "run_walk_forward_backtest",
]
