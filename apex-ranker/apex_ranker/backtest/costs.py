"""Transaction cost calculation module.

Implements the cost model defined in TRANSACTION_COST_MODEL.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class CostConfig:
    """Transaction cost configuration parameters."""

    # Commission parameters (Japanese broker fee structure)
    commission_tiers: list[tuple[float, float, float, float]] = None  # (max_value, rate, min_fee, max_fee)

    # Slippage parameters
    base_spread_bps: float = 5.0  # Base bid-ask spread (bps)
    market_impact_factor: float = 100.0  # Market impact multiplier
    max_slippage_bps: float = 20.0  # Maximum slippage cap (bps)

    def __post_init__(self):
        if self.commission_tiers is None:
            # Default: SBI Securities/Rakuten Securities fee structure
            self.commission_tiers = [
                (5_000_000, 0.00055, 55, 550),  # <= 5M
                (10_000_000, 0.00033, 0, 1_650),  # 5M-10M
                (20_000_000, 0.00022, 0, 3_300),  # 10M-20M
                (float("inf"), 0.00011, 0, 11_000),  # > 20M
            ]


def calculate_commission(trade_value_jpy: float, config: CostConfig) -> float:
    """
    Calculate commission based on Japanese broker fee structure.

    Args:
        trade_value_jpy: Trade value in JPY
        config: Cost configuration

    Returns:
        Commission in JPY
    """
    for max_value, rate, min_fee, max_fee in config.commission_tiers:
        if trade_value_jpy <= max_value:
            commission = trade_value_jpy * rate
            if min_fee > 0:
                commission = max(commission, min_fee)
            if max_fee > 0:
                commission = min(commission, max_fee)
            return commission

    # Should never reach here
    return 0.0


def calculate_slippage(
    trade_value_jpy: float,
    daily_volume_jpy: float,
    direction: str,
    config: CostConfig,
) -> float:
    """
    Calculate slippage based on linear market impact model.

    Args:
        trade_value_jpy: Trade value in JPY
        daily_volume_jpy: Daily trading volume in JPY
        direction: "buy" or "sell"
        config: Cost configuration

    Returns:
        Slippage in JPY (positive for cost, negative for benefit)
    """
    # Participation rate
    if daily_volume_jpy == 0:
        participation_rate = 0.01  # Assume 1% if volume unknown
    else:
        participation_rate = trade_value_jpy / daily_volume_jpy

    # Market impact (linear model)
    market_impact_bps = min(
        participation_rate * config.market_impact_factor,
        config.max_slippage_bps - config.base_spread_bps,
    )

    # Total slippage
    total_slippage_bps = config.base_spread_bps + market_impact_bps

    # Cap at maximum
    total_slippage_bps = min(total_slippage_bps, config.max_slippage_bps)

    # Direction adjustment (buy = cost, sell = benefit, but we count both as cost)
    sign = 1 if direction == "buy" else 1  # Always positive cost

    slippage_jpy = trade_value_jpy * (total_slippage_bps / 10000)

    return sign * slippage_jpy


def calculate_total_cost(
    trade_value_jpy: float,
    daily_volume_jpy: float,
    direction: str,
    config: CostConfig | None = None,
) -> Dict[str, float]:
    """
    Calculate total transaction cost for a single trade.

    Args:
        trade_value_jpy: Trade value in JPY
        daily_volume_jpy: Daily trading volume in JPY
        direction: "buy" or "sell"
        config: Cost configuration (uses default if None)

    Returns:
        Dict with cost breakdown:
            - commission: Commission in JPY
            - slippage: Slippage in JPY
            - total_cost: Total cost in JPY
            - total_cost_bps: Total cost in basis points
    """
    if config is None:
        config = CostConfig()

    commission = calculate_commission(trade_value_jpy, config)
    slippage = calculate_slippage(trade_value_jpy, daily_volume_jpy, direction, config)

    total_cost = commission + slippage
    total_cost_bps = (total_cost / trade_value_jpy) * 10000 if trade_value_jpy > 0 else 0.0

    return {
        "commission": commission,
        "slippage": slippage,
        "total_cost": total_cost,
        "total_cost_bps": total_cost_bps,
    }


def calculate_round_trip_cost(
    position_value_jpy: float,
    daily_volume_jpy: float,
    config: CostConfig | None = None,
) -> Dict[str, float]:
    """
    Calculate round-trip cost (buy + sell).

    Args:
        position_value_jpy: Position value in JPY
        daily_volume_jpy: Daily trading volume in JPY
        config: Cost configuration

    Returns:
        Dict with round-trip cost breakdown
    """
    if config is None:
        config = CostConfig()

    buy_cost = calculate_total_cost(position_value_jpy, daily_volume_jpy, "buy", config)
    sell_cost = calculate_total_cost(position_value_jpy, daily_volume_jpy, "sell", config)

    total_cost = buy_cost["total_cost"] + sell_cost["total_cost"]
    total_cost_bps = (total_cost / position_value_jpy) * 10000 if position_value_jpy > 0 else 0.0

    return {
        "buy_commission": buy_cost["commission"],
        "buy_slippage": buy_cost["slippage"],
        "buy_cost": buy_cost["total_cost"],
        "sell_commission": sell_cost["commission"],
        "sell_slippage": sell_cost["slippage"],
        "sell_cost": sell_cost["total_cost"],
        "total_cost": total_cost,
        "total_cost_bps": total_cost_bps,
    }


def calculate_portfolio_cost(
    portfolio_value: float,
    turnover: float,
    avg_cost_bps: float = 13.3,
) -> float:
    """
    Calculate portfolio-level transaction cost.

    Args:
        portfolio_value: Total portfolio value (JPY)
        turnover: Fraction of portfolio traded (0.0 - 1.0)
        avg_cost_bps: Average round-trip cost per position (bps)

    Returns:
        Total transaction cost (JPY)
    """
    traded_value = portfolio_value * turnover
    cost = traded_value * (avg_cost_bps / 10000)
    return cost


class CostCalculator:
    """Helper class for batch cost calculations."""

    def __init__(self, config: CostConfig | None = None):
        """
        Initialize cost calculator.

        Args:
            config: Cost configuration (uses default if None)
        """
        self.config = config if config is not None else CostConfig()

    def calculate_trade_cost(
        self,
        trade_value_jpy: float,
        daily_volume_jpy: float,
        direction: str,
    ) -> Dict[str, float]:
        """Calculate cost for a single trade."""
        return calculate_total_cost(
            trade_value_jpy,
            daily_volume_jpy,
            direction,
            self.config,
        )

    def calculate_portfolio_costs(
        self,
        trades: list[Dict],
        volumes: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Calculate costs for multiple trades.

        Args:
            trades: List of trade dicts with keys:
                - code: Stock code
                - value: Trade value (JPY)
                - direction: "buy" or "sell"
            volumes: Dict of {code: daily_volume_jpy}

        Returns:
            Dict with total costs and statistics
        """
        total_commission = 0.0
        total_slippage = 0.0
        trade_costs = []

        for trade in trades:
            code = trade["code"]
            value = trade["value"]
            direction = trade["direction"]
            volume = volumes.get(code, value * 100)  # Default: assume 1% participation

            cost = self.calculate_trade_cost(value, volume, direction)
            total_commission += cost["commission"]
            total_slippage += cost["slippage"]
            trade_costs.append(cost)

        total_cost = total_commission + total_slippage

        return {
            "total_commission": total_commission,
            "total_slippage": total_slippage,
            "total_cost": total_cost,
            "num_trades": len(trades),
            "avg_cost_per_trade": total_cost / len(trades) if trades else 0.0,
            "trade_costs": trade_costs,
        }

    def estimate_daily_cost(
        self,
        portfolio_value: float,
        turnover: float,
        avg_position_value: float,
        avg_daily_volume: float,
    ) -> Dict[str, float]:
        """
        Estimate daily transaction cost.

        Args:
            portfolio_value: Total portfolio value (JPY)
            turnover: Daily turnover rate (0.0 - 1.0)
            avg_position_value: Average position value (JPY)
            avg_daily_volume: Average daily volume (JPY)

        Returns:
            Dict with cost estimates
        """
        traded_value = portfolio_value * turnover
        num_trades = traded_value / avg_position_value if avg_position_value > 0 else 0

        # Estimate per-trade cost
        sample_cost = self.calculate_trade_cost(
            avg_position_value,
            avg_daily_volume,
            "buy",
        )

        total_cost = sample_cost["total_cost"] * num_trades * 2  # *2 for round-trip
        cost_bps = (total_cost / portfolio_value) * 10000 if portfolio_value > 0 else 0.0

        return {
            "traded_value": traded_value,
            "num_trades": num_trades,
            "total_cost": total_cost,
            "cost_bps": cost_bps,
            "cost_pct": cost_bps / 100,
        }
