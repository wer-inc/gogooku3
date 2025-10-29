"""Portfolio management for backtesting."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as Date
from typing import Dict

import numpy as np


@dataclass
class Position:
    """Single stock position."""

    code: str
    shares: float
    entry_price: float
    entry_date: Date
    current_price: float

    @property
    def value(self) -> float:
        """Current position value."""
        return self.shares * self.current_price

    @property
    def cost_basis(self) -> float:
        """Original position cost."""
        return self.shares * self.entry_price

    @property
    def pnl(self) -> float:
        """Unrealized P&L."""
        return self.value - self.cost_basis

    @property
    def return_pct(self) -> float:
        """Return percentage."""
        if self.cost_basis == 0:
            return 0.0
        return (self.pnl / self.cost_basis) * 100


@dataclass
class Trade:
    """Single trade record."""

    date: Date
    code: str
    direction: str  # "buy" or "sell"
    shares: float
    price: float
    value: float
    commission: float
    slippage: float
    total_cost: float


class Portfolio:
    """Portfolio tracker for backtesting."""

    def __init__(self, initial_capital: float):
        """
        Initialize portfolio.

        Args:
            initial_capital: Starting cash amount (JPY)
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.history: list[dict] = []
        self.trades: list[Trade] = []

        # Current state
        self.current_date: Date | None = None
        self.portfolio_value = initial_capital
        self.prev_portfolio_value = initial_capital

    @property
    def equity_value(self) -> float:
        """Total value of stock positions."""
        return sum(pos.value for pos in self.positions.values())

    @property
    def total_value(self) -> float:
        """Total portfolio value (cash + positions)."""
        return self.cash + self.equity_value

    @property
    def weights(self) -> Dict[str, float]:
        """Current position weights."""
        if self.total_value == 0:
            return {}
        return {
            code: pos.value / self.total_value for code, pos in self.positions.items()
        }

    @property
    def num_positions(self) -> int:
        """Number of open positions."""
        return len(self.positions)

    def update_prices(self, prices: Dict[str, float], date: Date) -> None:
        """
        Update current prices for all positions.

        Args:
            prices: Dict of {code: price}
            date: Current date
        """
        self.current_date = date

        for code, position in self.positions.items():
            if code in prices:
                position.current_price = prices[code]

        # Update portfolio value
        self.prev_portfolio_value = self.portfolio_value
        self.portfolio_value = self.total_value

    def get_daily_return(self) -> float:
        """Calculate daily return."""
        if self.prev_portfolio_value == 0:
            return 0.0
        return (self.portfolio_value / self.prev_portfolio_value - 1) * 100

    def rebalance(
        self,
        target_weights: Dict[str, float],
        prices: Dict[str, float],
        date: Date,
        transaction_costs: Dict[str, float] | None = None,
    ) -> list[Trade]:
        """
        Rebalance portfolio to target weights.

        Args:
            target_weights: Dict of {code: weight} for target allocation
            prices: Dict of {code: price} for execution
            date: Execution date
            transaction_costs: Dict of {code: cost} for each trade

        Returns:
            List of executed trades
        """
        if transaction_costs is None:
            transaction_costs = {}

        trades = []
        current_weights = self.weights
        target_value = self.total_value

        # Determine trades needed
        all_codes = set(current_weights.keys()) | set(target_weights.keys())

        for code in all_codes:
            current_weight = current_weights.get(code, 0.0)
            target_weight = target_weights.get(code, 0.0)

            if abs(target_weight - current_weight) < 1e-6:
                continue  # No change needed

            target_position_value = target_value * target_weight
            current_position_value = (
                self.positions[code].value if code in self.positions else 0.0
            )

            trade_value = target_position_value - current_position_value

            if trade_value > 0:
                # Buy
                direction = "buy"
                shares = trade_value / prices[code]
            else:
                # Sell
                direction = "sell"
                shares = abs(trade_value) / prices[code]

            # Apply transaction costs
            cost = transaction_costs.get(code, 0.0)

            trade = Trade(
                date=date,
                code=code,
                direction=direction,
                shares=shares,
                price=prices[code],
                value=abs(trade_value),
                commission=cost * 0.5,  # Assume 50% commission, 50% slippage
                slippage=cost * 0.5,
                total_cost=cost,
            )

            trades.append(trade)
            self.trades.append(trade)

            # Execute trade
            if direction == "buy":
                if code in self.positions:
                    # Add to existing position (average up)
                    old_pos = self.positions[code]
                    new_shares = old_pos.shares + shares
                    new_entry_price = (
                        (old_pos.shares * old_pos.entry_price) + (shares * prices[code])
                    ) / new_shares
                    self.positions[code] = Position(
                        code=code,
                        shares=new_shares,
                        entry_price=new_entry_price,
                        entry_date=old_pos.entry_date,
                        current_price=prices[code],
                    )
                else:
                    # New position
                    self.positions[code] = Position(
                        code=code,
                        shares=shares,
                        entry_price=prices[code],
                        entry_date=date,
                        current_price=prices[code],
                    )
                self.cash -= trade_value + cost
            else:
                # Sell
                if code in self.positions:
                    old_pos = self.positions[code]
                    if shares >= old_pos.shares - 1e-6:  # Close position
                        del self.positions[code]
                    else:
                        # Partial sell
                        self.positions[code].shares -= shares
                self.cash += abs(trade_value) - cost

        return trades

    def calculate_turnover(self, trades: list[Trade]) -> float:
        """
        Calculate turnover from trades.

        Turnover = (Sum of |trade_value|) / Portfolio_Value / 2
        """
        if self.total_value == 0:
            return 0.0
        total_traded = sum(trade.value for trade in trades)
        return (total_traded / self.total_value) / 2

    def log_state(self, date: Date) -> dict:
        """
        Log current portfolio state.

        Returns:
            Dict with portfolio snapshot
        """
        state = {
            "date": str(date),
            "cash": self.cash,
            "equity_value": self.equity_value,
            "portfolio_value": self.portfolio_value,
            "num_positions": self.num_positions,
            "daily_return": self.get_daily_return(),
            "positions": [
                {
                    "code": code,
                    "shares": pos.shares,
                    "value": pos.value,
                    "weight": self.weights.get(code, 0.0),
                    "pnl": pos.pnl,
                    "return_pct": pos.return_pct,
                }
                for code, pos in self.positions.items()
            ],
        }
        self.history.append(state)
        return state

    def get_history(self) -> list[dict]:
        """Get full portfolio history."""
        return self.history

    def get_trades(self) -> list[Trade]:
        """Get all trades."""
        return self.trades

    def calculate_metrics(self) -> dict:
        """
        Calculate portfolio performance metrics.

        Returns:
            Dict with performance metrics
        """
        if not self.history:
            return {}

        returns = [state["daily_return"] for state in self.history[1:]]
        values = [state["portfolio_value"] for state in self.history]

        total_return = (
            (self.portfolio_value / self.initial_capital - 1) * 100
            if self.initial_capital > 0
            else 0.0
        )

        # Calculate max drawdown
        peak = self.initial_capital
        max_dd = 0.0
        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd

        # Sharpe ratio (annualized, assume 252 trading days)
        if len(returns) > 1:
            mean_daily_return = np.mean(returns)
            std_daily_return = np.std(returns)
            sharpe_ratio = (
                mean_daily_return / std_daily_return * np.sqrt(252)
                if std_daily_return > 0
                else 0.0
            )
        else:
            sharpe_ratio = 0.0

        # Sortino ratio (downside deviation)
        negative_returns = [r for r in returns if r < 0]
        if negative_returns:
            downside_std = np.std(negative_returns)
            sortino_ratio = (
                np.mean(returns) / downside_std * np.sqrt(252)
                if downside_std > 0
                else 0.0
            )
        else:
            sortino_ratio = 0.0

        # Win rate
        win_rate = len([r for r in returns if r > 0]) / len(returns) if returns else 0.0

        # Transaction costs
        total_cost = sum(trade.total_cost for trade in self.trades)
        avg_daily_cost = total_cost / len(self.history) if self.history else 0.0

        # Turnover
        turnovers = [
            self.calculate_turnover(
                [t for t in self.trades if str(t.date) == state["date"]]
            )
            for state in self.history
        ]
        avg_turnover = np.mean(turnovers) if turnovers else 0.0

        return {
            "total_return": total_return,
            "annualized_return": (
                (1 + total_return / 100) ** (252 / len(values)) - 1
            )
            * 100,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_dd,
            "calmar_ratio": (
                ((1 + total_return / 100) ** (252 / len(values)) - 1) * 100 / max_dd
                if max_dd > 0
                else 0.0
            ),
            "win_rate": win_rate,
            "avg_turnover": avg_turnover,
            "total_trades": len(self.trades),
            "transaction_costs": {
                "total_cost": total_cost,
                "cost_pct_of_pv": (
                    total_cost / self.initial_capital * 100
                    if self.initial_capital > 0
                    else 0.0
                ),
                "avg_daily_cost_bps": (
                    avg_daily_cost / self.initial_capital * 10000
                    if self.initial_capital > 0
                    else 0.0
                ),
            },
        }
