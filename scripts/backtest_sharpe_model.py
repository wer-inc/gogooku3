#!/usr/bin/env python3
"""
Backtest Framework for Sharpe Optimization Validation

This script simulates realistic trading to evaluate if Sharpe improvements
are achievable with transaction cost modeling.

Usage:
    # Using existing checkpoint
    python scripts/backtest_sharpe_model.py \
        --checkpoint output/checkpoints/epoch_120_*.pth \
        --data-path output/ml_dataset_latest_full.parquet \
        --output-dir output/backtest_results

    # Quick test mode
    python scripts/backtest_sharpe_model.py --checkpoint <path> --days 100 --fast-mode
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from tqdm import tqdm

warnings.filterwarnings('ignore')


class TransactionCostModel:
    """Realistic transaction cost modeling for Japanese equities"""

    def __init__(
        self,
        base_cost_bps: float = 10.0,      # Base: 0.1% per trade
        market_impact_k: float = 0.001,    # Market impact coefficient
        slippage_bps: float = 2.0,         # Bid-ask spread: 0.02%
    ):
        self.base_cost_bps = base_cost_bps
        self.market_impact_k = market_impact_k
        self.slippage_bps = slippage_bps

    def compute_cost(
        self,
        trade_value: float,
        adv: float = 1e9,  # Average daily volume in JPY
    ) -> float:
        """
        Compute total transaction cost for a trade.

        Args:
            trade_value: Absolute value of trade in JPY
            adv: Average daily volume in JPY

        Returns:
            Total cost in JPY
        """
        # Base commission
        base_cost = trade_value * (self.base_cost_bps / 10000)

        # Market impact (square root model)
        participation_rate = trade_value / max(adv, 1e6)
        market_impact = trade_value * self.market_impact_k * np.sqrt(participation_rate)

        # Slippage (bid-ask spread)
        slippage = trade_value * (self.slippage_bps / 10000)

        return base_cost + market_impact + slippage


class PortfolioSimulator:
    """Daily rebalancing portfolio simulator"""

    def __init__(
        self,
        initial_capital: float = 100_000_000,  # 100M JPY
        cost_model: TransactionCostModel | None = None,
        long_only: bool = False,
        max_position: float = 0.05,  # Max 5% per stock
    ):
        self.initial_capital = initial_capital
        self.cost_model = cost_model or TransactionCostModel()
        self.long_only = long_only
        self.max_position = max_position

        # State
        self.capital = initial_capital
        self.positions = {}  # {code: shares}
        self.prices = {}     # {code: price}

        # History
        self.history = {
            'date': [],
            'portfolio_value': [],
            'cash': [],
            'daily_return': [],
            'turnover': [],
            'transaction_costs': [],
        }

    def get_portfolio_value(self) -> float:
        """Current total portfolio value"""
        stock_value = sum(
            self.positions.get(code, 0) * self.prices.get(code, 0)
            for code in self.positions
        )
        return self.capital + stock_value

    def rebalance(
        self,
        date: str,
        predictions: Dict[str, float],
        prices: Dict[str, float],
        market_caps: Dict[str, float] | None = None,
    ) -> Tuple[float, float]:
        """
        Rebalance portfolio based on predictions.

        Args:
            date: Trading date
            predictions: {code: predicted_return}
            prices: {code: current_price}
            market_caps: Optional market cap for liquidity estimation

        Returns:
            (turnover, transaction_costs)
        """
        portfolio_value = self.get_portfolio_value()

        # Update prices
        self.prices.update(prices)

        # Compute target positions (equal weight within long/short buckets)
        target_weights = self._compute_target_weights(predictions)
        target_values = {
            code: weight * portfolio_value
            for code, weight in target_weights.items()
        }

        # Current position values
        current_values = {
            code: self.positions.get(code, 0) * prices.get(code, 0)
            for code in set(list(target_values.keys()) + list(self.positions.keys()))
        }

        # Compute trades
        trades = {}
        total_trade_value = 0
        total_costs = 0

        for code in set(list(target_values.keys()) + list(current_values.keys())):
            current_val = current_values.get(code, 0)
            target_val = target_values.get(code, 0)
            trade_val = target_val - current_val

            if abs(trade_val) > 1e-6:  # Minimum trade threshold
                trades[code] = trade_val
                total_trade_value += abs(trade_val)

                # Compute cost
                adv = market_caps.get(code, 1e9) * 0.01 if market_caps else 1e9
                cost = self.cost_model.compute_cost(abs(trade_val), adv)
                total_costs += cost

        # Execute trades
        for code, trade_val in trades.items():
            price = prices.get(code)
            if price is None or price <= 0:
                continue

            shares_to_trade = trade_val / price
            self.positions[code] = self.positions.get(code, 0) + shares_to_trade

            # Remove zero positions
            if abs(self.positions[code]) < 1e-6:
                del self.positions[code]

        # Deduct costs from cash
        self.capital -= total_costs

        # Compute turnover
        turnover = total_trade_value / portfolio_value if portfolio_value > 0 else 0

        # Record history
        new_portfolio_value = self.get_portfolio_value()
        daily_return = (new_portfolio_value / portfolio_value - 1) if portfolio_value > 0 else 0

        self.history['date'].append(date)
        self.history['portfolio_value'].append(new_portfolio_value)
        self.history['cash'].append(self.capital)
        self.history['daily_return'].append(daily_return)
        self.history['turnover'].append(turnover)
        self.history['transaction_costs'].append(total_costs)

        return turnover, total_costs

    def _compute_target_weights(self, predictions: Dict[str, float]) -> Dict[str, float]:
        """
        Compute target portfolio weights from predictions.

        Strategy: Rank-based long-short
        - Top 20%: Equal-weight long
        - Bottom 20%: Equal-weight short (if not long_only)
        - Middle 60%: Zero weight
        """
        if not predictions:
            return {}

        # Sort by prediction
        sorted_codes = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        n = len(sorted_codes)

        # Top and bottom 20%
        n_long = max(1, int(n * 0.2))
        n_short = max(1, int(n * 0.2)) if not self.long_only else 0

        long_codes = [code for code, _ in sorted_codes[:n_long]]
        short_codes = [code for code, _ in sorted_codes[-n_short:]] if not self.long_only else []

        # Equal weight within buckets
        weights = {}

        if long_codes:
            long_weight = min(self.max_position, 1.0 / len(long_codes))
            for code in long_codes:
                weights[code] = long_weight

        if short_codes:
            short_weight = min(self.max_position, 1.0 / len(short_codes))
            for code in short_codes:
                weights[code] = -short_weight

        return weights

    def get_results(self) -> pd.DataFrame:
        """Get backtest results as DataFrame"""
        return pd.DataFrame(self.history)


class BacktestEngine:
    """Main backtest orchestrator"""

    def __init__(
        self,
        model: nn.Module | None,
        device: str = 'cuda',
        cost_model: TransactionCostModel | None = None,
    ):
        self.model = model
        self.device = device
        self.cost_model = cost_model or TransactionCostModel()

        if self.model is not None:
            self.model.eval()
            self.model.to(device)

    def run_backtest(
        self,
        data: pl.DataFrame,
        start_date: str | None = None,
        end_date: str | None = None,
        initial_capital: float = 100_000_000,
        long_only: bool = False,
    ) -> pd.DataFrame:
        """
        Run backtest on historical data.

        Args:
            data: Polars DataFrame with predictions/features
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Starting capital in JPY
            long_only: Long-only constraint

        Returns:
            Results DataFrame with daily metrics
        """
        # Filter dates
        if start_date:
            data = data.filter(pl.col('Date') >= start_date)
        if end_date:
            data = data.filter(pl.col('Date') <= end_date)

        # Get unique dates
        dates = sorted(data['Date'].unique().to_list())

        print(f"Running backtest from {dates[0]} to {dates[-1]} ({len(dates)} days)")

        # Initialize simulator
        simulator = PortfolioSimulator(
            initial_capital=initial_capital,
            cost_model=self.cost_model,
            long_only=long_only,
        )

        # Daily simulation
        for date in tqdm(dates, desc="Backtesting"):
            day_data = data.filter(pl.col('Date') == date)

            if len(day_data) == 0:
                continue

            # Generate predictions (placeholder - would use model.predict())
            # For now, use actual returns as "predictions" to test framework
            # Filter out None/NaN values
            codes = day_data['Code'].to_list()
            returns = day_data['returns_1d'].to_list()
            predictions = {
                code: ret for code, ret in zip(codes, returns)
                if ret is not None and not (isinstance(ret, float) and np.isnan(ret))
            }

            prices = dict(zip(
                day_data['Code'].to_list(),
                [100.0] * len(day_data)  # Placeholder prices
            ))

            # Rebalance
            simulator.rebalance(date, predictions, prices)

        return simulator.get_results()


def compute_performance_metrics(results: pd.DataFrame) -> Dict[str, float]:
    """Compute comprehensive performance metrics"""

    returns = results['daily_return'].values
    portfolio_values = results['portfolio_value'].values

    # Sharpe ratio (annualized)
    mean_return = returns.mean() * 252
    std_return = returns.std() * np.sqrt(252)
    sharpe = mean_return / std_return if std_return > 0 else 0

    # Sortino ratio (downside risk only)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 1e-8
    sortino = mean_return / downside_std

    # Maximum drawdown
    cummax = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - cummax) / cummax
    max_drawdown = drawdowns.min()

    # Calmar ratio
    calmar = mean_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # Turnover
    avg_turnover = results['turnover'].mean()

    # Transaction costs
    total_costs = results['transaction_costs'].sum()
    total_return = portfolio_values[-1] / portfolio_values[0] - 1
    cost_drag = total_costs / portfolio_values[0]

    return {
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'annual_return': mean_return,
        'annual_volatility': std_return,
        'max_drawdown': max_drawdown,
        'avg_daily_turnover': avg_turnover,
        'total_return': total_return,
        'transaction_cost_drag': cost_drag,
    }


def main():
    parser = argparse.ArgumentParser(description="Backtest Sharpe-optimized model")
    parser.add_argument('--checkpoint', type=str, help="Path to model checkpoint")
    parser.add_argument('--data-path', type=str, required=True, help="Path to ML dataset")
    parser.add_argument('--output-dir', type=str, default='output/backtest_results', help="Output directory")
    parser.add_argument('--start-date', type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument('--end-date', type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument('--days', type=int, help="Number of days to backtest")
    parser.add_argument('--long-only', action='store_true', help="Long-only strategy")
    parser.add_argument('--base-cost-bps', type=float, default=10.0, help="Base transaction cost (bps)")
    parser.add_argument('--fast-mode', action='store_true', help="Fast mode (skip model, use actual returns)")

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data_path}...")
    data = pl.read_parquet(args.data_path)
    print(f"Loaded {len(data):,} rows, {len(data.columns)} columns")

    # Filter to test period if specified
    if args.days:
        dates = sorted(data['Date'].unique().to_list())
        args.start_date = dates[-args.days]
        args.end_date = dates[-1]
        print(f"Using last {args.days} days: {args.start_date} to {args.end_date}")

    # Load model (or skip in fast mode)
    model = None
    if not args.fast_mode and args.checkpoint:
        print(f"Loading model from {args.checkpoint}...")
        # TODO: Load model architecture and weights
        print("⚠️  Model loading not yet implemented - using placeholder")

    # Run backtest
    cost_model = TransactionCostModel(base_cost_bps=args.base_cost_bps)

    if args.fast_mode:
        print("Fast mode: Using actual returns as predictions")

    engine = BacktestEngine(model, cost_model=cost_model)
    results = engine.run_backtest(
        data,
        start_date=args.start_date,
        end_date=args.end_date,
        long_only=args.long_only,
    )

    # Compute metrics
    metrics = compute_performance_metrics(results)

    # Print results
    print("\n" + "="*80)
    print("BACKTEST RESULTS")
    print("="*80)
    for key, value in metrics.items():
        print(f"{key:30s}: {value:10.4f}")
    print("="*80)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(output_dir / 'daily_results.csv', index=False)
    pd.DataFrame([metrics]).to_csv(output_dir / 'summary_metrics.csv', index=False)

    print(f"\n✅ Results saved to {output_dir}/")

    return 0


if __name__ == '__main__':
    sys.exit(main())
