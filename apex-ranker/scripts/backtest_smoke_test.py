#!/usr/bin/env python3
"""
Simple backtest driver for smoke test (Phase 3.2).

Tests integration of Portfolio + Costs + Splitter with short-term data.

Usage:
    python apex-ranker/scripts/backtest_smoke_test.py \
        --data output/ml_dataset_latest_full.parquet \
        --start-date 2025-09-01 \
        --end-date 2025-09-30 \
        --output results/backtest_smoke_test.json

Author: Claude Code
Date: 2025-10-29
"""
from __future__ import annotations

import argparse
import json
from datetime import date as Date
from datetime import datetime
from pathlib import Path

import polars as pl
from apex_ranker.backtest import Portfolio


def load_dataset_for_backtest(
    data_path: Path,
    start_date: str,
    end_date: str,
) -> pl.DataFrame:
    """
    Load dataset and filter to backtest period.

    Args:
        data_path: Path to parquet dataset
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Filtered DataFrame with required columns
    """
    print(f"[Backtest] Loading dataset: {data_path}")

    # Load required columns for backtest
    required_cols = [
        "Date",
        "Code",
        "returns_1d",  # For daily returns calculation
        "returns_5d",
        "returns_20d",
        "Volume",  # For transaction cost calculation
    ]

    df = pl.read_parquet(data_path, columns=required_cols)

    # Convert string dates to Date type for comparison
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    # Filter by date range
    df = df.filter((pl.col("Date") >= start) & (pl.col("Date") <= end))

    print(f"[Backtest] Loaded {len(df):,} rows")
    print(f"[Backtest] Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"[Backtest] Unique stocks: {df['Code'].n_unique()}")

    return df


def generate_mock_predictions(
    df: pl.DataFrame,
    date: Date,
    top_k: int = 50,
) -> dict[str, float]:
    """
    Generate mock predictions for smoke test.

    In real backtest, this would call inference_v0.py.
    For smoke test, we use 5-day forward returns as proxy.

    Args:
        df: Full dataset
        date: Prediction date
        top_k: Number of stocks to select

    Returns:
        Dict of {code: score}
    """
    # Get data for this date
    date_data = df.filter(pl.col("Date") == date)

    if len(date_data) == 0:
        return {}

    # Use 5-day returns as mock predictions
    predictions = (
        date_data.select(["Code", "returns_5d"])
        .sort("returns_5d", descending=True)
        .head(top_k * 2)  # Get more candidates to handle NaN
        .filter(pl.col("returns_5d").is_not_null())
        .head(top_k)
    )

    return {
        row["Code"]: float(row["returns_5d"])
        for row in predictions.iter_rows(named=True)
    }


def run_backtest_smoke_test(
    data_path: Path,
    start_date: str,
    end_date: str,
    initial_capital: float = 10_000_000,
    top_k: int = 50,
    output_path: Path | None = None,
) -> dict:
    """
    Run simple backtest smoke test.

    Args:
        data_path: Path to dataset
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        initial_capital: Starting capital (JPY)
        top_k: Number of stocks to hold
        output_path: Output JSON path

    Returns:
        Dict with backtest results
    """
    print("\n" + "=" * 80)
    print("Phase 3.2: Backtest Smoke Test")
    print("=" * 80)

    # Load dataset
    df = load_dataset_for_backtest(data_path, start_date, end_date)

    # Get unique trading dates
    trading_dates_raw = df["Date"].unique().sort().to_list()
    # Handle both string and date types
    trading_dates = []
    for d in trading_dates_raw:
        if isinstance(d, Date):
            trading_dates.append(d)
        elif isinstance(d, str):
            trading_dates.append(datetime.strptime(d, "%Y-%m-%d").date())
        else:
            # datetime.datetime -> date
            trading_dates.append(d.date())

    print(f"\n[Backtest] Trading days: {len(trading_dates)}")

    # Initialize components
    portfolio = Portfolio(initial_capital)

    print(f"[Backtest] Initial capital: ¥{initial_capital:,.0f}")
    print(f"[Backtest] Top-K stocks: {top_k}")

    # Track results
    daily_results = []
    total_trades = 0

    # Run backtest day-by-day
    for i, date in enumerate(trading_dates[:-1]):  # Skip last day (no T+1 price)
        next_date = trading_dates[i + 1]

        # Step 1: Generate predictions at market close (T)
        predictions = generate_mock_predictions(df, date, top_k)

        if not predictions:
            print(f"[Backtest] {date}: No predictions available, skipping")
            continue

        # Step 2: Calculate target weights (equal-weight Top-K)
        num_positions = len(predictions)
        target_weights = {code: 1.0 / num_positions for code in predictions.keys()}

        # Step 3: Get execution prices (closing price at T)
        # For smoke test, use mock prices (1000 JPY per share)
        # In production, this would be actual closing prices
        execution_prices = {code: 1000.0 for code in target_weights.keys()}

        # Step 4: Rebalance portfolio (costs will be integrated in Phase 3.3)
        trades = portfolio.rebalance(
            target_weights=target_weights,
            prices=execution_prices,
            date=date,
        )

        total_trades += len(trades)

        # Step 6: Mark-to-market at T+1
        # Get actual returns for next day
        next_date_data = df.filter(pl.col("Date") == next_date)
        returns_map = {
            row["Code"]: row["returns_1d"]
            for row in next_date_data.select(["Code", "returns_1d"]).iter_rows(
                named=True
            )
            if row["returns_1d"] is not None
        }

        # Calculate next day prices for all positions
        # Mock: base price 1000 * (1 + return%)
        next_prices = {}
        for code in portfolio.positions.keys():
            if code in returns_map:
                next_prices[code] = 1000.0 * (1 + returns_map[code] / 100)
            else:
                # If no return data, assume unchanged
                next_prices[code] = 1000.0

        portfolio.update_prices(next_prices, next_date)

        # Step 7: Log portfolio state
        state = portfolio.log_state(next_date)
        daily_results.append(state)

        # Print daily summary
        if i % 5 == 0:  # Print every 5 days
            print(
                f"[Backtest] {next_date}: "
                f"PV=¥{state['portfolio_value']:,.0f}, "
                f"Positions={state['num_positions']}, "
                f"Daily Return={state['daily_return']:.2f}%"
            )

    # Calculate final metrics
    print("\n" + "=" * 80)
    print("Backtest Results")
    print("=" * 80)

    metrics = portfolio.calculate_metrics()

    print("\nPerformance Metrics:")
    print(f"  Total Return: {metrics['total_return']:.2f}%")
    print(f"  Annualized Return: {metrics['annualized_return']:.2f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"  Sortino Ratio: {metrics['sortino_ratio']:.3f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"  Calmar Ratio: {metrics['calmar_ratio']:.3f}")
    print(f"  Win Rate: {metrics['win_rate']:.1%}")

    print("\nTransaction Statistics:")
    print(f"  Total Trades: {metrics['total_trades']}")
    print(f"  Avg Daily Turnover: {metrics['avg_turnover']:.1%}")
    tx_costs = metrics["transaction_costs"]
    print(f"  Total Transaction Costs: ¥{tx_costs['total_cost']:,.0f}")
    print(f"  Cost as % of PV: {tx_costs['cost_pct_of_pv']:.2f}%")
    print(f"  Avg Daily Cost (bps): {tx_costs['avg_daily_cost_bps']:.1f}")

    # Prepare output
    results = {
        "config": {
            "data_path": str(data_path),
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": initial_capital,
            "top_k": top_k,
        },
        "summary": {
            "trading_days": len(daily_results),
            "total_trades": total_trades,
        },
        "performance": metrics,
        "daily_results": daily_results[:10],  # First 10 days only
    }

    # Save results
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n[Backtest] Results saved to: {output_path}")

    print("\n✅ Smoke test completed successfully")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Simple backtest smoke test (Phase 3.2)"
    )
    parser.add_argument(
        "--data",
        default="output/ml_dataset_latest_full.parquet",
        help="Path to dataset",
    )
    parser.add_argument(
        "--start-date",
        default="2025-09-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        default="2025-09-30",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10_000_000,
        help="Initial capital (JPY)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of stocks to hold",
    )
    parser.add_argument(
        "--output",
        default="results/backtest_smoke_test.json",
        help="Output JSON file",
    )
    args = parser.parse_args()

    run_backtest_smoke_test(
        data_path=Path(args.data),
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        top_k=args.top_k,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
