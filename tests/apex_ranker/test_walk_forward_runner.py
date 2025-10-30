from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
from apex_ranker.backtest.walk_forward import WalkForwardSplitter
from apex_ranker.backtest.walk_forward_runner import run_walk_forward_backtest


def test_run_walk_forward_backtest_with_stub(tmp_path: Path) -> None:
    """Ensure walker aggregates per-fold metrics correctly."""
    dates = pl.date_range(
        date(2021, 1, 1),
        date(2021, 3, 31),
        interval="1d",
        eager=True,
    )
    dates = dates.filter(dates.dt.weekday() < 5)
    df = pl.DataFrame({"Date": dates})
    data_path = tmp_path / "dataset.parquet"
    df.write_parquet(data_path)

    splitter = WalkForwardSplitter(
        train_days=10,
        test_days=4,
        step_days=5,
        mode="rolling",
        min_train_days=5,
    )

    counter = {"value": 0}

    def fake_backtest_fn(**kwargs):
        counter["value"] += 1
        idx = counter["value"]
        return {
            "performance": {
                "total_return": 10.0 * idx,
                "annualized_return": 12.0 * idx,
                "sharpe_ratio": 1.0 + idx,
                "max_drawdown": 5.0,
                "win_rate": 0.6,
                "avg_turnover": 0.3,
                "transaction_costs": {
                    "cost_pct_of_pv": 20.0 * idx,
                    "total_cost": 1_000 * idx,
                },
            },
            "summary": {
                "rebalance_count": 2,
                "prediction_mode": "mock",
            },
        }

    results = run_walk_forward_backtest(
        data_path=data_path,
        splitter=splitter,
        backtest_fn=fake_backtest_fn,
        date_column="Date",
        use_mock_predictions=True,
        max_folds=2,
        initial_capital=1_000_000,
        rebalance_frequency="monthly",
        min_test_days=2,
    )

    assert len(results["folds"]) == 2
    metrics = results["metrics"]
    sharpe_mean = metrics["sharpe_ratio"]["mean"]
    assert sharpe_mean == 2.5  # Sharpe values: 2.0 and 3.0
    cost_mean = metrics["transaction_cost_pct"]["mean"]
    assert cost_mean == 30.0  # (20 + 40) / 2


def test_run_walk_forward_backtest_skips_short_folds(tmp_path: Path) -> None:
    dates = pl.date_range(
        date(2021, 1, 1),
        date(2021, 1, 20),
        interval="1d",
        eager=True,
    )
    dates = dates.filter(dates.dt.weekday() < 5)
    df = pl.DataFrame({"Date": dates})
    data_path = tmp_path / "dataset.parquet"
    df.write_parquet(data_path)

    splitter = WalkForwardSplitter(
        train_days=5,
        test_days=5,
        step_days=5,
        mode="rolling",
        min_train_days=5,
    )

    results = run_walk_forward_backtest(
        data_path=data_path,
        splitter=splitter,
        backtest_fn=lambda **kwargs: {},
        date_column="Date",
        use_mock_predictions=True,
        min_test_days=10,
    )

    assert len(results["folds"]) == 0
    assert len(results["skipped_folds"]) > 0
