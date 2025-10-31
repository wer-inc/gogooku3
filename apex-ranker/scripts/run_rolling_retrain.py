#!/usr/bin/env python3
"""
Monthly rolling retraining and evaluation pipeline for APEX-Ranker.

For each calendar month within the specified evaluation range, the pipeline:
1. Trains a fresh model on the preceding rolling window (default: 252 trading days)
2. Evaluates the trained model on the target month using the cost-aware backtest
3. Aggregates key metrics (Sharpe, turnover, transaction cost) to track degradation

This orchestrator shells out to the existing training (`train_v0.py`) and
backtesting (`backtest_smoke_test.py`) scripts, leveraging their configuration.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class RollingWindow:
    """Train/test window definition for a single evaluation month."""

    month_label: str
    train_start: date
    train_end: date
    val_start: date
    val_end: date
    test_start: date
    test_end: date


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rolling 252d retraining + monthly evaluation for APEX-Ranker"
    )
    parser.add_argument("--data", required=True, help="Parquet dataset path")
    parser.add_argument(
        "--config",
        default="apex-ranker/configs/v0_base.yaml",
        help="Model config YAML for training/inference",
    )
    parser.add_argument(
        "--output-dir",
        default="results/rolling_retrain",
        help="Directory to store models, evaluations, and summary",
    )
    parser.add_argument(
        "--start-date",
        default="2024-01-01",
        help="Evaluation start date (YYYY-MM-DD, inclusive)",
    )
    parser.add_argument(
        "--end-date",
        default="2025-12-31",
        help="Evaluation end date (YYYY-MM-DD, inclusive)",
    )
    parser.add_argument(
        "--train-days",
        type=int,
        default=252,
        help="Number of trading days in rolling training window (default: 252)",
    )
    parser.add_argument(
        "--val-days",
        type=int,
        default=21,
        help="Number of trading days reserved for validation inside train window (default: 21)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Optional cap on training epochs passed to train_v0.py",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Candidate top-K passed to backtest",
    )
    parser.add_argument(
        "--target-top-k",
        type=int,
        default=35,
        help="Optimised holdings target passed to backtest",
    )
    parser.add_argument(
        "--min-position-weight",
        type=float,
        default=0.02,
        help="Minimum position weight constraint for backtest",
    )
    parser.add_argument(
        "--turnover-limit",
        type=float,
        default=0.35,
        help="Turnover constraint for backtest",
    )
    parser.add_argument(
        "--cost-penalty",
        type=float,
        default=1.0,
        help="Transaction cost penalty multiplier for backtest",
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=float,
        default=2.0,
        help="Candidate pool multiplier for backtest optimisation",
    )
    parser.add_argument(
        "--min-alpha",
        type=float,
        default=0.1,
        help="Minimum adjustment factor when enforcing turnover constraint",
    )
    parser.add_argument(
        "--panel-cache-dir",
        default="cache/panel",
        help="Directory for persisted panel caches (shared across months)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=20,
        help="Prediction horizon passed to backtest",
    )
    parser.add_argument(
        "--train-script",
        default="apex-ranker/scripts/train_v0.py",
        help="Training script path (defaults to train_v0.py)",
    )
    parser.add_argument(
        "--backtest-script",
        default="apex-ranker/scripts/backtest_smoke_test.py",
        help="Backtest script path (defaults to backtest_smoke_test.py)",
    )
    return parser.parse_args(argv)


def load_trading_calendar(data_path: Path, date_column: str = "Date") -> list[date]:
    frame = pl.read_parquet(str(data_path), columns=[date_column])
    return [
        d if isinstance(d, date) else d.date()
        for d in frame[date_column].unique().sort().to_list()
    ]


def month_range(start: date, end: date) -> Iterable[date]:
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        yield cursor
        year = cursor.year + (cursor.month // 12)
        month = 1 if cursor.month == 12 else cursor.month + 1
        cursor = date(year, month, 1)


def find_month_window(
    trading_days: list[date],
    month_start: date,
    train_days: int,
    val_days: int,
) -> RollingWindow | None:
    month_end = date(
        month_start.year + (month_start.month // 12),
        1 if month_start.month == 12 else month_start.month + 1,
        1,
    ) - timedelta(days=1)

    test_dates = [d for d in trading_days if month_start <= d <= month_end]
    if not test_dates:
        return None

    test_start = test_dates[0]
    test_end = test_dates[-1]

    test_start_idx = trading_days.index(test_start)
    train_end_idx = test_start_idx - 1
    if train_end_idx < 0:
        return None

    train_start_idx = train_end_idx - (train_days - 1)
    if train_start_idx < 0:
        return None

    val_length = min(val_days, train_days // 5)
    val_end_idx = train_end_idx
    val_start_idx = max(train_start_idx, val_end_idx - (val_length - 1))

    train_start = trading_days[train_start_idx]
    train_end = trading_days[train_end_idx]
    val_start = trading_days[val_start_idx]
    val_end = trading_days[val_end_idx]

    label = f"{test_start:%Y-%m}"
    return RollingWindow(
        month_label=label,
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        test_end=test_end,
    )


def run_subprocess(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(cmd)}"
        )


def date_to_str(value: date) -> str:
    return value.strftime("%Y-%m-%d")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    data_path = Path(args.data).expanduser()
    config_path = Path(args.config).expanduser()
    output_root = Path(args.output_dir).expanduser()
    models_dir = output_root / "models"
    eval_dir = output_root / "evaluations"
    models_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    trading_days = load_trading_calendar(data_path)
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()

    windows: list[RollingWindow] = []
    for month_start in month_range(start_date, end_date):
        window = find_month_window(
            trading_days,
            month_start,
            train_days=args.train_days,
            val_days=args.val_days,
        )
        if window is not None:
            windows.append(window)

    if not windows:
        raise ValueError(
            "No evaluation windows constructed. Check date range and dataset coverage."
        )

    print(
        f"[Scheduler] Prepared {len(windows)} monthly windows between {args.start_date} and {args.end_date}"
    )

    summary_records: list[dict[str, Any]] = []
    sharpe_history: list[float] = []

    for window in windows:
        print("\n" + "=" * 80)
        print(f"[Scheduler] Month: {window.month_label}")
        print(
            f"[Scheduler] Train {window.train_start} → {window.train_end} "
            f"(val {window.val_start} → {window.val_end}) | "
            f"Test {window.test_start} → {window.test_end}"
        )

        model_path = models_dir / f"apex_ranker_{window.month_label}.pt"
        train_cmd = [
            sys.executable,
            str(REPO_ROOT / args.train_script),
            "--config",
            str(config_path),
            "--output",
            str(model_path),
            "--train-start-date",
            date_to_str(window.train_start),
            "--train-end-date",
            date_to_str(window.train_end),
            "--val-start-date",
            date_to_str(window.val_start),
            "--val-end-date",
            date_to_str(window.val_end),
        ]
        if args.max_epochs is not None:
            train_cmd += ["--max-epochs", str(args.max_epochs)]

        run_subprocess(train_cmd, cwd=REPO_ROOT)

        eval_output = eval_dir / f"{window.month_label}.json"
        daily_csv = eval_dir / f"{window.month_label}_daily.csv"
        trades_csv = eval_dir / f"{window.month_label}_trades.csv"

        backtest_cmd = [
            sys.executable,
            str(REPO_ROOT / args.backtest_script),
            "--data",
            str(data_path),
            "--start-date",
            date_to_str(window.test_start),
            "--end-date",
            date_to_str(window.test_end),
            "--model",
            str(model_path),
            "--config",
            str(config_path),
            "--top-k",
            str(args.top_k),
            "--target-top-k",
            str(args.target_top_k),
            "--min-position-weight",
            f"{args.min_position_weight}",
            "--turnover-limit",
            f"{args.turnover_limit}",
            "--cost-penalty",
            f"{args.cost_penalty}",
            "--candidate-multiplier",
            f"{args.candidate_multiplier}",
            "--min-alpha",
            f"{args.min_alpha}",
            "--panel-cache-dir",
            args.panel_cache_dir,
            "--horizon",
            str(args.horizon),
            "--output",
            str(eval_output),
            "--daily-csv",
            str(daily_csv),
            "--trades-csv",
            str(trades_csv),
        ]

        run_subprocess(backtest_cmd, cwd=REPO_ROOT)

        with eval_output.open("r") as fh:
            evaluation = json.load(fh)

        performance = evaluation.get("performance", {})
        tx_costs = performance.get("transaction_costs", {})
        sharpe = float(performance.get("sharpe_ratio", 0.0))
        sharpe_history.append(sharpe)
        baseline_sharpe = sharpe_history[0]
        sharpe_delta = sharpe - baseline_sharpe

        previous_sharpe = (
            sharpe_history[-2] if len(sharpe_history) > 1 else baseline_sharpe
        )

        record = {
            "month": window.month_label,
            "train_start": date_to_str(window.train_start),
            "train_end": date_to_str(window.train_end),
            "test_start": date_to_str(window.test_start),
            "test_end": date_to_str(window.test_end),
            "sharpe_ratio": sharpe,
            "total_return_pct": float(performance.get("total_return", 0.0)),
            "max_drawdown_pct": float(performance.get("max_drawdown", 0.0)),
            "avg_turnover": float(performance.get("avg_turnover", 0.0)),
            "transaction_cost_pct": float(tx_costs.get("cost_pct_of_pv", 0.0)),
            "transaction_cost_total": float(tx_costs.get("total_cost", 0.0)),
            "sharpe_delta_vs_first": sharpe_delta,
            "sharpe_delta_vs_prev": sharpe - previous_sharpe,
            "eval_output": str(eval_output),
            "daily_csv": str(daily_csv),
            "trades_csv": str(trades_csv),
            "model_path": str(model_path),
        }
        summary_records.append(record)

    summary = {
        "config": {
            "data_path": str(data_path),
            "config_path": str(config_path),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "train_days": args.train_days,
            "val_days": args.val_days,
            "top_k": args.top_k,
            "target_top_k": args.target_top_k,
            "min_position_weight": args.min_position_weight,
            "turnover_limit": args.turnover_limit,
            "cost_penalty": args.cost_penalty,
            "candidate_multiplier": args.candidate_multiplier,
            "min_alpha": args.min_alpha,
            "panel_cache_dir": args.panel_cache_dir,
            "horizon": args.horizon,
        },
        "records": summary_records,
    }

    sharpe_values = [record["sharpe_ratio"] for record in summary_records]
    cost_values = [record["transaction_cost_pct"] for record in summary_records]

    if sharpe_values:
        summary["metrics"] = {
            "sharpe_mean": float(sum(sharpe_values) / len(sharpe_values)),
            "sharpe_min": float(min(sharpe_values)),
            "sharpe_max": float(max(sharpe_values)),
            "cost_pct_mean": float(sum(cost_values) / len(cost_values))
            if cost_values
            else 0.0,
        }

    summary_path = output_root / "rolling_retrain_summary.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[Scheduler] Summary written to {summary_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
