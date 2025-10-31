#!/usr/bin/env python3
"""
Walk-forward backtest runner for APEX-Ranker.

Executes the monthly (or configurable) backtest across walk-forward folds and
aggregates metrics for Phase 4.2 validation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apex_ranker.backtest import WalkForwardSplitter, run_walk_forward_backtest


def _load_backtest_function() -> Callable[..., dict]:
    """Dynamically load the single-period backtest runner."""
    script_path = Path(__file__).resolve().parent / "backtest_smoke_test.py"
    spec = importlib.util.spec_from_file_location(
        "apex_ranker_backtest_smoke", script_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if not hasattr(module, "run_backtest_smoke_test"):
        raise AttributeError(
            "backtest_smoke_test.py does not define run_backtest_smoke_test"
        )
    return module.run_backtest_smoke_test


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run walk-forward backtest evaluation")
    parser.add_argument("--data", required=True, help="Parquet dataset path")
    parser.add_argument("--model", default=None, help="Model checkpoint (.pt)")
    parser.add_argument("--config", default=None, help="Model config YAML")
    parser.add_argument(
        "--date-column", default="Date", help="Dataset date column name"
    )

    parser.add_argument(
        "--train-days", type=int, default=252, help="Training window size in days"
    )
    parser.add_argument(
        "--test-days", type=int, default=63, help="Test window size in days"
    )
    parser.add_argument("--step-days", type=int, default=21, help="Step size in days")
    parser.add_argument(
        "--min-train-days",
        type=int,
        default=126,
        help="Minimum allowable training days (must be <= train-days)",
    )
    parser.add_argument(
        "--mode",
        choices=["rolling", "expanding"],
        default="rolling",
        help="Walk-forward mode",
    )
    parser.add_argument(
        "--gap-days", type=int, default=0, help="Gap between train/test windows"
    )

    parser.add_argument(
        "--rebalance-freq", default="monthly", help="Rebalance frequency"
    )
    parser.add_argument(
        "--top-k", type=int, default=50, help="Number of stocks to hold"
    )
    parser.add_argument(
        "--target-top-k",
        type=int,
        default=35,
        help="Target holdings count after optimisation (default: 35)",
    )
    parser.add_argument(
        "--min-position-weight",
        type=float,
        default=0.02,
        help="Minimum allocation weight per position (default: 0.02)",
    )
    parser.add_argument(
        "--turnover-limit",
        type=float,
        default=0.35,
        help="Maximum turnover fraction allowed per rebalance (default: 0.35)",
    )
    parser.add_argument(
        "--cost-penalty",
        type=float,
        default=1.0,
        help="Penalty multiplier applied to estimated round-trip transaction costs",
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=float,
        default=2.0,
        help="Multiplier controlling candidate pool size vs. target top-k (default: 2.0)",
    )
    parser.add_argument(
        "--min-alpha",
        type=float,
        default=0.1,
        help="Minimum adjustment factor when enforcing turnover constraints (default: 0.1)",
    )
    parser.add_argument(
        "--panel-cache-dir",
        default="cache/panel",
        help="Directory for persisted panel caches (default: cache/panel)",
    )
    parser.add_argument(
        "--horizon", type=int, default=20, help="Prediction horizon in days"
    )
    parser.add_argument(
        "--initial-capital", type=float, default=100_000_000.0, help="Initial capital"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device used for inference",
    )
    parser.add_argument(
        "--use-mock-predictions",
        action="store_true",
        help="Use mock predictions instead of model inference",
    )

    parser.add_argument(
        "--start-date", default=None, help="Minimum test start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", default=None, help="Maximum test end date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--fold-offset",
        type=int,
        default=0,
        help="Skip this many folds before starting",
    )
    parser.add_argument(
        "--max-folds", type=int, default=None, help="Limit number of folds to run"
    )
    parser.add_argument(
        "--min-test-days",
        type=int,
        default=20,
        help="Minimum trading days in test period (default: 20)",
    )

    parser.add_argument(
        "--fold-output-dir", default=None, help="Directory for per-fold JSON outputs"
    )
    parser.add_argument(
        "--fold-metrics-dir",
        default=None,
        help="Directory for per-fold daily metrics CSV",
    )
    parser.add_argument(
        "--fold-trades-dir", default=None, help="Directory for per-fold trade logs"
    )

    parser.add_argument(
        "--output", required=True, help="Path to write aggregate JSON results"
    )
    parser.add_argument(
        "--summary", default=None, help="Optional path to write human-readable summary"
    )

    parser.add_argument(
        "--max-folds-preview",
        type=int,
        default=3,
        help="Number of folds to preview before confirmation (for future use)",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    if not args.use_mock_predictions and args.model is None:
        raise ValueError(
            "Model path is required unless --use-mock-predictions is specified"
        )
    if not args.use_mock_predictions and args.config is None:
        raise ValueError(
            "Config path is required unless --use-mock-predictions is specified"
        )

    splitter = WalkForwardSplitter(
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        mode=args.mode,
        min_train_days=args.min_train_days,
        gap_days=args.gap_days,
    )

    backtest_fn = _load_backtest_function()

    start_time = time.time()

    def progress(info: dict[str, Any]) -> None:
        fold = info["fold"]
        perf = fold.get("performance", {})
        sharpe = perf.get("sharpe_ratio")
        total_return = perf.get("total_return")
        completed = info["completed"]
        total = info["total"]
        print(
            f"[Fold {fold['fold_id']:02d}/{total}] "
            f"{fold['test']['start']} → {fold['test']['end']} | "
            f"Sharpe={sharpe:.3f} | Return={total_return:.2f}%",
            flush=True,
        )

    extra_kwargs = {
        "optimization_target_top_k": args.target_top_k,
        "min_position_weight": args.min_position_weight,
        "turnover_limit": args.turnover_limit,
        "cost_penalty": args.cost_penalty,
        "candidate_multiplier": args.candidate_multiplier,
        "min_alpha": args.min_alpha,
        "panel_cache_dir": Path(args.panel_cache_dir).expanduser()
        if args.panel_cache_dir
        else None,
    }

    results = run_walk_forward_backtest(
        data_path=data_path,
        splitter=splitter,
        backtest_fn=backtest_fn,
        date_column=args.date_column,
        model_path=args.model,
        config_path=args.config,
        initial_capital=args.initial_capital,
        top_k=args.top_k,
        horizon=args.horizon,
        rebalance_frequency=args.rebalance_freq,
        device=args.device,
        use_mock_predictions=args.use_mock_predictions,
        start_date=args.start_date,
        end_date=args.end_date,
        min_test_days=args.min_test_days,
        max_folds=args.max_folds,
        fold_offset=args.fold_offset,
        fold_output_dir=args.fold_output_dir,
        fold_metrics_dir=args.fold_metrics_dir,
        fold_trades_dir=args.fold_trades_dir,
        progress_callback=progress,
        extra_backtest_kwargs=extra_kwargs,
    )

    duration = time.time() - start_time
    print(f"\nCompleted walk-forward backtest in {duration/60:.2f} minutes")

    metrics = results["metrics"]
    sharpe_stats = metrics.get("sharpe_ratio", {})
    return_stats = metrics.get("total_return", {})
    cost_stats = metrics.get("transaction_cost_pct", {})
    skipped = results.get("skipped_folds", [])
    failed = results.get("failed_folds", [])

    print(
        f"\nFolds executed : {len(results['folds'])} | skipped={len(skipped)} | failed={len(failed)}"
    )

    print("\nAggregate Metrics:")
    print(
        f"  Sharpe Ratio  | mean={sharpe_stats.get('mean', 0.0):.3f} "
        f"median={sharpe_stats.get('median', 0.0):.3f} "
        f"min={sharpe_stats.get('min', 0.0):.3f} max={sharpe_stats.get('max', 0.0):.3f}"
    )
    print(
        f"  Total Return  | mean={return_stats.get('mean', 0.0):.2f}% "
        f"median={return_stats.get('median', 0.0):.2f}%"
    )
    print(
        f"  Tx Cost (%PV) | mean={cost_stats.get('mean', 0.0):.2f}% "
        f"median={cost_stats.get('median', 0.0):.2f}%"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Output] Aggregate results written to {output_path}")

    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w") as f:
            f.write("Walk-Forward Backtest Summary\n")
            f.write("==============================\n")
            f.write(f"Folds executed : {len(results['folds'])}\n")
            f.write(f"Folds skipped  : {len(skipped)}\n")
            f.write(f"Folds failed   : {len(failed)}\n")
            f.write(f"Sharpe (mean)  : {sharpe_stats.get('mean', 0.0):.3f}\n")
            f.write(f"Sharpe (median): {sharpe_stats.get('median', 0.0):.3f}\n")
            f.write(f"Return (mean)  : {return_stats.get('mean', 0.0):.2f}%\n")
            f.write(f"TxCost (mean)  : {cost_stats.get('mean', 0.0):.2f}%\n")
            f.write(f"Runtime (min)  : {duration/60:.2f}\n")
        print(f"[Output] Summary written to {summary_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
