#!/usr/bin/env python3
"""
Walk-forward backtest runner for APEX-Ranker with Regime-Adaptive Support (Phase 4.3.2).

Executes the monthly (or configurable) backtest across walk-forward folds using
regime-adaptive exposure control instead of static 100% allocation.

Key Differences from run_walk_forward_backtest.py:
- Uses run_regime_adaptive_backtest() instead of run_backtest_smoke_test()
- Adds --enable-regime-detection flag
- Adds --regime-lookback parameter
- Tracks regime statistics across folds
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apex_ranker.backtest import WalkForwardSplitter


def _load_backtest_function() -> Callable[..., dict]:
    """Dynamically load the regime-adaptive backtest runner."""
    script_path = Path(__file__).resolve().parent / "backtest_regime_adaptive.py"
    spec = importlib.util.spec_from_file_location(
        "apex_ranker_backtest_regime", script_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if not hasattr(module, "run_regime_adaptive_backtest"):
        raise AttributeError(
            "backtest_regime_adaptive.py does not define run_regime_adaptive_backtest"
        )
    return module.run_regime_adaptive_backtest


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run regime-adaptive walk-forward backtest evaluation"
    )
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

    # Regime detection parameters
    parser.add_argument(
        "--enable-regime-detection",
        action="store_true",
        default=True,
        help="Enable regime-adaptive exposure control (default: True)",
    )
    parser.add_argument(
        "--disable-regime-detection",
        dest="enable_regime_detection",
        action="store_false",
        help="Disable regime detection (use static 100%% exposure)",
    )
    parser.add_argument(
        "--regime-lookback",
        type=int,
        default=20,
        help="Lookback window for regime calculation in days (default: 20)",
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


def run_walk_forward_backtest_regime(
    data_path: Path,
    splitter: WalkForwardSplitter,
    backtest_fn: Callable,
    date_column: str,
    model_path: Path | None,
    config_path: Path | None,
    initial_capital: float,
    top_k: int,
    horizon: int,
    rebalance_frequency: str,
    device: str,
    use_mock_predictions: bool,
    enable_regime_detection: bool,
    regime_lookback: int,
    start_date: str | None,
    end_date: str | None,
    min_test_days: int,
    max_folds: int | None,
    fold_offset: int,
    fold_output_dir: str | None,
    fold_metrics_dir: str | None,
    fold_trades_dir: str | None,
    progress_callback: Callable[[dict], None] | None,
) -> dict:
    """
    Run regime-adaptive walk-forward backtest.

    This is a simplified version of the full walk_forward logic from apex_ranker.backtest.
    It iterates through folds and calls run_regime_adaptive_backtest() for each fold.
    """
    from datetime import datetime

    import polars as pl

    # Load dataset
    df = pl.read_parquet(data_path)
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found in dataset")

    df = df.with_columns(pl.col(date_column).cast(pl.Date))
    all_dates = sorted(df[date_column].unique().to_list())

    # Generate folds
    folds = splitter.split(all_dates)

    # Apply filters
    if start_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        folds = [f for f in folds if f.test_end >= start_dt]

    if end_date:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
        folds = [f for f in folds if f.test_start <= end_dt]

    # Filter by minimum test days
    folds = [f for f in folds if (f.test_end - f.test_start).days >= min_test_days]

    # Apply fold offset and max_folds
    if fold_offset > 0:
        folds = folds[fold_offset:]
    if max_folds is not None:
        folds = folds[:max_folds]

    print(f"Total folds to execute: {len(folds)}")
    print("=" * 80)
    print("Phase 4.3.2: Regime-Adaptive Walk-Forward Backtest")
    print("=" * 80)
    print(f"Regime Detection: {'ENABLED' if enable_regime_detection else 'DISABLED'}")
    if enable_regime_detection:
        print(f"Regime Lookback: {regime_lookback} days")
    print("=" * 80)
    print()

    # Execute folds
    fold_results = []
    skipped_folds = []
    failed_folds = []

    for i, fold in enumerate(folds):
        fold_id = f"fold_{i:02d}"
        print(f"[Fold {i+1:02d}/{len(folds)}] {fold.test_start} → {fold.test_end}")

        try:
            # Set up output paths for this fold
            fold_json_path = None
            fold_csv_path = None
            fold_trades_path = None

            if fold_output_dir:
                Path(fold_output_dir).mkdir(parents=True, exist_ok=True)
                fold_json_path = Path(fold_output_dir) / f"{fold_id}.json"

            if fold_metrics_dir:
                Path(fold_metrics_dir).mkdir(parents=True, exist_ok=True)
                fold_csv_path = Path(fold_metrics_dir) / f"{fold_id}.csv"

            if fold_trades_dir:
                Path(fold_trades_dir).mkdir(parents=True, exist_ok=True)
                fold_trades_path = Path(fold_trades_dir) / f"{fold_id}.csv"

            # Run regime-adaptive backtest for this fold
            result = backtest_fn(
                data_path=data_path,
                start_date=fold.test_start.strftime("%Y-%m-%d"),
                end_date=fold.test_end.strftime("%Y-%m-%d"),
                initial_capital=initial_capital,
                top_k=top_k,
                output_path=fold_json_path,
                model_path=model_path,
                config_path=config_path,
                horizon=horizon,
                device=device,
                use_mock=use_mock_predictions,
                rebalance_freq=rebalance_frequency,
                enable_regime_detection=enable_regime_detection,
                regime_lookback=regime_lookback,
                daily_metrics_path=fold_csv_path,
            )

            # Extract performance metrics
            perf = result.get("performance", {})
            fold_results.append(
                {
                    "fold_id": fold_id,
                    "train": {
                        "start": fold.train_start.strftime("%Y-%m-%d"),
                        "end": fold.train_end.strftime("%Y-%m-%d"),
                    },
                    "test": {
                        "start": fold.test_start.strftime("%Y-%m-%d"),
                        "end": fold.test_end.strftime("%Y-%m-%d"),
                    },
                    "performance": perf,
                    "regime_stats": result.get("regime_stats", {}),
                }
            )

            # Progress callback
            if progress_callback:
                progress_callback(
                    {
                        "fold": fold_results[-1],
                        "completed": len(fold_results),
                        "total": len(folds),
                    }
                )

        except Exception as e:
            print(f"[Fold {i+1:02d}] ❌ FAILED: {e}")
            failed_folds.append(
                {
                    "fold_id": fold_id,
                    "test_start": fold.test_start.strftime("%Y-%m-%d"),
                    "test_end": fold.test_end.strftime("%Y-%m-%d"),
                    "error": str(e),
                }
            )

    # Aggregate metrics
    metrics = {}
    if fold_results:
        for key in [
            "sharpe_ratio",
            "total_return",
            "max_drawdown",
            "transaction_cost_pct",
        ]:
            values = [
                f["performance"].get(key, 0.0)
                for f in fold_results
                if key in f["performance"]
            ]
            if values:
                import numpy as np

                metrics[key] = {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "std": float(np.std(values)),
                }

    # Aggregate regime statistics
    regime_summary = {}
    if enable_regime_detection:
        total_detections = sum(
            f["regime_stats"].get("regime_detections", 0) for f in fold_results
        )
        regime_counts = {}
        exposure_values = []

        for f in fold_results:
            regime_stats = f.get("regime_stats", {})
            exposure_values.append(regime_stats.get("avg_exposure", 1.0))

            # Aggregate regime counts
            for regime_name, stats in regime_stats.items():
                if isinstance(stats, dict) and "count" in stats:
                    if regime_name not in regime_counts:
                        regime_counts[regime_name] = {"count": 0, "total_exposure": 0.0}
                    regime_counts[regime_name]["count"] += stats["count"]
                    regime_counts[regime_name]["total_exposure"] += (
                        stats["avg_exposure"] * stats["count"]
                    )

        # Calculate regime summaries
        import numpy as np

        regime_summary = {
            "total_detections": total_detections,
            "avg_exposure_across_folds": float(np.mean(exposure_values))
            if exposure_values
            else 1.0,
            "min_exposure": float(np.min(exposure_values)) if exposure_values else 1.0,
            "max_exposure": float(np.max(exposure_values)) if exposure_values else 1.0,
            "regime_distribution": {
                name: {
                    "count": data["count"],
                    "pct": 100.0 * data["count"] / total_detections
                    if total_detections > 0
                    else 0.0,
                    "avg_exposure": data["total_exposure"] / data["count"]
                    if data["count"] > 0
                    else 1.0,
                }
                for name, data in regime_counts.items()
            },
        }

    return {
        "folds": fold_results,
        "skipped_folds": skipped_folds,
        "failed_folds": failed_folds,
        "metrics": metrics,
        "regime_summary": regime_summary,
        "parameters": {
            "regime_detection_enabled": enable_regime_detection,
            "regime_lookback": regime_lookback,
            "rebalance_frequency": rebalance_frequency,
            "top_k": top_k,
            "horizon": horizon,
        },
    }


def main(argv: Optional[list[str]] = None) -> None:
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
        regime_stats = fold.get("regime_stats", {})
        avg_exposure = regime_stats.get("avg_exposure", 1.0) * 100
        completed = info["completed"]
        total = info["total"]
        print(
            f"[Fold {fold['fold_id']}/{total}] "
            f"{fold['test']['start']} → {fold['test']['end']} | "
            f"Sharpe={sharpe:.3f} | Return={total_return:.2f}% | Exposure={avg_exposure:.0f}%",
            flush=True,
        )

    results = run_walk_forward_backtest_regime(
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
        enable_regime_detection=args.enable_regime_detection,
        regime_lookback=args.regime_lookback,
        start_date=args.start_date,
        end_date=args.end_date,
        min_test_days=args.min_test_days,
        max_folds=args.max_folds,
        fold_offset=args.fold_offset,
        fold_output_dir=args.fold_output_dir,
        fold_metrics_dir=args.fold_metrics_dir,
        fold_trades_dir=args.fold_trades_dir,
        progress_callback=progress,
    )

    duration = time.time() - start_time
    print(
        f"\nCompleted regime-adaptive walk-forward backtest in {duration/60:.2f} minutes"
    )

    metrics = results["metrics"]
    sharpe_stats = metrics.get("sharpe_ratio", {})
    return_stats = metrics.get("total_return", {})
    cost_stats = metrics.get("transaction_cost_pct", {})
    regime_summary = results.get("regime_summary", {})
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

    if args.enable_regime_detection:
        print("\nRegime Statistics:")
        print(f"  Total Detections  : {regime_summary.get('total_detections', 0)}")
        print(
            f"  Average Exposure  : {regime_summary.get('avg_exposure_across_folds', 1.0)*100:.1f}%"
        )
        print(
            f"  Exposure Range    : {regime_summary.get('min_exposure', 1.0)*100:.1f}% - "
            f"{regime_summary.get('max_exposure', 1.0)*100:.1f}%"
        )

        regime_dist = regime_summary.get("regime_distribution", {})
        if regime_dist:
            print("\n  Regime Distribution:")
            for regime_name, stats in sorted(
                regime_dist.items(), key=lambda x: x[1]["count"], reverse=True
            ):
                print(
                    f"    {regime_name:12s}: {stats['count']:3d} detections ({stats['pct']:5.1f}%), "
                    f"Avg Exposure={stats['avg_exposure']*100:.1f}%"
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
            f.write("Regime-Adaptive Walk-Forward Backtest Summary\n")
            f.write("==============================================\n")
            f.write(f"Folds executed : {len(results['folds'])}\n")
            f.write(f"Folds skipped  : {len(skipped)}\n")
            f.write(f"Folds failed   : {len(failed)}\n")
            f.write(f"Sharpe (mean)  : {sharpe_stats.get('mean', 0.0):.3f}\n")
            f.write(f"Sharpe (median): {sharpe_stats.get('median', 0.0):.3f}\n")
            f.write(f"Return (mean)  : {return_stats.get('mean', 0.0):.2f}%\n")
            f.write(f"TxCost (mean)  : {cost_stats.get('mean', 0.0):.2f}%\n")

            if args.enable_regime_detection:
                f.write("\nRegime Detection:\n")
                f.write("  Enabled       : Yes\n")
                f.write(f"  Lookback      : {args.regime_lookback} days\n")
                f.write(
                    f"  Total Events  : {regime_summary.get('total_detections', 0)}\n"
                )
                f.write(
                    f"  Avg Exposure  : {regime_summary.get('avg_exposure_across_folds', 1.0)*100:.1f}%\n"
                )

            f.write(f"\nRuntime (min)  : {duration/60:.2f}\n")
        print(f"[Output] Summary written to {summary_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
