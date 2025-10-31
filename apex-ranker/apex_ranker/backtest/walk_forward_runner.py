"""Walk-forward backtest execution helpers."""
from __future__ import annotations

import statistics
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from .walk_forward import WalkForwardFold, WalkForwardSplitter


@dataclass
class WalkForwardRunConfig:
    """Configuration metadata captured for a walk-forward run."""

    data_path: Path
    model_path: Path | None
    config_path: Path | None
    train_days: int
    test_days: int
    step_days: int
    gap_days: int
    mode: str
    rebalance_frequency: str
    top_k: int
    horizon: int
    initial_capital: float
    min_test_days: int
    total_folds: int
    evaluated_folds: int
    skipped_folds: int
    failed_folds: int
    start_date: str | None
    end_date: str | None
    created_at: str


def _ensure_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value)


def _load_unique_dates(data_path: Path, date_column: str) -> list[date]:
    frame = pl.read_parquet(data_path, columns=[date_column])
    series = frame[date_column].unique().sort()
    return [d if isinstance(d, date) else d.date() for d in series.to_list()]


def _filter_folds_by_range(
    folds: Iterable[WalkForwardFold],
    start: date | None,
    end: date | None,
) -> list[WalkForwardFold]:
    filtered: list[WalkForwardFold] = []
    for fold in folds:
        if start and fold.test_end < start:
            continue
        if end and fold.test_start > end:
            continue
        filtered.append(fold)
    return filtered


def _count_unique_dates(dates: list[date], start: date, end: date) -> int:
    return sum(1 for d in dates if start <= d <= end)


def _compute_metric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "stdev": 0.0}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def run_walk_forward_backtest(
    *,
    data_path: str | Path,
    splitter: WalkForwardSplitter,
    backtest_fn: Callable[..., dict],
    date_column: str = "Date",
    model_path: str | Path | None = None,
    config_path: str | Path | None = None,
    initial_capital: float = 100_000_000.0,
    top_k: int = 50,
    horizon: int = 20,
    rebalance_frequency: str = "monthly",
    device: str = "auto",
    use_mock_predictions: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    min_test_days: int = 20,
    max_folds: int | None = None,
    fold_offset: int = 0,
    fold_output_dir: str | Path | None = None,
    fold_metrics_dir: str | Path | None = None,
    fold_trades_dir: str | Path | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    extra_backtest_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute walk-forward backtests across generated folds."""

    data_path = Path(data_path)
    model_path = _ensure_path(model_path)
    config_path = _ensure_path(config_path)
    fold_output_dir = _ensure_path(fold_output_dir)
    fold_metrics_dir = _ensure_path(fold_metrics_dir)
    fold_trades_dir = _ensure_path(fold_trades_dir)
    extra_kwargs = dict(extra_backtest_kwargs or {})

    unique_dates = _load_unique_dates(data_path, date_column)
    folds = splitter.split(unique_dates)

    start_bound = (
        datetime.strptime(start_date, "%Y-%m-%d").date() if start_date else None
    )
    end_bound = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else None
    folds = _filter_folds_by_range(folds, start_bound, end_bound)

    if fold_offset:
        folds = folds[fold_offset:]
    if max_folds is not None:
        folds = folds[:max_folds]

    if not folds:
        raise ValueError(
            "No folds selected to execute. Check date filters or parameters."
        )

    if fold_output_dir:
        fold_output_dir.mkdir(parents=True, exist_ok=True)
    if fold_metrics_dir:
        fold_metrics_dir.mkdir(parents=True, exist_ok=True)
    if fold_trades_dir:
        fold_trades_dir.mkdir(parents=True, exist_ok=True)

    fold_results: list[dict[str, Any]] = []
    skipped_folds: list[dict[str, Any]] = []
    failed_folds: list[dict[str, Any]] = []

    for idx, fold in enumerate(folds, start=1):
        available_test_days = _count_unique_dates(
            unique_dates, fold.test_start, fold.test_end
        )
        if available_test_days < min_test_days:
            skip_record = {
                "fold_id": fold.fold_id,
                "test_start": fold.test_start.isoformat(),
                "test_end": fold.test_end.isoformat(),
                "available_test_days": available_test_days,
                "required_test_days": min_test_days,
            }
            skipped_folds.append(skip_record)
            if progress_callback:
                progress_callback(
                    {
                        "fold": skip_record,
                        "completed": idx,
                        "total": len(folds),
                        "status": "skipped",
                    }
                )
            continue

        fold_json_path = (
            fold_output_dir / f"fold_{fold.fold_id:03d}.json"
            if fold_output_dir
            else None
        )
        fold_metrics_path = (
            fold_metrics_dir / f"fold_{fold.fold_id:03d}_daily.csv"
            if fold_metrics_dir
            else None
        )
        fold_trades_path = (
            fold_trades_dir / f"fold_{fold.fold_id:03d}_trades.csv"
            if fold_trades_dir
            else None
        )

        backtest_kwargs: dict[str, Any] = {
            "data_path": data_path,
            "start_date": fold.test_start.isoformat(),
            "end_date": fold.test_end.isoformat(),
            "initial_capital": initial_capital,
            "top_k": top_k,
            "output_path": fold_json_path,
            "horizon": horizon,
            "device": device,
            "rebalance_freq": rebalance_frequency,
            "daily_metrics_path": fold_metrics_path,
            "trades_path": fold_trades_path,
        }

        if use_mock_predictions:
            backtest_kwargs["use_mock"] = True
        else:
            if model_path is None or config_path is None:
                raise ValueError(
                    "model_path and config_path are required unless use_mock_predictions=True"
                )
            backtest_kwargs["model_path"] = model_path
            backtest_kwargs["config_path"] = config_path

        backtest_kwargs.update(extra_kwargs)

        try:
            result = backtest_fn(**backtest_kwargs)
        except Exception as exc:  # pragma: no cover - defensive logging
            message = "".join(traceback.format_exception(exc))
            failed_folds.append(
                {
                    "fold_id": fold.fold_id,
                    "test_start": fold.test_start.isoformat(),
                    "test_end": fold.test_end.isoformat(),
                    "error": message,
                }
            )
            if progress_callback:
                progress_callback(
                    {
                        "fold": failed_folds[-1],
                        "completed": idx,
                        "total": len(folds),
                        "status": "failed",
                    }
                )
            continue

        performance = result.get("performance", {})
        summary = result.get("summary", {})

        fold_record = {
            "fold_id": fold.fold_id,
            "index": idx - 1,
            "train": {
                "start": fold.train_start.isoformat(),
                "end": fold.train_end.isoformat(),
                "days": fold.train_days,
            },
            "test": {
                "start": fold.test_start.isoformat(),
                "end": fold.test_end.isoformat(),
                "days": fold.test_days,
                "available_trading_days": available_test_days,
            },
            "performance": performance,
            "summary": summary,
            "status": "success",
        }

        fold_results.append(fold_record)

        if progress_callback:
            progress_callback(
                {
                    "fold": fold_record,
                    "completed": idx,
                    "total": len(folds),
                    "status": "success",
                }
            )

    def _collect(metric_name: str) -> list[float]:
        values = []
        for record in fold_results:
            perf = record["performance"]
            if metric_name == "transaction_cost_pct":
                tx = perf.get("transaction_costs", {})
                value = tx.get("cost_pct_of_pv")
            else:
                value = perf.get(metric_name)
            if value is not None:
                values.append(float(value))
        return values

    aggregate_metrics = {
        "total_return": _compute_metric_summary(_collect("total_return")),
        "annualized_return": _compute_metric_summary(_collect("annualized_return")),
        "sharpe_ratio": _compute_metric_summary(_collect("sharpe_ratio")),
        "max_drawdown": _compute_metric_summary(_collect("max_drawdown")),
        "win_rate": _compute_metric_summary(_collect("win_rate")),
        "avg_turnover": _compute_metric_summary(_collect("avg_turnover")),
        "transaction_cost_pct": _compute_metric_summary(
            _collect("transaction_cost_pct")
        ),
    }

    run_config = WalkForwardRunConfig(
        data_path=data_path,
        model_path=model_path,
        config_path=config_path,
        train_days=splitter.train_days,
        test_days=splitter.test_days,
        step_days=splitter.step_days,
        gap_days=splitter.gap_days,
        mode=splitter.mode,
        rebalance_frequency=rebalance_frequency,
        top_k=top_k,
        horizon=horizon,
        initial_capital=initial_capital,
        min_test_days=min_test_days,
        total_folds=len(folds),
        evaluated_folds=len(fold_results),
        skipped_folds=len(skipped_folds),
        failed_folds=len(failed_folds),
        start_date=start_date,
        end_date=end_date,
        created_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )

    config_payload = run_config.__dict__.copy()
    config_payload["data_path"] = str(config_payload["data_path"])
    if config_payload["model_path"] is not None:
        config_payload["model_path"] = str(config_payload["model_path"])
    if config_payload["config_path"] is not None:
        config_payload["config_path"] = str(config_payload["config_path"])

    return {
        "config": config_payload,
        "metrics": aggregate_metrics,
        "folds": fold_results,
        "skipped_folds": skipped_folds,
        "failed_folds": failed_folds,
    }
