#!/usr/bin/env python3
"""
Compare two walk-forward backtest runs with robust statistical tests.

This script expects the aggregated JSON artifacts produced by
``run_walk_forward_backtest.py`` (``--output`` flag). It aligns per-fold
cross-sectional metrics and daily returns, then reports:

* Diebold–Mariano test on NDCG@K improvements
* Moving block bootstrap confidence intervals for NDCG@K deltas
* Ledoit–Wolf Sharpe difference test on daily returns
* Deflated Sharpe Ratio for each run

Usage:
    python apex-ranker/scripts/analyze_run_comparison.py \
        --baseline results/baseline.json \
        --candidate results/candidate.json \
        --metric ndcg_at_k \
        --output results/comparison_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from apex_ranker.utils import (
    block_bootstrap_ci,
    deflated_sharpe_ratio,
    diebold_mariano,
    ledoit_wolf_sharpe_diff,
)


def _load_run(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    evaluation = payload.get("evaluation", {})
    data = {
        "config": payload.get("config", {}),
        "metrics": payload.get("metrics", {}),
        "folds": payload.get("folds", []),
        "evaluation": {
            "daily": evaluation.get("daily_metrics", []),
            "summaries": evaluation.get("summaries", []),
            "bootstrap": evaluation.get("bootstrap", []),
            "risk": evaluation.get("risk", []),
        },
    }
    return data


def _build_lookup(
    daily: list[dict[str, Any]],
    *,
    date_key: str,
) -> dict[tuple[int, str], dict[str, Any]]:
    lookup: dict[tuple[int, str], dict[str, Any]] = {}
    for entry in daily:
        fold_id = entry.get("fold_id")
        date_value = entry.get(date_key)
        if fold_id is None or date_value is None:
            continue
        lookup[(int(fold_id), str(date_value))] = entry
    return lookup


def _align_series(
    baseline: dict[tuple[int, str], dict[str, Any]],
    candidate: dict[tuple[int, str], dict[str, Any]],
    *,
    metric: str,
    date_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    keys = sorted(set(baseline.keys()) & set(candidate.keys()))
    series_a: list[float] = []
    series_b: list[float] = []
    for key in keys:
        a_val = baseline[key].get(metric)
        b_val = candidate[key].get(metric)
        if a_val is None or b_val is None:
            continue
        a_float = float(a_val)
        b_float = float(b_val)
        if not np.isfinite(a_float) or not np.isfinite(b_float):
            continue
        series_a.append(a_float)
        series_b.append(b_float)
    if not series_a or not series_b:
        raise ValueError(f"No overlapping observations for metric '{metric}'.")
    return np.asarray(series_a, dtype=float), np.asarray(series_b, dtype=float)


def analyse_runs(
    baseline_path: Path,
    candidate_path: Path,
    *,
    metric: str,
    horizon: int,
    alpha: float,
) -> dict[str, Any]:
    baseline = _load_run(baseline_path)
    candidate = _load_run(candidate_path)

    daily_base = baseline["evaluation"]["daily"]
    daily_candidate = candidate["evaluation"]["daily"]
    if not daily_base or not daily_candidate:
        raise ValueError("Both runs must include daily evaluation metrics.")

    # Cross-sectional metric alignment (prediction date)
    base_lookup = _build_lookup(daily_base, date_key="prediction_date")
    cand_lookup = _build_lookup(daily_candidate, date_key="prediction_date")
    metric_a, metric_b = _align_series(
        base_lookup,
        cand_lookup,
        metric=metric,
        date_key="prediction_date",
    )

    # Diebold–Mariano on metric improvements (higher is better)
    loss_a = -metric_a
    loss_b = -metric_b
    dm_result = diebold_mariano(loss_a, loss_b, horizon=horizon)

    # Bootstrap confidence interval for delta
    delta = metric_b - metric_a
    bootstrap_ci = block_bootstrap_ci(
        delta,
        block_size=max(2, int(np.sqrt(delta.size))),
        n_bootstrap=1000,
        alpha=alpha,
    )

    # Daily returns alignment (use realised date)
    base_return_lookup = _build_lookup(daily_base, date_key="result_date")
    cand_return_lookup = _build_lookup(daily_candidate, date_key="result_date")
    returns_a_pct, returns_b_pct = _align_series(
        base_return_lookup,
        cand_return_lookup,
        metric="daily_return_pct",
        date_key="result_date",
    )
    returns_a = returns_a_pct / 100.0
    returns_b = returns_b_pct / 100.0

    sharpe_result = ledoit_wolf_sharpe_diff(
        returns_a,
        returns_b,
        periods_per_year=252,
    )

    dsr_baseline = deflated_sharpe_ratio(returns_a)
    dsr_candidate = deflated_sharpe_ratio(returns_b)

    report = {
        "inputs": {
            "baseline": str(baseline_path),
            "candidate": str(candidate_path),
            "metric": metric,
            "horizon": horizon,
            "alpha": alpha,
            "observations": int(metric_a.size),
        },
        "diebold_mariano": dm_result,
        "bootstrap_delta": bootstrap_ci,
        "ledoit_wolf": sharpe_result,
        "deflated_sharpe": {
            "baseline": dsr_baseline.__dict__ if dsr_baseline else None,
            "candidate": dsr_candidate.__dict__ if dsr_candidate else None,
        },
        "summary": {
            "metric_mean_baseline": float(np.mean(metric_a)),
            "metric_mean_candidate": float(np.mean(metric_b)),
            "metric_delta_mean": float(np.mean(delta)),
        },
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run statistical comparison between two walk-forward outputs."
    )
    parser.add_argument("--baseline", required=True, help="Baseline run JSON.")
    parser.add_argument("--candidate", required=True, help="Candidate run JSON.")
    parser.add_argument(
        "--metric",
        default="ndcg_at_k",
        choices=["ndcg_at_k", "precision_at_k", "rank_ic", "wil_at_k"],
        help="Cross-sectional metric to compare (default: ndcg_at_k).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Forecast horizon for DM lag adjustment (default: 1).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for bootstrap CI (default: 0.05).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyse_runs(
        Path(args.baseline),
        Path(args.candidate),
        metric=args.metric,
        horizon=args.horizon,
        alpha=args.alpha,
    )

    print(json.dumps(report, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\n[Report] Saved to {output_path}")


if __name__ == "__main__":
    main()
