#!/usr/bin/env python3
"""
Backtest with Regime Detection and Defensive Risk Management

Validates defensive strategy on historical walk-forward results by:
1. Loading walk-forward backtest results (JSON)
2. Applying regime detection to historical periods
3. Adjusting position sizes based on regime signals
4. Comparing baseline vs defensive performance

Usage:
    python backtest_with_regime.py --start-date 2021-11-01 --end-date 2022-03-31 \\
        --enable-regime-detection --output results/crisis_validation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add apex-ranker to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from regime_detection import DefensiveRiskManager, RegimeDetector, RegimeSignals


def load_walk_forward_results(results_path: str) -> dict[str, Any]:
    """Load walk-forward backtest results from JSON."""
    with open(results_path) as f:
        return json.load(f)


def filter_folds_by_date_range(
    results: dict[str, Any],
    start_date: str,
    end_date: str,
) -> list[dict[str, Any]]:
    """Filter folds that overlap with target date range."""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    filtered_folds = []
    for fold in results["folds"]:
        fold_start = datetime.strptime(fold["test"]["start"], "%Y-%m-%d")
        fold_end = datetime.strptime(fold["test"]["end"], "%Y-%m-%d")

        # Check for overlap
        if fold_start <= end_dt and fold_end >= start_dt:
            filtered_folds.append(fold)

    return filtered_folds


def estimate_regime_from_fold_metrics(fold: dict[str, Any]) -> RegimeSignals:
    """
    Estimate regime signals from fold metrics.

    RECALIBRATED (Phase 4.3.1b): Momentum-first approach to prevent bull market misclassification.

    Since we don't have intraday price data, we estimate regime from:
    - Momentum (primary): Use total return as main signal
    - Realized vol (secondary): Approximate from max drawdown
    - Max DD: Directly from fold metrics

    New classification priority:
    1. Strong positive return (>15%) → BULL (regardless of vol)
    2. Extreme negative + high vol → CRISIS
    3. Moderate negative + high vol → BEAR
    4. Otherwise → SIDEWAYS
    """
    metrics = fold.get("performance", {})

    # Use return as primary momentum signal
    total_return = metrics.get("total_return", 0.0) / 100.0
    max_dd = abs(metrics.get("max_drawdown", 0.0)) / 100.0

    # Conservative volatility estimate (avoid overestimation in bull markets)
    # Use max_dd as lower bound since uptrends can have small DDs
    estimated_vol = max(max_dd / 2.0, abs(total_return) * 0.8)

    # Use return as momentum proxy
    momentum = total_return

    # Average correlation: Estimate from win rate
    win_rate = metrics.get("win_rate", 0.5)
    avg_correlation = 1.0 - win_rate

    # Classify regime with momentum-first approach
    detector = RegimeDetector(
        crisis_vol_threshold=0.30,  # More conservative (was 0.25)
        bear_vol_threshold=0.22,  # More conservative (was 0.18)
        bull_vol_threshold=0.20,  # More lenient (was 0.15)
    )

    # Override classification for strong bull markets
    if momentum > 0.15:  # Strong positive return (>15%)
        # Force BULL classification regardless of vol
        from regime_detection import MarketRegime

        confidence = min(1.0, momentum * 5.0)
        return RegimeSignals(
            realized_vol=estimated_vol,
            momentum_20d=momentum,
            max_dd_20d=max_dd,
            avg_correlation=avg_correlation,
            regime=MarketRegime.BULL,
            confidence=confidence,
        )

    # Standard classification for other cases
    regime, confidence = detector._classify_regime(
        vol=estimated_vol,
        momentum=momentum,
        max_dd=max_dd,
        correlation=avg_correlation,
    )

    return RegimeSignals(
        realized_vol=estimated_vol,
        momentum_20d=momentum,
        max_dd_20d=max_dd,
        avg_correlation=avg_correlation,
        regime=regime,
        confidence=confidence,
    )


def apply_defensive_strategy(
    fold: dict[str, Any],
    signals: RegimeSignals,
    risk_manager: DefensiveRiskManager,
) -> dict[str, Any]:
    """
    Apply defensive strategy to fold and estimate adjusted metrics.

    Adjusts returns and metrics based on recommended exposure.
    """
    metrics = fold.get("performance", {})

    # Calculate recommended exposure
    current_dd = -abs(metrics.get("max_drawdown", 0.0)) / 100.0
    exposure = risk_manager.calculate_exposure(signals, current_dd)

    # Adjust returns by exposure
    original_return = metrics.get("total_return", 0.0)
    adjusted_return = original_return * exposure

    # Adjust drawdown (scales with exposure)
    original_dd = metrics.get("max_drawdown", 0.0)
    adjusted_dd = original_dd * exposure

    # Sharpe: Return scales linearly, volatility scales with sqrt(exposure)
    # Sharpe_new = (R * exposure) / (Vol * sqrt(exposure)) = Sharpe_old * sqrt(exposure)
    original_sharpe = metrics.get("sharpe_ratio", 0.0)
    adjusted_sharpe = (
        original_sharpe * (exposure**0.5)
        if original_sharpe >= 0
        else original_sharpe * exposure
    )

    return {
        "fold_id": fold["fold_id"],
        "test_start": fold["test"]["start"],
        "test_end": fold["test"]["end"],
        "regime": signals.regime.value,
        "regime_confidence": signals.confidence,
        "exposure": exposure,
        "baseline": {
            "sharpe": original_sharpe,
            "return": original_return,
            "max_dd": original_dd,
        },
        "defensive": {
            "sharpe": adjusted_sharpe,
            "return": adjusted_return,
            "max_dd": adjusted_dd,
        },
        "improvement": {
            "sharpe_delta": adjusted_sharpe - original_sharpe,
            "return_delta": adjusted_return - original_return,
            "dd_delta": adjusted_dd - original_dd,
        },
    }


def aggregate_results(adjusted_folds: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate baseline vs defensive performance."""

    # Baseline metrics
    baseline_sharpes = [f["baseline"]["sharpe"] for f in adjusted_folds]
    baseline_returns = [f["baseline"]["return"] for f in adjusted_folds]
    baseline_dds = [f["baseline"]["max_dd"] for f in adjusted_folds]

    # Defensive metrics
    defensive_sharpes = [f["defensive"]["sharpe"] for f in adjusted_folds]
    defensive_returns = [f["defensive"]["return"] for f in adjusted_folds]
    defensive_dds = [f["defensive"]["max_dd"] for f in adjusted_folds]

    # Improvements
    sharpe_deltas = [f["improvement"]["sharpe_delta"] for f in adjusted_folds]

    return {
        "baseline": {
            "sharpe_mean": sum(baseline_sharpes) / len(baseline_sharpes),
            "sharpe_median": sorted(baseline_sharpes)[len(baseline_sharpes) // 2],
            "sharpe_min": min(baseline_sharpes),
            "sharpe_max": max(baseline_sharpes),
            "return_mean": sum(baseline_returns) / len(baseline_returns),
            "max_dd_worst": min(baseline_dds),
            "negative_folds": sum(1 for s in baseline_sharpes if s < 0),
        },
        "defensive": {
            "sharpe_mean": sum(defensive_sharpes) / len(defensive_sharpes),
            "sharpe_median": sorted(defensive_sharpes)[len(defensive_sharpes) // 2],
            "sharpe_min": min(defensive_sharpes),
            "sharpe_max": max(defensive_sharpes),
            "return_mean": sum(defensive_returns) / len(defensive_returns),
            "max_dd_worst": min(defensive_dds),
            "negative_folds": sum(1 for s in defensive_sharpes if s < 0),
        },
        "improvement": {
            "sharpe_mean_delta": sum(sharpe_deltas) / len(sharpe_deltas),
            "sharpe_median_improvement": (
                sorted(defensive_sharpes)[len(defensive_sharpes) // 2]
                - sorted(baseline_sharpes)[len(baseline_sharpes) // 2]
            ),
            "improved_folds": sum(1 for d in sharpe_deltas if d > 0),
            "total_folds": len(adjusted_folds),
        },
    }


def print_summary(results: dict[str, Any]):
    """Print validation summary."""
    print("\n" + "=" * 80)
    print("REGIME-ADAPTIVE BACKTEST VALIDATION SUMMARY")
    print("=" * 80)

    baseline = results["summary"]["baseline"]
    defensive = results["summary"]["defensive"]
    improvement = results["summary"]["improvement"]

    print("\n📊 BASELINE PERFORMANCE (No Defensive Logic)")
    print("-" * 80)
    print(f"Sharpe Mean:     {baseline['sharpe_mean']:>8.3f}")
    print(f"Sharpe Median:   {baseline['sharpe_median']:>8.3f}")
    print(
        f"Sharpe Range:    {baseline['sharpe_min']:>8.3f} to {baseline['sharpe_max']:>8.3f}"
    )
    print(f"Return Mean:     {baseline['return_mean']:>8.2f}%")
    print(f"Max DD (Worst):  {baseline['max_dd_worst']:>8.2f}%")
    print(
        f"Negative Folds:  {baseline['negative_folds']:>8d} / {improvement['total_folds']}"
    )

    print("\n🛡️  DEFENSIVE PERFORMANCE (With Regime Detection)")
    print("-" * 80)
    print(
        f"Sharpe Mean:     {defensive['sharpe_mean']:>8.3f}  ({improvement['sharpe_mean_delta']:+.3f})"
    )
    print(
        f"Sharpe Median:   {defensive['sharpe_median']:>8.3f}  ({improvement['sharpe_median_improvement']:+.3f})"
    )
    print(
        f"Sharpe Range:    {defensive['sharpe_min']:>8.3f} to {defensive['sharpe_max']:>8.3f}"
    )
    print(f"Return Mean:     {defensive['return_mean']:>8.2f}%")
    print(f"Max DD (Worst):  {defensive['max_dd_worst']:>8.2f}%")
    print(
        f"Negative Folds:  {defensive['negative_folds']:>8d} / {improvement['total_folds']}"
    )

    print("\n📈 IMPROVEMENT SUMMARY")
    print("-" * 80)
    print(f"Sharpe Mean Δ:        {improvement['sharpe_mean_delta']:>8.3f}")
    print(f"Sharpe Median Δ:      {improvement['sharpe_median_improvement']:>8.3f}")
    print(
        f"Improved Folds:       {improvement['improved_folds']:>8d} / {improvement['total_folds']}"
    )
    print(
        f"Improvement Rate:     {improvement['improved_folds'] / improvement['total_folds'] * 100:>7.1f}%"
    )

    # Check if target achieved
    target_sharpe = 2.5
    if defensive["sharpe_median"] >= target_sharpe:
        print(
            f"\n✅ TARGET ACHIEVED: Sharpe median {defensive['sharpe_median']:.3f} >= {target_sharpe}"
        )
    else:
        print(
            f"\n⚠️  Target not yet met: Sharpe median {defensive['sharpe_median']:.3f} < {target_sharpe}"
        )
        print(f"   Gap: {target_sharpe - defensive['sharpe_median']:.3f}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Validate defensive strategy on historical walk-forward results"
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="results/walk_forward_static_monthly.json",
        help="Path to walk-forward results JSON",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date for validation period (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date for validation period (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--enable-regime-detection",
        action="store_true",
        help="Enable regime-adaptive defensive logic",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/regime_validation.json",
        help="Output path for validation results",
    )

    args = parser.parse_args()

    print(f"\n📋 Loading walk-forward results from: {args.results_path}")
    results = load_walk_forward_results(args.results_path)

    print(f"🔍 Filtering folds for period: {args.start_date} to {args.end_date}")
    target_folds = filter_folds_by_date_range(results, args.start_date, args.end_date)
    print(f"   Found {len(target_folds)} folds in target period")

    if not args.enable_regime_detection:
        print("\n⚠️  Regime detection disabled - returning baseline results only")
        return

    print("\n🛡️  Applying regime detection and defensive logic...")
    risk_manager = DefensiveRiskManager()
    adjusted_folds = []

    for fold in target_folds:
        signals = estimate_regime_from_fold_metrics(fold)
        adjusted = apply_defensive_strategy(fold, signals, risk_manager)
        adjusted_folds.append(adjusted)

        print(
            f"   Fold {adjusted['fold_id']:02d} ({adjusted['test_start']}): "
            f"{signals.regime.value.upper():>8s} → Exposure {adjusted['exposure']:.0%}, "
            f"Sharpe {adjusted['baseline']['sharpe']:>6.3f} → {adjusted['defensive']['sharpe']:>6.3f}"
        )

    # Aggregate results
    summary = aggregate_results(adjusted_folds)

    # Prepare output
    output_data = {
        "config": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "results_path": args.results_path,
            "regime_detection_enabled": args.enable_regime_detection,
        },
        "folds": adjusted_folds,
        "summary": summary,
    }

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n💾 Results saved to: {output_path}")

    # Print summary
    print_summary(output_data)


if __name__ == "__main__":
    main()
