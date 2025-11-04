#!/usr/bin/env python3
"""Analyze APEX-Ranker AB experiment results and recommend optimal gamma."""
import json
from pathlib import Path


def load_result(path):
    """Load backtest JSON result."""
    with open(path) as f:
        return json.load(f)

def analyze_experiments(results_dir="results/p0_ab_final"):
    """Compare all AB experiment results."""
    results_dir = Path(results_dir)

    experiments = {
        "BASE": "BASE.json",
        "A.3 only": "A3_only.json",
        "A.4 (γ=0.3)": "A4_g030.json",
        "A.3+A.4 (γ=0.2)": "A3A4_g020.json",
        "A.3+A.4 (γ=0.3)": "A3A4_g030.json",
        "A.3+A.4 (γ=0.5)": "A3A4_g050.json",
    }

    results = {}
    baseline_sharpe = None

    print("=" * 100)
    print(" " * 30 + "APEX-Ranker AB Experiment Results")
    print("=" * 100)
    print()

    # Load all results
    for name, filename in experiments.items():
        filepath = results_dir / filename
        if not filepath.exists():
            print(f"⚠️  {name}: File not found - {filepath}")
            continue

        data = load_result(filepath)
        perf = data['performance']
        summary = data.get('summary', {})

        results[name] = {
            'sharpe': perf['sharpe_ratio'],
            'return': perf['total_return'] * 100,
            'max_dd': perf['max_drawdown'] * 100,
            'sortino': perf['sortino_ratio'],
            'calmar': perf['calmar_ratio'],
            'turnover': perf['avg_turnover'] * 100,
            'trades': perf['total_trades'],
            'costs': perf['transaction_costs']['total_cost'] / 1e6,
            'rebalances': summary.get('rebalance_count', 'N/A'),
        }

        if name == "BASE":
            baseline_sharpe = perf['sharpe_ratio']

    # Print comparison table
    print(f"{'Experiment':<25} {'Sharpe':<10} {'vs BASE':<12} {'Return':<12} {'MaxDD':<10} {'Turnover':<10} {'Trades':<8} {'Decision':<10}")
    print("-" * 100)

    best_sharpe = 0
    best_config = None

    for name, metrics in results.items():
        sharpe = metrics['sharpe']
        if baseline_sharpe and name != "BASE":
            sharpe_delta_pct = ((sharpe / baseline_sharpe) - 1) * 100
            sharpe_delta_str = f"{sharpe_delta_pct:+.1f}%"
        else:
            sharpe_delta_str = "-"

        # Decision logic (user spec)
        if name == "BASE":
            decision = "📊 Baseline"
        elif sharpe > baseline_sharpe * 1.05:  # +5% threshold
            decision = "✅ PASS"
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_config = name
        elif sharpe > baseline_sharpe:
            decision = "⚠️  Marginal"
        else:
            decision = "❌ REJECT"

        print(f"{name:<25} {sharpe:<10.3f} {sharpe_delta_str:<12} {metrics['return']:>+10.2f}% {metrics['max_dd']:>8.2f}% {metrics['turnover']:>8.2f}% {metrics['trades']:<8} {decision}")

    print("-" * 100)
    print()

    # Detailed metrics
    print("=" * 100)
    print(" " * 35 + "Detailed Metrics")
    print("=" * 100)
    print()
    print(f"{'Experiment':<25} {'Sortino':<10} {'Calmar':<10} {'Costs (¥M)':<12} {'Rebalances':<12}")
    print("-" * 100)

    for name, metrics in results.items():
        print(f"{name:<25} {metrics['sortino']:<10.3f} {metrics['calmar']:<10.3f} ¥{metrics['costs']:<10.2f} {metrics['rebalances']:<12}")

    print("-" * 100)
    print()

    # Recommendation
    print("=" * 100)
    print(" " * 35 + "RECOMMENDATION")
    print("=" * 100)
    print()

    if best_config:
        print(f"✅ **RECOMMENDED CONFIG**: {best_config}")
        print(f"   Sharpe: {best_sharpe:.3f} ({((best_sharpe / baseline_sharpe) - 1) * 100:+.1f}% vs BASE)")
        print()
        print("📋 **PRODUCTION SETTINGS**:")

        if "γ=0.2" in best_config:
            gamma = 0.2
        elif "γ=0.3" in best_config:
            gamma = 0.3
        elif "γ=0.5" in best_config:
            gamma = 0.5
        else:
            gamma = None

        if gamma is not None:
            print(f"   --ei-neutralize-gamma {gamma}")
            print("   --ei-ridge-alpha 10.0")
            print("   --ei-risk-factors \"Sector33Code,volatility_60d\"")

        if "A.3" in best_config:
            print("   --ei-hysteresis-entry-k 35")
            print("   --ei-hysteresis-exit-k 60")
    else:
        print("⚠️  **NO CONFIG MEETS +5% SHARPE THRESHOLD**")
        print("   Continue using BASE (no A.3/A.4)")
        print()
        print("   Consider:")
        print("   1. Expanding risk factors (add beta, size when available)")
        print("   2. Fine-tuning gamma in [0.15, 0.35] range")
        print("   3. Adjusting exit_k threshold [55, 65]")

    print()
    print("=" * 100)

if __name__ == "__main__":
    analyze_experiments()
