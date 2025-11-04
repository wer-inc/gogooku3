#!/usr/bin/env python3
"""Backtest results analyzer with supply guard verification.

Usage:
    python analyze_backtest_results.py results/backtest_v0_latest_filled_wk5_VERIFIED.json
"""

import json
import sys
from pathlib import Path

import numpy as np


def analyze_backtest(result_path: Path) -> dict:
    """Analyze backtest results with 5-item minimal check.

    Returns:
        Dictionary with analysis results including:
        - Summary metrics (return, Sharpe, drawdown)
        - Supply guard verification
        - Daily PnL health snapshot
        - Cost breakdown
    """
    with result_path.open() as f:
        data = json.load(f)

    summary = data["summary"]
    daily = data.get("daily", [])

    # === 1. Supply Guard Verification ===
    selected_counts = [d.get("selected_count", 0) for d in daily if "selected_count" in d]
    fallback_flags = [d.get("fallback_used", False) for d in daily if "fallback_used" in d]

    if selected_counts:
        sc_p10, sc_median, sc_p90 = np.percentile(selected_counts, [10, 50, 90])
        sc_min = min(selected_counts)
        supply_guard_violations = sum(1 for sc in selected_counts if sc < 53)
        supply_guard_rate = 1.0 - (supply_guard_violations / len(selected_counts))
    else:
        sc_p10 = sc_median = sc_p90 = sc_min = 0
        supply_guard_violations = 0
        supply_guard_rate = 0.0

    if fallback_flags:
        fallback_rate = sum(fallback_flags) / len(fallback_flags)
    else:
        fallback_rate = 0.0

    # === 2. Summary Metrics ===
    total_return = summary["cumulative_return"]
    sharpe = summary["sharpe_ratio"]
    max_dd = summary["max_drawdown"]
    trade_count = summary.get("trade_count", 0)

    # === 3. Daily PnL Health Snapshot ===
    daily_returns = [d.get("return", 0.0) for d in daily if "return" in d]
    if daily_returns:
        daily_rets_sorted = sorted(enumerate(daily_returns), key=lambda x: x[1])
        top10_worst = daily_rets_sorted[:10]
        top10_best = daily_rets_sorted[-10:]

        extreme_days = [
            (i, ret, daily[i].get("date", "unknown"))
            for i, ret in daily_rets_sorted
            if abs(ret) > 0.15
        ]
    else:
        top10_worst = top10_best = extreme_days = []

    # === 4. Cost Breakdown ===
    total_costs = [d.get("cost", 0.0) for d in daily if "cost" in d]
    turnovers = [d.get("turnover", 0.0) for d in daily if "turnover" in d]

    if total_costs:
        total_cost_sum = sum(total_costs)
        avg_cost_per_rebalance = np.mean([c for c in total_costs if c > 0])
    else:
        total_cost_sum = avg_cost_per_rebalance = 0.0

    if turnovers:
        avg_turnover = np.mean([t for t in turnovers if t > 0])
    else:
        avg_turnover = 0.0

    # === 5. Portfolio Stability (Jaccard Similarity) ===
    def jaccard_similarity(set_a, set_b):
        """Calculate Jaccard similarity between two sets."""
        if not set_a and not set_b:
            return 1.0
        set_a, set_b = set(set_a), set(set_b)
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    # Extract holdings from portfolio_history if available
    jaccard_similarities = []
    if "portfolio_history" in data:
        prev_holdings = None
        for ph in data["portfolio_history"]:
            if "holdings" in ph and ph["holdings"]:
                current_holdings = [h.get("code") for h in ph["holdings"] if "code" in h]
                if prev_holdings is not None and current_holdings:
                    jacc = jaccard_similarity(prev_holdings, current_holdings)
                    jaccard_similarities.append(jacc)
                prev_holdings = current_holdings

    if jaccard_similarities:
        avg_jaccard = np.mean(jaccard_similarities)
        median_jaccard = np.median(jaccard_similarities)
        jaccard_stability = "High" if avg_jaccard > 0.65 else "Medium" if avg_jaccard > 0.45 else "Low"
    else:
        avg_jaccard = median_jaccard = 0.0
        jaccard_stability = "N/A"

    # === Build Analysis Report ===
    analysis = {
        "summary": {
            "total_return_pct": total_return * 100,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_dd * 100,
            "trade_count": trade_count,
        },
        "supply_guard": {
            "selected_count_p10": sc_p10,
            "selected_count_median": sc_median,
            "selected_count_p90": sc_p90,
            "selected_count_min": sc_min,
            "supply_guard_rate_pct": supply_guard_rate * 100,
            "violations": supply_guard_violations,
            "fallback_rate_pct": fallback_rate * 100,
            "status": "PASS" if supply_guard_rate >= 0.99 and fallback_rate < 0.20 else "FAIL",
        },
        "pnl_health": {
            "top10_worst_days": [
                {"index": i, "return_pct": ret * 100, "date": daily[i].get("date", "unknown")}
                for i, ret in top10_worst
            ],
            "top10_best_days": [
                {"index": i, "return_pct": ret * 100, "date": daily[i].get("date", "unknown")}
                for i, ret in top10_best
            ],
            "extreme_days_count": len(extreme_days),
            "extreme_days": [
                {"index": i, "return_pct": ret * 100, "date": date}
                for i, ret, date in extreme_days
            ],
        },
        "costs": {
            "total_cost_jpy": total_cost_sum,
            "avg_cost_per_rebalance_jpy": avg_cost_per_rebalance,
            "avg_turnover_pct": avg_turnover * 100,
            "cost_to_capital_pct": (total_cost_sum / 10_000_000) * 100,
        },
        "portfolio_stability": {
            "avg_jaccard": avg_jaccard,
            "median_jaccard": median_jaccard,
            "stability_class": jaccard_stability,
            "num_rebalances": len(jaccard_similarities),
        },
    }

    return analysis


def print_analysis(analysis: dict) -> None:
    """Pretty-print analysis results."""
    print("=" * 80)
    print("BACKTEST ANALYSIS REPORT")
    print("=" * 80)
    print()

    # Summary
    print("📊 SUMMARY METRICS")
    print("-" * 80)
    s = analysis["summary"]
    print(f"  Total Return:    {s['total_return_pct']:>8.2f}%")
    print(f"  Sharpe Ratio:    {s['sharpe_ratio']:>8.3f}")
    print(f"  Max Drawdown:    {s['max_drawdown_pct']:>8.2f}%")
    print(f"  Trade Count:     {s['trade_count']:>8,}")
    print()

    # Supply Guard
    print("🛡️  SUPPLY GUARD VERIFICATION")
    print("-" * 80)
    sg = analysis["supply_guard"]
    print(f"  Selected Count (p10/median/p90): {sg['selected_count_p10']:.0f} / {sg['selected_count_median']:.0f} / {sg['selected_count_p90']:.0f}")
    print(f"  Selected Count (min):            {sg['selected_count_min']:.0f}")
    print(f"  Supply Guard Rate (≥53):         {sg['supply_guard_rate_pct']:.1f}% ({sg['violations']} violations)")
    print(f"  Fallback Rate:                   {sg['fallback_rate_pct']:.1f}%")
    print(f"  Status:                          {sg['status']}")
    print()

    # PnL Health
    print("📈 PNL HEALTH SNAPSHOT")
    print("-" * 80)
    ph = analysis["pnl_health"]
    print(f"  Extreme Days (|ret| > 15%):      {ph['extreme_days_count']}")
    print()
    print("  Top 5 Worst Days:")
    for day in ph["top10_worst_days"][:5]:
        print(f"    {day['date']:>10s}  Return: {day['return_pct']:>7.2f}%")
    print()
    print("  Top 5 Best Days:")
    for day in ph["top10_best_days"][-5:]:
        print(f"    {day['date']:>10s}  Return: {day['return_pct']:>7.2f}%")
    print()

    # Costs
    print("💰 COST BREAKDOWN")
    print("-" * 80)
    c = analysis["costs"]
    print(f"  Total Cost:                      ¥{c['total_cost_jpy']:>12,.0f}")
    print(f"  Avg Cost per Rebalance:          ¥{c['avg_cost_per_rebalance_jpy']:>12,.0f}")
    print(f"  Avg Turnover:                    {c['avg_turnover_pct']:>8.2f}%")
    print(f"  Cost to Initial Capital:         {c['cost_to_capital_pct']:>8.2f}%")
    print()

    # Portfolio Stability
    print("📦 PORTFOLIO STABILITY")
    print("-" * 80)
    ps = analysis["portfolio_stability"]
    print(f"  Avg Jaccard Similarity:          {ps['avg_jaccard']:>8.3f}")
    print(f"  Median Jaccard Similarity:       {ps['median_jaccard']:>8.3f}")
    print(f"  Stability Class:                 {ps['stability_class']}")
    print(f"  Number of Rebalances:            {ps['num_rebalances']:>8,}")
    print()
    print("=" * 80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_backtest_results.py <result.json>")
        sys.exit(1)

    result_path = Path(sys.argv[1])
    if not result_path.exists():
        print(f"Error: File not found: {result_path}")
        sys.exit(1)

    analysis = analyze_backtest(result_path)
    print_analysis(analysis)

    # Save analysis to JSON
    output_path = result_path.with_suffix(".analysis.json")
    with output_path.open("w") as f:
        json.dump(analysis, f, indent=2)
    print(f"📁 Analysis saved to: {output_path}")


if __name__ == "__main__":
    main()
