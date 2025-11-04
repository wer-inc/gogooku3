#!/usr/bin/env python3
"""A.4 SAFE バックテスト結果比較スクリプト

Usage:
    python scripts/compare_a4_results.py
"""

import json
from pathlib import Path

import pandas as pd


def load_result(path: str) -> dict:
    """Load backtest result JSON file."""
    with open(path) as f:
        data = json.load(f)
    return data


def extract_metrics(data: dict) -> dict:
    """Extract key metrics from backtest result."""
    perf = data.get("performance", {})
    summary = data.get("summary", {})

    return {
        "total_return": perf.get("total_return", float("nan")),
        "ann_return": perf.get("annualized_return", float("nan")),
        "sharpe_ratio": perf.get("sharpe_ratio", float("nan")),
        "sortino_ratio": perf.get("sortino_ratio", float("nan")),
        "max_drawdown": perf.get("max_drawdown", float("nan")),
        "calmar_ratio": perf.get("calmar_ratio", float("nan")),
        "win_rate": perf.get("win_rate", float("nan")),
        "avg_turnover": perf.get("avg_turnover", float("nan")),
        "total_trades": summary.get("total_trades", float("nan")),
        "cost_pct": perf.get("transaction_costs", {}).get("cost_pct_of_pv", float("nan")),
    }


def main():
    """Compare A.4 SAFE backtest results."""
    results_dir = Path("results")

    # Define test configurations
    tests = {
        "BASE": "bt_enhanced_monthly_h20_BASE.json",
        "A4_g02": "bt_a4safe_g02_a10_monthly_h20_k35.json",
        "A4_g03": "bt_a4safe_g03_a10_monthly_h20_k35.json",
        "A4_g05": "bt_a4safe_g05_a10_monthly_h20_k35.json",
        "A3A4_g03": "bt_a3_a4safe_g03_a10_monthly_h20_k35.json",
    }

    # Load results
    rows = []
    for tag, filename in tests.items():
        path = results_dir / filename
        if not path.exists():
            print(f"⚠️  Missing: {path}")
            continue

        try:
            data = load_result(path)
            metrics = extract_metrics(data)
            metrics["tag"] = tag
            rows.append(metrics)
        except Exception as e:
            print(f"❌ Error loading {path}: {e}")
            continue

    if not rows:
        print("❌ No results found. Check that backtests have completed.")
        return

    # Create DataFrame
    df = pd.DataFrame(rows)
    df = df.set_index("tag")

    # Reorder columns
    cols = ["sharpe_ratio", "total_return", "ann_return", "max_drawdown",
            "sortino_ratio", "calmar_ratio", "win_rate", "avg_turnover",
            "total_trades", "cost_pct"]
    df = df[[c for c in cols if c in df.columns]]

    # Print results
    print("\n" + "="*80)
    print("A.4 SAFE バックテスト比較結果 (2024-01-01 to 2025-10-31)")
    print("="*80)
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))

    # Calculate delta vs BASE
    if "BASE" in df.index:
        print("\n" + "="*80)
        print("Δ vs BASE")
        print("="*80)
        delta = df.drop(index="BASE") - df.loc["BASE"]
        print(delta.to_string(float_format=lambda x: f"{x:+.4f}"))

        # Calculate percentage change for key metrics
        print("\n" + "="*80)
        print("Δ% vs BASE")
        print("="*80)
        pct_change = ((df.drop(index="BASE") / df.loc["BASE"]) - 1) * 100
        print(pct_change.to_string(float_format=lambda x: f"{x:+.2f}%"))

    # GO/NO-GO decision
    print("\n" + "="*80)
    print("GO基準判定")
    print("="*80)

    if "BASE" not in df.index:
        print("⚠️  BASE結果が見つかりません。判定不可。")
        return

    base_sharpe = df.loc["BASE", "sharpe_ratio"]
    base_dd = df.loc["BASE", "max_drawdown"]

    for tag in df.index:
        if tag == "BASE":
            continue

        sharpe = df.loc[tag, "sharpe_ratio"]
        dd = df.loc[tag, "max_drawdown"]

        sharpe_delta_pct = ((sharpe / base_sharpe) - 1) * 100
        dd_delta = dd - base_dd

        # GO criteria
        go_sharpe = sharpe_delta_pct >= 10  # +10% Sharpe improvement
        go_dd = dd_delta <= 5  # Max DD increase ≤ +5pp

        status = "✅ GO" if (go_sharpe and go_dd) else "❌ NO-GO"
        reasons = []
        if not go_sharpe:
            reasons.append(f"Sharpe {sharpe_delta_pct:+.1f}% < +10%")
        if not go_dd:
            reasons.append(f"MaxDD {dd_delta:+.1f}pp > +5pp")

        reason_str = " (" + ", ".join(reasons) + ")" if reasons else ""
        print(f"{tag:15s}: {status}{reason_str}")
        print(f"               Sharpe: {sharpe:.4f} ({sharpe_delta_pct:+.1f}%)")
        print(f"               MaxDD:  {dd:.2f}% ({dd_delta:+.1f}pp)")
        print()


if __name__ == "__main__":
    main()
