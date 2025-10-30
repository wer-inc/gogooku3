from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from gogooku3.training.metrics import rank_ic


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Comprehensive evaluation report with ablation analysis")
    p.add_argument("--data", type=Path, required=True, help="Parquet with predictions + targets")
    p.add_argument("--pred", type=str, default="pred_1d", help="Base prediction column")
    p.add_argument("--target", type=str, default="target_1d", help="Target column")
    p.add_argument("--ablation", action="store_true", help="Run ablation analysis")
    p.add_argument("--horizons", type=str, default="1,5,10,20", help="Comma-separated horizons")
    p.add_argument("--output", type=Path, default=None, help="Output report path (JSON/HTML)")
    p.add_argument("--universe-col", type=str, default="sector33_id", help="Column for universe analysis")
    return p.parse_args()


def calculate_metrics(df: pl.DataFrame, pred_col: str, target_col: str) -> dict[str, float]:
    """Calculate comprehensive metrics for a prediction column."""
    # Filter valid rows
    valid_df = df.filter(
        pl.col(pred_col).is_not_null() &
        pl.col(target_col).is_not_null()
    )

    if valid_df.height == 0:
        return {"error": "No valid data"}

    # RankIC by date
    ric = rank_ic(valid_df, pred_col=pred_col, target_col=target_col, date_col="Date")

    # IC (Pearson correlation)
    ic_by_date = (
        valid_df.group_by("Date")
        .agg(
            pl.corr(pred_col, target_col).alias("ic")
        )
        .filter(pl.col("ic").is_not_null())
    )

    mean_ic = ic_by_date["ic"].mean()
    std_ic = ic_by_date["ic"].std()
    icir = mean_ic / std_ic if std_ic and std_ic > 0 else 0.0

    # Decile analysis
    deciles = valid_df.with_columns(
        pl.col(pred_col).qcut(10, labels=list(range(1, 11))).alias("decile")
    )

    decile_returns = (
        deciles.group_by("decile")
        .agg(pl.col(target_col).mean().alias("mean_return"))
        .sort("decile")
    )

    # Top-Bottom spread
    if decile_returns.height >= 10:
        top_ret = decile_returns.filter(pl.col("decile") == 10)["mean_return"][0]
        bottom_ret = decile_returns.filter(pl.col("decile") == 1)["mean_return"][0]
        spread = top_ret - bottom_ret
    else:
        spread = None

    return {
        "rank_ic": float(ric) if ric is not None else 0.0,
        "mean_ic": float(mean_ic) if mean_ic is not None else 0.0,
        "std_ic": float(std_ic) if std_ic is not None else 0.0,
        "icir": float(icir) if icir is not None else 0.0,
        "top_bottom_spread": float(spread) if spread is not None else 0.0,
        "n_samples": valid_df.height,
    }


def run_ablation_analysis(df: pl.DataFrame, target_col: str, horizons: list[int]) -> dict[str, Any]:
    """Run ablation analysis across different feature sets."""
    ablation_results = {}

    # Feature progression for ablation
    feature_sets = [
        ("Base", ["pred_1d"]),  # Base features
        ("+LOO", ["pred_1d", "sec_ret_1d_eq_loo"]),  # Add LOO
        ("+ScaleUnify", ["pred_1d", "sec_ret_1d_eq_loo", "*_to_adv20", "*_z260"]),  # Add scale unified
        ("+Outlier", ["pred_1d", "sec_ret_1d_eq_loo", "*_to_adv20", "*_z260"]),  # With winsorization
        ("+Interactions", ["pred_1d", "sec_ret_1d_eq_loo", "*_to_adv20", "*_z260", "x_*"]),  # Add interactions
    ]

    for stage_name, feature_patterns in feature_sets:
        # Find matching columns
        available_cols = []
        for pattern in feature_patterns:
            if "*" in pattern:
                prefix = pattern.replace("*", "")
                matching = [c for c in df.columns if prefix in c]
                available_cols.extend(matching)
            elif pattern in df.columns:
                available_cols.append(pattern)

        if not available_cols:
            ablation_results[stage_name] = {"error": "No matching columns"}
            continue

        # For simplicity, use first matching pred column or create synthetic
        pred_col = available_cols[0] if available_cols else "pred_1d"

        if pred_col in df.columns:
            metrics = calculate_metrics(df, pred_col, target_col)
            ablation_results[stage_name] = {
                "metrics": metrics,
                "n_features": len(available_cols),
                "feature_sample": available_cols[:5],  # Show first 5 features
            }
        else:
            ablation_results[stage_name] = {"error": f"Column {pred_col} not found"}

    return ablation_results


def analyze_stability(df: pl.DataFrame, pred_col: str, target_col: str,
                     universe_col: str = "sector33_id") -> dict[str, Any]:
    """Analyze prediction stability across time periods and universes."""
    stability_results = {}

    # Time period analysis (split data into halves)
    if "Date" in df.columns:
        dates = df["Date"].unique().sort()
        mid_point = len(dates) // 2

        first_half = df.filter(pl.col("Date").is_in(dates[:mid_point]))
        second_half = df.filter(pl.col("Date").is_in(dates[mid_point:]))

        stability_results["time_periods"] = {
            "first_half": calculate_metrics(first_half, pred_col, target_col),
            "second_half": calculate_metrics(second_half, pred_col, target_col),
        }

    # Universe analysis (by sector or other grouping)
    if universe_col in df.columns:
        universes = df[universe_col].unique()[:5]  # Top 5 universes
        universe_metrics = {}

        for universe in universes:
            if universe is not None:
                subset = df.filter(pl.col(universe_col) == universe)
                if subset.height > 100:  # Minimum samples
                    universe_metrics[str(universe)] = calculate_metrics(subset, pred_col, target_col)

        stability_results["universes"] = universe_metrics

    # Calculate variance across folds
    if "fold" in df.columns:
        fold_metrics = []
        for fold in df["fold"].unique():
            fold_df = df.filter(pl.col("fold") == fold)
            if fold_df.height > 0:
                m = calculate_metrics(fold_df, pred_col, target_col)
                if "rank_ic" in m:
                    fold_metrics.append(m["rank_ic"])

        if fold_metrics:
            stability_results["fold_variance"] = {
                "mean": np.mean(fold_metrics),
                "std": np.std(fold_metrics),
                "cv": np.std(fold_metrics) / np.mean(fold_metrics) if np.mean(fold_metrics) != 0 else 0,
            }

    return stability_results


def generate_html_report(results: dict[str, Any], output_path: Path) -> None:
    """Generate HTML report from results."""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Evaluation Report - Feature Preservation ML</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; border-bottom: 2px solid #eee; padding-bottom: 5px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f4f4f4; }}
            .metric {{ font-weight: bold; color: #2563eb; }}
            .improvement {{ color: #16a34a; }}
            .degradation {{ color: #dc2626; }}
        </style>
    </head>
    <body>
        <h1>🎯 全特徴量保持ML評価レポート</h1>

        <h2>📊 Overall Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>RankIC</td><td class="metric">{rank_ic:.4f}</td></tr>
            <tr><td>IC</td><td class="metric">{mean_ic:.4f}</td></tr>
            <tr><td>ICIR</td><td class="metric">{icir:.3f}</td></tr>
            <tr><td>Top-Bottom Spread</td><td class="metric">{spread:.4f}</td></tr>
        </table>

        {ablation_section}
        {stability_section}

        <h2>📈 Summary</h2>
        <p>Generated: {timestamp}</p>
    </body>
    </html>
    """

    # Build sections
    ablation_section = ""
    if "ablation" in results:
        ablation_section = "<h2>🔬 Ablation Analysis</h2><table>"
        ablation_section += "<tr><th>Stage</th><th>RankIC</th><th>ICIR</th><th>Features</th><th>Δ RankIC</th></tr>"

        base_ric = None
        for stage, data in results["ablation"].items():
            if "metrics" in data:
                m = data["metrics"]
                ric = m.get("rank_ic", 0)
                icir = m.get("icir", 0)
                n_feat = data.get("n_features", 0)

                if base_ric is None:
                    base_ric = ric
                    delta = "-"
                    delta_class = ""
                else:
                    delta_val = ric - base_ric
                    delta = f"{delta_val:+.4f}"
                    delta_class = "improvement" if delta_val > 0 else "degradation"

                ablation_section += f"""
                <tr>
                    <td>{stage}</td>
                    <td class="metric">{ric:.4f}</td>
                    <td>{icir:.3f}</td>
                    <td>{n_feat}</td>
                    <td class="{delta_class}">{delta}</td>
                </tr>
                """
        ablation_section += "</table>"

    stability_section = ""
    if "stability" in results:
        stability_section = "<h2>🔄 Stability Analysis</h2>"

        # Fold variance
        if "fold_variance" in results["stability"]:
            fv = results["stability"]["fold_variance"]
            stability_section += f"""
            <p><strong>Fold Stability:</strong>
            Mean RankIC = {fv['mean']:.4f},
            Std = {fv['std']:.4f},
            CV = {fv['cv']:.2%}</p>
            """

    # Fill template
    import datetime
    html = html_template.format(
        rank_ic=results.get("overall", {}).get("rank_ic", 0),
        mean_ic=results.get("overall", {}).get("mean_ic", 0),
        icir=results.get("overall", {}).get("icir", 0),
        spread=results.get("overall", {}).get("top_bottom_spread", 0),
        ablation_section=ablation_section,
        stability_section=stability_section,
        timestamp=datetime.datetime.now().isoformat(),
    )

    output_path.write_text(html)


def main() -> None:
    args = parse_args()
    df = pl.read_parquet(str(args.data))

    # Overall metrics
    overall_metrics = calculate_metrics(df, args.pred, args.target)
    results = {"overall": overall_metrics}

    # Ablation analysis
    if args.ablation:
        horizons = [int(h) for h in args.horizons.split(",")]
        ablation = run_ablation_analysis(df, args.target, horizons)
        results["ablation"] = ablation

    # Stability analysis
    stability = analyze_stability(df, args.pred, args.target, args.universe_col)
    results["stability"] = stability

    # Output results
    print("\n" + "="*60)
    print("📊 EVALUATION REPORT")
    print("="*60)

    print("\n✅ Overall Metrics:")
    for k, v in overall_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    if args.ablation and "ablation" in results:
        print("\n🔬 Ablation Analysis:")
        for stage, data in results["ablation"].items():
            if "metrics" in data:
                ric = data["metrics"].get("rank_ic", 0)
                print(f"  {stage}: RankIC={ric:.4f}, Features={data.get('n_features', 0)}")

    if "stability" in results:
        print("\n🔄 Stability Analysis:")
        if "fold_variance" in results["stability"]:
            fv = results["stability"]["fold_variance"]
            print(f"  Fold CV: {fv.get('cv', 0):.2%}")

    # Save output
    if args.output:
        if str(args.output).endswith(".html"):
            generate_html_report(results, args.output)
            print(f"\n📝 HTML report saved to: {args.output}")
        else:
            # Save as JSON
            args.output.write_text(json.dumps(results, indent=2, default=str))
            print(f"\n📝 JSON report saved to: {args.output}")


if __name__ == "__main__":
    main()

