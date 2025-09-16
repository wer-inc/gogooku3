#!/usr/bin/env python3
"""
Experiment Comparison Tool for ATFT-GAT-FAN

Compare multiple training runs to find the best configuration.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


class ExperimentComparison:
    """Compare multiple ATFT-GAT-FAN experiments"""

    def __init__(self, run_dirs: List[Path]):
        """
        Initialize with multiple run directories

        Args:
            run_dirs: List of training run directories
        """
        self.run_dirs = [Path(d) for d in run_dirs]
        self.experiments = self._load_experiments()

    def _load_experiments(self) -> Dict:
        """Load metrics from all experiments"""
        experiments = {}

        for run_dir in self.run_dirs:
            if not run_dir.exists():
                continue

            exp_name = run_dir.name
            exp_data = {"path": run_dir}

            # Load metrics
            metrics_path = run_dir / "metrics_summary.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    exp_data["metrics"] = json.load(f)

            # Load config
            config_path = run_dir / "config.yaml"
            if config_path.exists():
                import yaml

                with open(config_path) as f:
                    exp_data["config"] = yaml.safe_load(f)

            experiments[exp_name] = exp_data

        return experiments

    def compare_final_metrics(self) -> pd.DataFrame:
        """Compare final metrics across experiments"""
        comparison = []

        for exp_name, exp_data in self.experiments.items():
            if "metrics" not in exp_data:
                continue

            metrics = exp_data["metrics"]
            row = {"experiment": exp_name}

            # Extract final values
            for key in ["val_loss", "val_ic_1d", "val_ic_5d", "val_rank_ic_1d", "val_rank_ic_5d", "val_sharpe"]:
                if key in metrics:
                    values = metrics[key]
                    if isinstance(values, list) and values:
                        row[key] = values[-1]
                    else:
                        row[key] = values

            comparison.append(row)

        df = pd.DataFrame(comparison)
        return df.sort_values("val_sharpe", ascending=False) if "val_sharpe" in df.columns else df

    def plot_metric_comparison(self, metric_key: str = "val_loss"):
        """Plot a specific metric across all experiments"""
        fig = go.Figure()

        for exp_name, exp_data in self.experiments.items():
            if "metrics" not in exp_data:
                continue

            metrics = exp_data["metrics"]
            if metric_key in metrics:
                values = metrics[metric_key]
                if isinstance(values, list):
                    epochs = list(range(len(values)))
                    fig.add_trace(
                        go.Scatter(
                            x=epochs,
                            y=values,
                            mode="lines",
                            name=exp_name,
                        )
                    )

        fig.update_layout(
            title=f"Comparison: {metric_key}",
            xaxis_title="Epoch",
            yaxis_title=metric_key,
            height=500,
        )

        return fig

    def plot_comprehensive_comparison(self):
        """Create comprehensive comparison dashboard"""
        # Create subplots
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Validation Loss",
                "IC (1d)",
                "RankIC (1d)",
                "RankIC (5d)",
                "Sharpe Ratio",
                "Best Metrics Summary",
            ),
        )

        # Metrics to plot
        metrics_to_plot = [
            ("val_loss", 1, 1),
            ("val_ic_1d", 1, 2),
            ("val_rank_ic_1d", 2, 1),
            ("val_rank_ic_5d", 2, 2),
            ("val_sharpe", 3, 1),
        ]

        # Plot each metric
        for metric_key, row, col in metrics_to_plot:
            for exp_name, exp_data in self.experiments.items():
                if "metrics" not in exp_data:
                    continue

                metrics = exp_data["metrics"]
                if metric_key in metrics:
                    values = metrics[metric_key]
                    if isinstance(values, list):
                        epochs = list(range(len(values)))
                        fig.add_trace(
                            go.Scatter(
                                x=epochs,
                                y=values,
                                mode="lines",
                                name=exp_name,
                                showlegend=(row == 1 and col == 1),
                            ),
                            row=row,
                            col=col,
                        )

        # Add summary table
        summary_df = self.compare_final_metrics()
        if not summary_df.empty:
            # Create bar chart of final Sharpe ratios
            if "val_sharpe" in summary_df.columns:
                fig.add_trace(
                    go.Bar(
                        x=summary_df["experiment"],
                        y=summary_df["val_sharpe"],
                        name="Final Sharpe",
                        showlegend=False,
                    ),
                    row=3,
                    col=2,
                )

        fig.update_layout(height=1000, title_text="Experiment Comparison Dashboard")
        return fig

    def statistical_significance(self) -> pd.DataFrame:
        """Perform statistical significance tests between experiments"""
        # This would require multiple seeds/folds per experiment
        # Placeholder for now
        return pd.DataFrame()

    def find_best_experiment(self, metric: str = "val_sharpe") -> str:
        """Find the best experiment based on a metric"""
        best_exp = None
        best_value = float("-inf") if "loss" not in metric else float("inf")

        for exp_name, exp_data in self.experiments.items():
            if "metrics" not in exp_data:
                continue

            metrics = exp_data["metrics"]
            if metric in metrics:
                values = metrics[metric]
                if isinstance(values, list) and values:
                    value = values[-1]
                else:
                    value = values

                if "loss" in metric:
                    if value < best_value:
                        best_value = value
                        best_exp = exp_name
                else:
                    if value > best_value:
                        best_value = value
                        best_exp = exp_name

        return best_exp

    def generate_report(self, output_path: Path):
        """Generate comprehensive comparison report"""
        report = []
        report.append("# Experiment Comparison Report\n")

        # Summary table
        report.append("## Summary\n")
        summary_df = self.compare_final_metrics()
        report.append(summary_df.to_markdown())
        report.append("\n")

        # Best experiment
        best_exp = self.find_best_experiment()
        report.append(f"## Best Experiment: {best_exp}\n")

        # Hyperparameter differences
        report.append("## Hyperparameter Comparison\n")
        for exp_name, exp_data in self.experiments.items():
            if "config" in exp_data:
                config = exp_data["config"]
                report.append(f"### {exp_name}\n")
                report.append(f"- Learning Rate: {config.get('train', {}).get('learning_rate', 'N/A')}\n")
                report.append(f"- Batch Size: {config.get('train', {}).get('batch_size', 'N/A')}\n")
                report.append(f"- Model: {config.get('model', {}).get('name', 'N/A')}\n")

        # Save report
        with open(output_path, "w") as f:
            f.writelines(report)

        # Save plots
        fig = self.plot_comprehensive_comparison()
        fig.write_html(output_path.with_suffix(".html"))

        print(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare ATFT-GAT-FAN experiments")
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="List of run directories to compare",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comparison_report.md",
        help="Output report path",
    )

    args = parser.parse_args()

    comparison = ExperimentComparison(args.run_dirs)

    # Generate report
    comparison.generate_report(Path(args.output))

    # Print summary
    print("\nFinal Metrics Comparison:")
    print(comparison.compare_final_metrics())

    best = comparison.find_best_experiment()
    print(f"\nBest experiment (by Sharpe): {best}")


if __name__ == "__main__":
    main()