#!/usr/bin/env python3
"""
Deep Learning Training Visualization Tool for ATFT-GAT-FAN

Provides comprehensive visualization for:
- Training metrics (loss, IC, RankIC, Sharpe)
- Model predictions vs actual returns
- Feature importance and attention weights
- Portfolio performance analysis
- Cross-sectional and time-series analysis
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import seaborn as sns
from plotly.subplots import make_subplots

# Set style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ATFTVisualization:
    """Visualization tool for ATFT-GAT-FAN training results"""

    def __init__(self, run_dir: Path):
        """
        Initialize visualization with run directory

        Args:
            run_dir: Path to training run directory (e.g., runs/last)
        """
        self.run_dir = Path(run_dir)
        self.metrics = self._load_metrics()
        self.predictions = self._load_predictions()
        self.config = self._load_config()

    def _load_metrics(self) -> dict:
        """Load metrics from JSON files"""
        metrics_files = [
            "metrics_summary.json",
            "train_metrics.json",
            "val_metrics.json",
        ]

        metrics = {}
        for file in metrics_files:
            path = self.run_dir / file
            if path.exists():
                with open(path) as f:
                    metrics[file.replace(".json", "")] = json.load(f)
                logger.info(f"Loaded {file}")

        return metrics

    def _load_predictions(self) -> pl.DataFrame | None:
        """Load predictions parquet if available"""
        pred_path = self.run_dir / "predictions_val.parquet"
        if pred_path.exists():
            logger.info("Loaded predictions_val.parquet")
            return pl.read_parquet(pred_path)
        return None

    def _load_config(self) -> dict:
        """Load training configuration"""
        config_path = self.run_dir / "config.yaml"
        if config_path.exists():
            import yaml

            with open(config_path) as f:
                return yaml.safe_load(f)
        return {}

    def plot_training_curves(self, save_path: Path | None = None):
        """Plot training curves with multiple metrics"""
        if "metrics_summary" not in self.metrics:
            logger.warning("No metrics_summary found")
            return

        metrics_data = self.metrics["metrics_summary"]

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Loss Curves",
                "IC (Information Coefficient)",
                "RankIC",
                "Sharpe Ratio",
            ),
        )

        # Extract epochs and metrics
        epochs = list(range(len(metrics_data.get("train_loss", []))))

        # Plot Loss
        if "train_loss" in metrics_data:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=metrics_data["train_loss"],
                    mode="lines",
                    name="Train Loss",
                    line={"color": "blue"},
                ),
                row=1,
                col=1,
            )
        if "val_loss" in metrics_data:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=metrics_data["val_loss"],
                    mode="lines",
                    name="Val Loss",
                    line={"color": "red"},
                ),
                row=1,
                col=1,
            )

        # Plot IC
        for horizon in [1, 5, 10, 20]:
            train_key = f"train_ic_{horizon}d"
            val_key = f"val_ic_{horizon}d"

            if train_key in metrics_data:
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=metrics_data[train_key],
                        mode="lines",
                        name=f"Train IC {horizon}d",
                        line={"dash": "solid"},
                    ),
                    row=1,
                    col=2,
                )

            if val_key in metrics_data:
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=metrics_data[val_key],
                        mode="lines",
                        name=f"Val IC {horizon}d",
                        line={"dash": "dash"},
                    ),
                    row=1,
                    col=2,
                )

        # Plot RankIC
        for horizon in [1, 5]:
            val_key = f"val_rank_ic_{horizon}d"
            if val_key in metrics_data:
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=metrics_data[val_key],
                        mode="lines",
                        name=f"RankIC {horizon}d",
                    ),
                    row=2,
                    col=1,
                )

        # Plot Sharpe
        if "val_sharpe" in metrics_data:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=metrics_data["val_sharpe"],
                    mode="lines",
                    name="Validation Sharpe",
                    line={"color": "green", "width": 2},
                ),
                row=2,
                col=2,
            )

        # Update layout
        fig.update_layout(
            height=800,
            title_text="ATFT-GAT-FAN Training Metrics",
            showlegend=True,
        )
        fig.update_xaxes(title_text="Epoch")

        # Save or show
        if save_path:
            fig.write_html(save_path / "training_curves.html")
            logger.info(f"Saved training curves to {save_path}/training_curves.html")
        else:
            fig.show()

    def plot_predictions_analysis(self, save_path: Path | None = None):
        """Analyze predictions vs actual returns"""
        if self.predictions is None:
            logger.warning("No predictions available")
            return

        # Convert to pandas for easier plotting
        df = self.predictions.to_pandas()

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Predicted vs Actual (1d)",
                "Prediction Distribution",
                "Residuals Analysis",
                "Quantile Performance",
            ),
        )

        # Scatter plot: predicted vs actual
        if "pred_1d" in df.columns and "target_1d" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["target_1d"],
                    y=df["pred_1d"],
                    mode="markers",
                    marker={"size": 3, "opacity": 0.5},
                    name="Predictions",
                ),
                row=1,
                col=1,
            )

            # Add perfect prediction line
            min_val = min(df["target_1d"].min(), df["pred_1d"].min())
            max_val = max(df["target_1d"].max(), df["pred_1d"].max())
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    line={"color": "red", "dash": "dash"},
                    name="Perfect Prediction",
                ),
                row=1,
                col=1,
            )

        # Distribution comparison
        fig.add_trace(
            go.Histogram(
                x=df["pred_1d"],
                name="Predicted",
                opacity=0.7,
                nbinsx=50,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Histogram(
                x=df["target_1d"],
                name="Actual",
                opacity=0.7,
                nbinsx=50,
            ),
            row=1,
            col=2,
        )

        # Residuals
        residuals = df["target_1d"] - df["pred_1d"]
        fig.add_trace(
            go.Histogram(
                x=residuals,
                name="Residuals",
                nbinsx=50,
            ),
            row=2,
            col=1,
        )

        # Quantile performance
        df["pred_quintile"] = pd.qcut(df["pred_1d"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        quintile_returns = df.groupby("pred_quintile")["target_1d"].mean()

        fig.add_trace(
            go.Bar(
                x=quintile_returns.index,
                y=quintile_returns.values,
                name="Mean Return by Quintile",
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            height=800,
            title_text="Prediction Analysis",
            showlegend=True,
        )

        if save_path:
            fig.write_html(save_path / "predictions_analysis.html")
            logger.info(f"Saved predictions analysis to {save_path}/predictions_analysis.html")
        else:
            fig.show()

    def plot_portfolio_performance(self, save_path: Path | None = None):
        """Analyze portfolio performance based on predictions"""
        if self.predictions is None:
            logger.warning("No predictions available")
            return

        df = self.predictions.to_pandas()

        # Sort by date for time series analysis
        if "date" in df.columns:
            df = df.sort_values("date")

            # Calculate daily portfolio returns (long top quintile, short bottom quintile)
            df["pred_quintile"] = pd.qcut(
                df.groupby("date")["pred_1d"].transform(lambda x: x),
                5,
                labels=False,
                duplicates="drop",
            )

            # Long-short portfolio
            long_returns = df[df["pred_quintile"] == 4].groupby("date")["target_1d"].mean()
            short_returns = df[df["pred_quintile"] == 0].groupby("date")["target_1d"].mean()
            ls_returns = long_returns - short_returns

            # Cumulative returns
            cumulative_ls = (1 + ls_returns).cumprod()

            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Long-Short Portfolio Cumulative Returns",
                    "Daily Returns Distribution",
                    "Rolling Sharpe Ratio (20d)",
                    "Drawdown Analysis",
                ),
            )

            # Cumulative returns
            fig.add_trace(
                go.Scatter(
                    x=cumulative_ls.index,
                    y=cumulative_ls.values,
                    mode="lines",
                    name="Long-Short Portfolio",
                    line={"color": "green"},
                ),
                row=1,
                col=1,
            )

            # Returns distribution
            fig.add_trace(
                go.Histogram(
                    x=ls_returns.values,
                    name="L/S Returns",
                    nbinsx=50,
                ),
                row=1,
                col=2,
            )

            # Rolling Sharpe
            rolling_sharpe = (
                ls_returns.rolling(20).mean() / ls_returns.rolling(20).std() * np.sqrt(252)
            )
            fig.add_trace(
                go.Scatter(
                    x=rolling_sharpe.index,
                    y=rolling_sharpe.values,
                    mode="lines",
                    name="Rolling Sharpe",
                ),
                row=2,
                col=1,
            )

            # Drawdown
            running_max = cumulative_ls.expanding().max()
            drawdown = (cumulative_ls - running_max) / running_max
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values * 100,
                    mode="lines",
                    fill="tozeroy",
                    name="Drawdown (%)",
                    line={"color": "red"},
                ),
                row=2,
                col=2,
            )

            fig.update_layout(
                height=800,
                title_text="Portfolio Performance Analysis",
                showlegend=True,
            )

            # Calculate statistics
            sharpe = ls_returns.mean() / ls_returns.std() * np.sqrt(252)
            max_dd = drawdown.min()
            win_rate = (ls_returns > 0).mean()

            print("\n📊 Portfolio Statistics:")
            print(f"  Sharpe Ratio: {sharpe:.3f}")
            print(f"  Max Drawdown: {max_dd:.1%}")
            print(f"  Win Rate: {win_rate:.1%}")
            print(f"  Mean Daily Return: {ls_returns.mean():.4f}")
            print(f"  Daily Volatility: {ls_returns.std():.4f}")

            if save_path:
                fig.write_html(save_path / "portfolio_performance.html")
                logger.info(
                    f"Saved portfolio performance to {save_path}/portfolio_performance.html"
                )
            else:
                fig.show()

    def plot_attention_weights(self, save_path: Path | None = None):
        """Visualize GAT attention weights if available"""
        attention_path = self.run_dir / "attention_weights.npy"

        if not attention_path.exists():
            logger.warning("No attention weights found")
            return

        attention = np.load(attention_path)

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=attention[:50, :50],  # Show first 50x50 for clarity
                colorscale="Viridis",
                colorbar={"title": "Attention Weight"},
            )
        )

        fig.update_layout(
            title="Graph Attention Network - Attention Weights",
            xaxis_title="Target Stock",
            yaxis_title="Source Stock",
            height=600,
        )

        if save_path:
            fig.write_html(save_path / "attention_weights.html")
            logger.info(f"Saved attention weights to {save_path}/attention_weights.html")
        else:
            fig.show()

    def create_dashboard(self, save_path: Path | None = None):
        """Create comprehensive interactive dashboard"""
        import dash_bootstrap_components as dbc
        from dash import Dash, html

        app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        # Generate all plots
        output_dir = save_path or Path("visualization_output")
        output_dir.mkdir(exist_ok=True)

        self.plot_training_curves(output_dir)
        self.plot_predictions_analysis(output_dir)
        self.plot_portfolio_performance(output_dir)
        self.plot_attention_weights(output_dir)

        # Create dashboard layout
        app.layout = dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.H1(
                                "ATFT-GAT-FAN Training Visualization Dashboard",
                                className="text-center mb-4",
                            ),
                            width=12,
                        )
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H4("Training Metrics"),
                                            html.Iframe(
                                                src=f"file://{output_dir}/training_curves.html",
                                                style={"width": "100%", "height": "500px"},
                                            ),
                                        ]
                                    )
                                ]
                            ),
                            width=6,
                        ),
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H4("Prediction Analysis"),
                                            html.Iframe(
                                                src=f"file://{output_dir}/predictions_analysis.html",
                                                style={"width": "100%", "height": "500px"},
                                            ),
                                        ]
                                    )
                                ]
                            ),
                            width=6,
                        ),
                    ],
                    className="mb-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H4("Portfolio Performance"),
                                            html.Iframe(
                                                src=f"file://{output_dir}/portfolio_performance.html",
                                                style={"width": "100%", "height": "500px"},
                                            ),
                                        ]
                                    )
                                ]
                            ),
                            width=12,
                        ),
                    ]
                ),
            ],
            fluid=True,
        )

        logger.info("Dashboard created. Access at http://localhost:8050")
        return app


def main():
    parser = argparse.ArgumentParser(description="Visualize ATFT-GAT-FAN training results")
    parser.add_argument(
        "--run-dir",
        type=str,
        default="runs/last",
        help="Path to training run directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualization_output",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch interactive dashboard",
    )

    args = parser.parse_args()

    viz = ATFTVisualization(Path(args.run_dir))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if args.dashboard:
        app = viz.create_dashboard(output_dir)
        app.run_server(debug=True, port=8050)
    else:
        # Generate all visualizations
        viz.plot_training_curves(output_dir)
        viz.plot_predictions_analysis(output_dir)
        viz.plot_portfolio_performance(output_dir)
        viz.plot_attention_weights(output_dir)

        logger.info(f"✅ All visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
