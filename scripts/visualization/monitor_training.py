#!/usr/bin/env python3
"""
Real-time Training Monitor for ATFT-GAT-FAN

Monitors training progress in real-time with:
- Live metric tracking
- TensorBoard integration
- Weights & Biases (W&B) integration
- Resource usage monitoring
- Alert system for anomalies
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import psutil
import torch
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()


class TrainingMonitor:
    """Real-time training monitor with rich display"""

    def __init__(self, run_dir: Path, refresh_rate: float = 1.0):
        """
        Initialize monitor

        Args:
            run_dir: Training run directory
            refresh_rate: Refresh rate in seconds
        """
        self.run_dir = Path(run_dir)
        self.refresh_rate = refresh_rate
        self.metrics_history = []
        self.start_time = time.time()

    def read_latest_metrics(self) -> Dict:
        """Read latest metrics from files"""
        metrics = {}

        # Check various metric files
        metrics_files = {
            "train_metrics.json": "train",
            "val_metrics.json": "val",
            "metrics_summary.json": "summary",
        }

        for file, key in metrics_files.items():
            path = self.run_dir / file
            if path.exists():
                try:
                    with open(path) as f:
                        metrics[key] = json.load(f)
                except:
                    pass

        return metrics

    def get_system_stats(self) -> Dict:
        """Get system resource usage"""
        stats = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": psutil.virtual_memory().used / (1024**3),
        }

        # GPU stats if available
        if torch.cuda.is_available():
            stats["gpu_memory_percent"] = (
                torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
            )
            stats["gpu_memory_used_gb"] = torch.cuda.memory_allocated() / (1024**3)
            stats["gpu_utilization"] = "N/A"  # Would need nvidia-ml-py

        return stats

    def create_metrics_table(self, metrics: Dict) -> Table:
        """Create metrics display table"""
        table = Table(title="Training Metrics", expand=True)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Train", style="magenta")
        table.add_column("Validation", style="green")

        # Extract key metrics
        if "summary" in metrics:
            summary = metrics["summary"]

            # Loss
            train_loss = summary.get("train_loss", [])[-1] if "train_loss" in summary else "N/A"
            val_loss = summary.get("val_loss", [])[-1] if "val_loss" in summary else "N/A"
            table.add_row("Loss", f"{train_loss:.4f}" if isinstance(train_loss, float) else str(train_loss),
                         f"{val_loss:.4f}" if isinstance(val_loss, float) else str(val_loss))

            # IC metrics
            for horizon in [1, 5, 10, 20]:
                train_ic = summary.get(f"train_ic_{horizon}d", [])
                val_ic = summary.get(f"val_ic_{horizon}d", [])

                if train_ic:
                    train_val = train_ic[-1] if isinstance(train_ic, list) else train_ic
                else:
                    train_val = "N/A"

                if val_ic:
                    val_val = val_ic[-1] if isinstance(val_ic, list) else val_ic
                else:
                    val_val = "N/A"

                table.add_row(
                    f"IC {horizon}d",
                    f"{train_val:.4f}" if isinstance(train_val, float) else str(train_val),
                    f"{val_val:.4f}" if isinstance(val_val, float) else str(val_val),
                )

            # RankIC
            for horizon in [1, 5]:
                val_rankic = summary.get(f"val_rank_ic_{horizon}d", [])
                if val_rankic:
                    val_val = val_rankic[-1] if isinstance(val_rankic, list) else val_rankic
                    table.add_row(f"RankIC {horizon}d", "N/A",
                                 f"{val_val:.4f}" if isinstance(val_val, float) else str(val_val))

            # Sharpe
            val_sharpe = summary.get("val_sharpe", [])
            if val_sharpe:
                sharpe_val = val_sharpe[-1] if isinstance(val_sharpe, list) else val_sharpe
                table.add_row("Sharpe Ratio", "N/A",
                             f"{sharpe_val:.3f}" if isinstance(sharpe_val, float) else str(sharpe_val))

        return table

    def create_system_panel(self, stats: Dict) -> Panel:
        """Create system stats panel"""
        content = ""

        # CPU and Memory
        content += f"[cyan]CPU Usage:[/cyan] {stats['cpu_percent']:.1f}%\n"
        content += f"[cyan]Memory:[/cyan] {stats['memory_used_gb']:.1f}GB ({stats['memory_percent']:.1f}%)\n"

        # GPU if available
        if "gpu_memory_percent" in stats:
            content += f"[cyan]GPU Memory:[/cyan] {stats['gpu_memory_used_gb']:.1f}GB ({stats['gpu_memory_percent']:.1f}%)\n"

        # Training time
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        content += f"\n[yellow]Training Time:[/yellow] {hours:02d}:{minutes:02d}:{seconds:02d}"

        return Panel(content, title="System Resources", expand=True)

    def create_alerts_panel(self, metrics: Dict, stats: Dict) -> Panel:
        """Create alerts panel for anomalies"""
        alerts = []

        # Check for high memory usage
        if stats["memory_percent"] > 90:
            alerts.append("[red]⚠ High memory usage![/red]")

        # Check for GPU memory issues
        if "gpu_memory_percent" in stats and stats["gpu_memory_percent"] > 95:
            alerts.append("[red]⚠ GPU memory nearly full![/red]")

        # Check for NaN in metrics
        if "summary" in metrics:
            summary = metrics["summary"]
            if "val_loss" in summary:
                val_loss = summary["val_loss"]
                if isinstance(val_loss, list) and val_loss and (val_loss[-1] != val_loss[-1]):  # NaN check
                    alerts.append("[red]⚠ NaN detected in validation loss![/red]")

        # Check for training divergence
        if "summary" in metrics:
            summary = metrics["summary"]
            if "train_loss" in summary:
                losses = summary["train_loss"]
                if isinstance(losses, list) and len(losses) > 10:
                    recent_avg = sum(losses[-5:]) / 5
                    earlier_avg = sum(losses[-10:-5]) / 5
                    if recent_avg > earlier_avg * 2:
                        alerts.append("[yellow]⚠ Possible training divergence![/yellow]")

        if not alerts:
            alerts.append("[green]✓ All systems normal[/green]")

        content = "\n".join(alerts)
        return Panel(content, title="Alerts", expand=True)

    def create_layout(self, metrics: Dict, stats: Dict) -> Layout:
        """Create rich layout for display"""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        # Header
        header_text = Text("ATFT-GAT-FAN Training Monitor", style="bold magenta")
        header_text = Text.from_markup(
            f"[bold cyan]ATFT-GAT-FAN Training Monitor[/bold cyan] - {self.run_dir}"
        )
        layout["header"].update(Panel(header_text))

        # Body
        body = Layout()
        body.split_row(
            Layout(self.create_metrics_table(metrics), name="metrics"),
            Layout(name="side"),
        )

        body["side"].split_column(
            Layout(self.create_system_panel(stats)),
            Layout(self.create_alerts_panel(metrics, stats)),
        )

        layout["body"].update(body)

        # Footer
        footer_text = f"[dim]Press Ctrl+C to exit | Refresh rate: {self.refresh_rate}s[/dim]"
        layout["footer"].update(Panel(Text.from_markup(footer_text)))

        return layout

    def run(self):
        """Run the monitor"""
        console.print("[bold green]Starting Training Monitor...[/bold green]")
        console.print(f"Monitoring: {self.run_dir}")

        try:
            with Live(self.create_layout({}, self.get_system_stats()), refresh_per_second=1) as live:
                while True:
                    metrics = self.read_latest_metrics()
                    stats = self.get_system_stats()
                    live.update(self.create_layout(metrics, stats))
                    time.sleep(self.refresh_rate)

        except KeyboardInterrupt:
            console.print("\n[yellow]Monitor stopped.[/yellow]")


def launch_tensorboard(logdir: Path):
    """Launch TensorBoard in background"""
    import subprocess

    console.print(f"[cyan]Launching TensorBoard at http://localhost:6006[/cyan]")
    subprocess.Popen(
        ["tensorboard", "--logdir", str(logdir), "--port", "6006"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def setup_wandb(project_name: str = "ATFT-GAT-FAN", entity: Optional[str] = None):
    """Setup Weights & Biases tracking"""
    try:
        import wandb

        wandb.init(project=project_name, entity=entity)
        console.print(f"[green]W&B tracking enabled: {wandb.run.get_url()}[/green]")
        return wandb
    except ImportError:
        console.print("[yellow]W&B not installed. Skipping.[/yellow]")
        return None


def main():
    parser = argparse.ArgumentParser(description="Monitor ATFT-GAT-FAN training")
    parser.add_argument(
        "--run-dir",
        type=str,
        default="runs/last",
        help="Training run directory",
    )
    parser.add_argument(
        "--refresh-rate",
        type=float,
        default=1.0,
        help="Refresh rate in seconds",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Launch TensorBoard",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable W&B tracking",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="ATFT-GAT-FAN",
        help="W&B project name",
    )

    args = parser.parse_args()

    # Launch TensorBoard if requested
    if args.tensorboard:
        launch_tensorboard(Path(args.run_dir).parent)

    # Setup W&B if requested
    if args.wandb:
        setup_wandb(args.wandb_project)

    # Start monitor
    monitor = TrainingMonitor(Path(args.run_dir), args.refresh_rate)
    monitor.run()


if __name__ == "__main__":
    main()