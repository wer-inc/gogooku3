"""
Training metrics extraction for HPO optimization
Extracts RankIC, Sharpe, and other financial metrics from training logs and results
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for extracted training metrics"""
    rank_ic: dict[str, float]
    sharpe: dict[str, float]
    train_loss: float
    val_loss: float
    epoch: int
    training_time: float
    metadata: dict[str, Any]


class MetricsExtractor:
    """Extract and parse training metrics from various sources"""

    def __init__(self,
                 log_patterns: dict[str, str] | None = None,
                 metric_names: list[str] | None = None):
        """
        Initialize metrics extractor

        Args:
            log_patterns: Custom regex patterns for metric extraction
            metric_names: List of metric names to extract
        """
        self.horizons = ['1d', '5d', '10d', '20d']
        self.metric_names = metric_names or [
            'rank_ic', 'sharpe', 'information_ratio', 'max_drawdown'
        ]

        # Default patterns for log parsing
        self.log_patterns = log_patterns or {
            'rank_ic': r'rank_ic_(\w+):\s*([-\d\.]+)',
            'sharpe': r'sharpe_(\w+):\s*([-\d\.]+)',
            'train_loss': r'train_loss:\s*([-\d\.]+)',
            'val_loss': r'val_loss:\s*([-\d\.]+)',
            'epoch': r'epoch:\s*(\d+)',
            'lr': r'lr:\s*([-\de\.]+)'
        }

    def extract_from_log_file(self, log_path: str | Path) -> TrainingMetrics | None:
        """
        Extract metrics from training log file

        Args:
            log_path: Path to training log file

        Returns:
            TrainingMetrics object or None if extraction fails
        """
        try:
            log_path = Path(log_path)
            if not log_path.exists():
                logger.warning(f"Log file not found: {log_path}")
                return None

            with open(log_path, encoding='utf-8') as f:
                log_content = f.read()

            return self._parse_log_content(log_content)

        except Exception as e:
            logger.error(f"Failed to extract from log file {log_path}: {e}")
            return None

    def extract_from_tensorboard(self, tb_log_dir: str | Path) -> TrainingMetrics | None:
        """
        Extract metrics from TensorBoard logs

        Args:
            tb_log_dir: Path to TensorBoard log directory

        Returns:
            TrainingMetrics object or None if extraction fails
        """
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

            tb_log_dir = Path(tb_log_dir)
            if not tb_log_dir.exists():
                logger.warning(f"TensorBoard log directory not found: {tb_log_dir}")
                return None

            # Load TensorBoard data
            event_acc = EventAccumulator(str(tb_log_dir))
            event_acc.Reload()

            # Extract scalar metrics
            metrics = {}

            for tag in event_acc.Tags()['scalars']:
                values = event_acc.Scalars(tag)
                if values:
                    # Get the latest value
                    latest_value = values[-1].value
                    metrics[tag] = latest_value

            return self._parse_tensorboard_metrics(metrics)

        except ImportError:
            logger.warning("TensorBoard not available. Install with: pip install tensorboard")
            return None
        except Exception as e:
            logger.error(f"Failed to extract from TensorBoard {tb_log_dir}: {e}")
            return None

    def extract_from_wandb(self, run_path: str) -> TrainingMetrics | None:
        """
        Extract metrics from Weights & Biases run

        Args:
            run_path: W&B run path (entity/project/run_id)

        Returns:
            TrainingMetrics object or None if extraction fails
        """
        try:
            import wandb

            # Initialize W&B API
            api = wandb.Api()
            run = api.run(run_path)

            # Get metrics from summary
            summary = run.summary
            history = run.history()

            return self._parse_wandb_metrics(summary, history)

        except ImportError:
            logger.warning("W&B not available. Install with: pip install wandb")
            return None
        except Exception as e:
            logger.error(f"Failed to extract from W&B {run_path}: {e}")
            return None

    def extract_from_metrics_dict(self, metrics_dict: dict[str, Any]) -> TrainingMetrics | None:
        """
        Extract metrics from dictionary (direct from training)

        Args:
            metrics_dict: Dictionary containing training metrics

        Returns:
            TrainingMetrics object or None if extraction fails
        """
        try:
            rank_ic = {}
            sharpe = {}

            # Extract horizon-specific metrics
            for horizon in self.horizons:
                # RankIC metrics
                for key, value in metrics_dict.items():
                    if f"rank_ic_{horizon}" in key.lower():
                        rank_ic[horizon] = float(value)
                    elif f"sharpe_{horizon}" in key.lower():
                        sharpe[horizon] = float(value)

            # Extract general metrics
            train_loss = metrics_dict.get('train_loss', 0.0)
            val_loss = metrics_dict.get('val_loss', 0.0)
            epoch = metrics_dict.get('epoch', 0)
            training_time = metrics_dict.get('training_time', 0.0)

            return TrainingMetrics(
                rank_ic=rank_ic,
                sharpe=sharpe,
                train_loss=float(train_loss),
                val_loss=float(val_loss),
                epoch=int(epoch),
                training_time=float(training_time),
                metadata=metrics_dict.copy()
            )

        except Exception as e:
            logger.error(f"Failed to extract from metrics dict: {e}")
            return None

    def _parse_log_content(self, log_content: str) -> TrainingMetrics | None:
        """Parse metrics from log content"""
        try:
            rank_ic = {}
            sharpe = {}

            # Extract RankIC metrics
            rank_ic_matches = re.findall(self.log_patterns['rank_ic'], log_content)
            for horizon, value in rank_ic_matches:
                if horizon in self.horizons:
                    rank_ic[horizon] = float(value)

            # Extract Sharpe metrics
            sharpe_matches = re.findall(self.log_patterns['sharpe'], log_content)
            for horizon, value in sharpe_matches:
                if horizon in self.horizons:
                    sharpe[horizon] = float(value)

            # Extract general metrics
            train_loss_match = re.search(self.log_patterns['train_loss'], log_content)
            val_loss_match = re.search(self.log_patterns['val_loss'], log_content)
            epoch_match = re.search(self.log_patterns['epoch'], log_content)

            train_loss = float(train_loss_match.group(1)) if train_loss_match else 0.0
            val_loss = float(val_loss_match.group(1)) if val_loss_match else 0.0
            epoch = int(epoch_match.group(1)) if epoch_match else 0

            return TrainingMetrics(
                rank_ic=rank_ic,
                sharpe=sharpe,
                train_loss=train_loss,
                val_loss=val_loss,
                epoch=epoch,
                training_time=0.0,
                metadata={'source': 'log_file'}
            )

        except Exception as e:
            logger.error(f"Failed to parse log content: {e}")
            return None

    def _parse_tensorboard_metrics(self, metrics: dict[str, float]) -> TrainingMetrics | None:
        """Parse metrics from TensorBoard data"""
        try:
            rank_ic = {}
            sharpe = {}

            for key, value in metrics.items():
                # Parse horizon-specific metrics
                for horizon in self.horizons:
                    if f"rank_ic_{horizon}" in key.lower():
                        rank_ic[horizon] = value
                    elif f"sharpe_{horizon}" in key.lower():
                        sharpe[horizon] = value

            train_loss = metrics.get('train_loss', 0.0)
            val_loss = metrics.get('val_loss', 0.0)
            epoch = int(metrics.get('epoch', 0))

            return TrainingMetrics(
                rank_ic=rank_ic,
                sharpe=sharpe,
                train_loss=train_loss,
                val_loss=val_loss,
                epoch=epoch,
                training_time=0.0,
                metadata={'source': 'tensorboard', 'raw_metrics': metrics}
            )

        except Exception as e:
            logger.error(f"Failed to parse TensorBoard metrics: {e}")
            return None

    def _parse_wandb_metrics(self, summary: dict, history: pd.DataFrame) -> TrainingMetrics | None:
        """Parse metrics from W&B data"""
        try:
            rank_ic = {}
            sharpe = {}

            # Extract from summary (final values)
            for key, value in summary.items():
                for horizon in self.horizons:
                    if f"rank_ic_{horizon}" in key.lower() and value is not None:
                        rank_ic[horizon] = float(value)
                    elif f"sharpe_{horizon}" in key.lower() and value is not None:
                        sharpe[horizon] = float(value)

            # Get latest epoch metrics from history
            if not history.empty:
                latest_row = history.iloc[-1]
                train_loss = latest_row.get('train_loss', 0.0)
                val_loss = latest_row.get('val_loss', 0.0)
                epoch = int(latest_row.get('epoch', 0))
            else:
                train_loss = summary.get('train_loss', 0.0)
                val_loss = summary.get('val_loss', 0.0)
                epoch = int(summary.get('epoch', 0))

            return TrainingMetrics(
                rank_ic=rank_ic,
                sharpe=sharpe,
                train_loss=float(train_loss) if train_loss else 0.0,
                val_loss=float(val_loss) if val_loss else 0.0,
                epoch=epoch,
                training_time=0.0,
                metadata={'source': 'wandb', 'summary': dict(summary)}
            )

        except Exception as e:
            logger.error(f"Failed to parse W&B metrics: {e}")
            return None

    def validate_metrics(self, metrics: TrainingMetrics) -> dict[str, list[str]]:
        """
        Validate extracted metrics for completeness and sanity

        Args:
            metrics: TrainingMetrics object to validate

        Returns:
            Dictionary with 'errors' and 'warnings' keys
        """
        errors = []
        warnings = []

        # Check RankIC completeness
        missing_rank_ic = [h for h in self.horizons if h not in metrics.rank_ic]
        if missing_rank_ic:
            warnings.append(f"Missing RankIC for horizons: {missing_rank_ic}")

        # Check Sharpe completeness
        missing_sharpe = [h for h in self.horizons if h not in metrics.sharpe]
        if missing_sharpe:
            warnings.append(f"Missing Sharpe for horizons: {missing_sharpe}")

        # Sanity checks for RankIC (-1 to 1 range)
        for horizon, ic in metrics.rank_ic.items():
            if not (-1.0 <= ic <= 1.0):
                warnings.append(f"RankIC_{horizon}={ic:.3f} outside expected range [-1, 1]")

        # Sanity checks for Sharpe (reasonable range)
        for horizon, sharpe in metrics.sharpe.items():
            if not (-3.0 <= sharpe <= 5.0):
                warnings.append(f"Sharpe_{horizon}={sharpe:.3f} outside reasonable range [-3, 5]")

        # Check for NaN values
        for horizon, ic in metrics.rank_ic.items():
            if np.isnan(ic):
                errors.append(f"RankIC_{horizon} is NaN")

        for horizon, sharpe in metrics.sharpe.items():
            if np.isnan(sharpe):
                errors.append(f"Sharpe_{horizon} is NaN")

        return {'errors': errors, 'warnings': warnings}

    def format_metrics(self, metrics: TrainingMetrics) -> str:
        """Format metrics for logging"""
        lines = []
        lines.append(f"📊 Training Metrics (Epoch {metrics.epoch})")
        lines.append(f"   Loss: train={metrics.train_loss:.4f}, val={metrics.val_loss:.4f}")

        if metrics.rank_ic:
            rank_ic_str = ", ".join([f"{h}={v:.3f}" for h, v in metrics.rank_ic.items()])
            lines.append(f"   RankIC: {rank_ic_str}")

        if metrics.sharpe:
            sharpe_str = ", ".join([f"{h}={v:.3f}" for h, v in metrics.sharpe.items()])
            lines.append(f"   Sharpe: {sharpe_str}")

        if metrics.training_time > 0:
            lines.append(f"   Training time: {metrics.training_time:.1f}s")

        return "\n".join(lines)
