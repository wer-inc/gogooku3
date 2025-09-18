"""Optimized production data loader V2 for backward compatibility."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import torch
from torch.utils.data import Dataset
import polars as pl
import numpy as np

logger = logging.getLogger(__name__)


class ProductionDatasetOptimized(Dataset):
    """
    Optimized Production Dataset for backward compatibility with train_atft.py.

    This class provides the expected interface for the legacy training script
    while using modern Polars-based data loading for efficiency.
    """

    def __init__(
        self,
        files: Union[List[Path], List[str], Path, str],
        config: Any,
        mode: str = "train",
        target_scalers: Optional[Dict] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        required_feature_cols: Optional[List[str]] = None,
    ):
        """
        Initialize ProductionDatasetOptimized with compatibility interface.

        Args:
            files: Parquet files to load (can be list or single file)
            config: Configuration object with data settings
            mode: Dataset mode ("train", "val", "test")
            target_scalers: Optional target scalers dict
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            required_feature_cols: Optional list of required feature columns
        """
        self.config = config
        self.mode = mode
        self.target_scalers = target_scalers or {}
        self.start_date = start_date
        self.end_date = end_date

        # Convert files to list of Path objects
        if isinstance(files, (str, Path)):
            files = [files]
        self.files = [Path(f) for f in files]

        # Load and prepare data
        self._load_data()

        # Apply date filtering if specified
        if self.start_date or self.end_date:
            self._filter_by_date()

        # Extract features and targets
        self._prepare_features_and_targets(required_feature_cols)

        # Set up prediction horizons
        self.prediction_horizons = self._get_prediction_horizons()

        # Build targets dictionary for compatibility
        self.targets = self._build_targets_dict()

        logger.info(
            f"ProductionDatasetOptimized initialized: {len(self)} samples, "
            f"{len(self.feature_cols)} features, mode={mode}"
        )

    def _load_data(self):
        """Load data from parquet files using Polars for efficiency."""
        dfs = []
        for file_path in self.files:
            if file_path.exists():
                # Use lazy loading for memory efficiency
                df = pl.read_parquet(file_path)
                dfs.append(df)
            else:
                logger.warning(f"File not found: {file_path}")

        if not dfs:
            # Create empty dataset if no files found
            self.data = pl.DataFrame()
            logger.warning("No valid files found, creating empty dataset")
        else:
            # Concatenate all dataframes
            self.data = pl.concat(dfs, how="vertical")
            logger.info(f"Loaded {len(self.data)} rows from {len(dfs)} files")

    def _filter_by_date(self):
        """Filter data by date range if specified."""
        if len(self.data) == 0:
            return

        # Check for date column
        date_col = None
        for col in ["Date", "date", "datetime"]:
            if col in self.data.columns:
                date_col = col
                break

        if not date_col:
            logger.warning("No date column found for filtering")
            return

        # Apply date filters
        if self.start_date:
            self.data = self.data.filter(pl.col(date_col) >= self.start_date)
        if self.end_date:
            self.data = self.data.filter(pl.col(date_col) <= self.end_date)

        logger.info(f"Date filtered to {len(self.data)} rows")

    def _prepare_features_and_targets(self, required_feature_cols: Optional[List[str]] = None):
        """Extract features and target columns."""
        if len(self.data) == 0:
            self.feature_cols = []
            self.feature_data = np.array([])
            self.target_data = {}
            return

        # Identify target columns based on common patterns
        target_patterns = [
            "feat_ret_", "returns_", "ret_", "target_", "horizon_"
        ]
        target_cols = []
        for col in self.data.columns:
            if any(pattern in col.lower() for pattern in target_patterns):
                # Check if it ends with a day indicator
                if any(col.endswith(f"_{h}d") or col.endswith(f"_{h}")
                       for h in [1, 5, 10, 20, 30, 60]):
                    target_cols.append(col)

        # Identify feature columns (everything except targets and metadata)
        exclude_cols = set(["Date", "date", "datetime", "Code", "code", "symbol"])
        exclude_cols.update(target_cols)

        if required_feature_cols:
            # Use specified feature columns if provided
            self.feature_cols = [col for col in required_feature_cols
                                 if col in self.data.columns]
            if len(self.feature_cols) < len(required_feature_cols):
                missing = set(required_feature_cols) - set(self.feature_cols)
                logger.warning(f"Missing required features: {missing}")
        else:
            # Auto-detect feature columns
            self.feature_cols = [col for col in self.data.columns
                                 if col not in exclude_cols]

        # Convert to numpy arrays for faster access
        if self.feature_cols:
            self.feature_data = self.data.select(self.feature_cols).fill_null(0).to_numpy().astype(np.float32)
        else:
            self.feature_data = np.zeros((len(self.data), 1), dtype=np.float32)
            logger.warning("No feature columns found")

        # Extract target data
        self.target_data = {}
        for col in target_cols:
            if col in self.data.columns:
                self.target_data[col] = self.data[col].fill_null(0).to_numpy().astype(np.float32)

        logger.info(f"Prepared {len(self.feature_cols)} features, {len(self.target_data)} targets")

    def _get_prediction_horizons(self) -> List[int]:
        """Extract prediction horizons from config or target columns."""
        # Try to get from config first
        if hasattr(self.config, 'data') and hasattr(self.config.data, 'time_series'):
            if hasattr(self.config.data.time_series, 'prediction_horizons'):
                return self.config.data.time_series.prediction_horizons

        # Fallback: extract from target column names
        horizons = set()
        for col in self.target_data.keys():
            # Extract horizon number from column name
            import re
            match = re.search(r'_(\d+)d?$', col)
            if match:
                horizons.add(int(match.group(1)))

        if not horizons:
            # Default horizons
            horizons = [1, 5, 10, 20]
            logger.warning(f"No horizons found, using defaults: {horizons}")
        else:
            horizons = sorted(list(horizons))

        return horizons

    def _build_targets_dict(self) -> Dict[int, np.ndarray]:
        """Build targets dictionary with horizon keys for compatibility."""
        targets_dict = {}

        for horizon in self.prediction_horizons:
            # Look for matching target column
            target_col = None
            for col in self.target_data.keys():
                if f"_{horizon}d" in col or f"_{horizon}" in col:
                    target_col = col
                    break

            if target_col:
                targets_dict[horizon] = self.target_data[target_col]
            else:
                # Create dummy targets if not found
                logger.warning(f"No target found for horizon {horizon}, using zeros")
                targets_dict[horizon] = np.zeros(len(self), dtype=np.float32)

        return targets_dict

    def __len__(self) -> int:
        """Return dataset length."""
        if self.feature_data is None or len(self.feature_data) == 0:
            return 0
        return len(self.feature_data)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.

        Returns dict with:
            - features: Tensor of shape (seq_len, n_features) or (n_features,)
            - targets: Dict with horizon keys containing target values
        """
        # Get sequence length from config if available
        seq_len = 60  # default
        if hasattr(self.config, 'data') and hasattr(self.config.data, 'time_series'):
            if hasattr(self.config.data.time_series, 'sequence_length'):
                seq_len = self.config.data.time_series.sequence_length

        # Prepare features
        if idx + seq_len <= len(self.feature_data):
            # Get sequence of features
            features = self.feature_data[idx:idx + seq_len]
        else:
            # Pad if needed
            available = len(self.feature_data) - idx
            features = np.zeros((seq_len, self.feature_data.shape[1]), dtype=np.float32)
            if available > 0:
                features[:available] = self.feature_data[idx:idx + available]

        # Prepare targets
        targets = {}
        target_idx = min(idx + seq_len - 1, len(self) - 1) if seq_len > 1 else idx
        for horizon in self.prediction_horizons:
            if horizon in self.targets:
                targets[f"horizon_{horizon}"] = torch.tensor(
                    self.targets[horizon][target_idx], dtype=torch.float32
                ).unsqueeze(0)  # Add batch dimension for compatibility
            else:
                targets[f"horizon_{horizon}"] = torch.zeros(1, dtype=torch.float32)

        return {
            "features": torch.from_numpy(features),
            "targets": targets,
        }
