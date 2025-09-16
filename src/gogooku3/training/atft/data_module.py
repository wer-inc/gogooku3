"""
ATFT Data Module
データローディングとバッチ処理を管理するモジュール
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import polars as pl
import numpy as np
from omegaconf import DictConfig

# Prefer the project DayBatchSampler implementation; fall back to internal placeholder
try:  # pragma: no cover - runtime import guard
    from gogooku3.data.samplers.day_batch_sampler import (
        DayBatchSampler as ExtDayBatchSampler,
    )
    _USE_EXT_SAMPLER = True
except Exception:  # pragma: no cover - fallback for environments without full package
    _USE_EXT_SAMPLER = False

logger = logging.getLogger(__name__)


class StreamingParquetDataset(Dataset):
    """Streaming dataset for large parquet files."""

    def __init__(
        self,
        file_paths: List[Path],
        feature_columns: List[str],
        target_columns: List[str],
        sequence_length: int = 60,
        normalize_online: bool = True,
        cache_size: int = 10000,
    ):
        """
        Initialize streaming parquet dataset.

        Args:
            file_paths: List of parquet file paths
            feature_columns: Feature column names
            target_columns: Target column names
            sequence_length: Sequence length for time series
            normalize_online: Apply online normalization
            cache_size: Number of samples to cache in memory
        """
        self.file_paths = file_paths
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.sequence_length = sequence_length
        self.normalize_online = normalize_online
        self.cache_size = cache_size

        # Initialize cache
        self._cache = {}
        self._cache_indices = []

        # Get total length
        self._length = self._calculate_length()

    def _calculate_length(self) -> int:
        """Calculate total dataset length."""
        total_length = 0
        for file_path in self.file_paths:
            df = pl.scan_parquet(file_path).select(pl.count()).collect()
            total_length += df[0, 0] - self.sequence_length + 1
        return total_length

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample."""
        # Check cache
        if idx in self._cache:
            return self._cache[idx]

        # Load sample
        sample = self._load_sample(idx)

        # Update cache
        if len(self._cache) >= self.cache_size:
            # Remove oldest cached item
            oldest_idx = self._cache_indices.pop(0)
            del self._cache[oldest_idx]

        self._cache[idx] = sample
        self._cache_indices.append(idx)

        return sample

    def _load_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load a single sample from disk."""
        # Find which file contains this index
        current_idx = idx
        for file_path in self.file_paths:
            df = pl.read_parquet(file_path)
            file_length = len(df) - self.sequence_length + 1

            if current_idx < file_length:
                # Load sequence from this file
                start_idx = current_idx
                end_idx = start_idx + self.sequence_length

                # Get features and targets
                features = df[start_idx:end_idx][self.feature_columns].to_numpy()
                targets = df[end_idx - 1][self.target_columns].to_numpy()

                # Apply online normalization if needed
                if self.normalize_online:
                    features = self._normalize(features)

                return (
                    torch.tensor(features, dtype=torch.float32),
                    torch.tensor(targets, dtype=torch.float32),
                )

            current_idx -= file_length

        raise IndexError(f"Index {idx} out of range")

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """Apply online normalization."""
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True) + 1e-8
        return (features - mean) / std


class _InternalDayBatchSampler(Sampler):
    """Sampler that groups samples by day for batch processing."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        date_indices: Optional[Dict[str, List[int]]] = None,
        shuffle: bool = True,
    ):
        """
        Initialize day batch sampler.

        Args:
            dataset: Dataset to sample from
            batch_size: Batch size
            date_indices: Mapping from dates to sample indices
            shuffle: Whether to shuffle dates
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.date_indices = date_indices or self._build_date_indices()
        self.shuffle = shuffle

    def _build_date_indices(self) -> Dict[str, List[int]]:
        """Build mapping from dates to indices (placeholder)."""
        # This would need actual implementation based on dataset structure
        logger.warning("Using placeholder date indices - implement actual date grouping")
        return {"default": list(range(len(self.dataset)))}

    def __iter__(self):
        """Iterate over batches grouped by day."""
        dates = list(self.date_indices.keys())

        if self.shuffle:
            np.random.shuffle(dates)

        for date in dates:
            indices = self.date_indices[date]

            if self.shuffle:
                np.random.shuffle(indices)

            # Yield batches for this date
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                if batch_indices:  # Skip empty batches
                    yield batch_indices

    def __len__(self) -> int:
        """Return number of batches."""
        total_batches = 0
        for indices in self.date_indices.values():
            total_batches += (len(indices) + self.batch_size - 1) // self.batch_size
        return total_batches


# Bind the sampler symbol used below
if _USE_EXT_SAMPLER:
    DayBatchSampler = ExtDayBatchSampler  # type: ignore
else:  # pragma: no cover - keep internal placeholder
    DayBatchSampler = _InternalDayBatchSampler  # type: ignore


class ProductionDataModuleV2:
    """Production data module for ATFT training."""

    def __init__(self, config: DictConfig):
        """
        Initialize data module.

        Args:
            config: Data configuration
        """
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self) -> None:
        """Set up datasets."""
        data_dir = Path(self.config.data.source.data_dir)

        # Find train/val/test files
        train_files = sorted(data_dir.glob("train/*.parquet"))
        val_files = sorted(data_dir.glob("val/*.parquet"))
        test_files = sorted(data_dir.glob("test/*.parquet"))

        if not train_files:
            raise FileNotFoundError(f"No training files found in {data_dir}/train/")

        logger.info(f"📂 Found {len(train_files)} train, {len(val_files)} val, {len(test_files)} test files")

        # Get feature and target columns
        feature_columns = self._get_feature_columns()
        target_columns = self._get_target_columns()

        # Create datasets
        self.train_dataset = StreamingParquetDataset(
            file_paths=train_files,
            feature_columns=feature_columns,
            target_columns=target_columns,
            sequence_length=self.config.data.time_series.sequence_length,
            normalize_online=self.config.normalization.online_normalization.enabled,
        )

        if val_files:
            self.val_dataset = StreamingParquetDataset(
                file_paths=val_files,
                feature_columns=feature_columns,
                target_columns=target_columns,
                sequence_length=self.config.data.time_series.sequence_length,
                normalize_online=self.config.normalization.online_normalization.enabled,
            )

        if test_files:
            self.test_dataset = StreamingParquetDataset(
                file_paths=test_files,
                feature_columns=feature_columns,
                target_columns=target_columns,
                sequence_length=self.config.data.time_series.sequence_length,
                normalize_online=self.config.normalization.online_normalization.enabled,
            )

        logger.info(f"✅ Datasets created: train={len(self.train_dataset)} samples")

    def _get_feature_columns(self) -> List[str]:
        """Get feature column names."""
        if self.config.data.schema.feature_columns:
            return self.config.data.schema.feature_columns

        # Auto-detect from first file
        data_dir = Path(self.config.data.source.data_dir)
        first_file = next(data_dir.glob("**/*.parquet"), None)

        if not first_file:
            raise FileNotFoundError("No parquet files found for column detection")

        df = pl.scan_parquet(first_file).head(1).collect()
        exclude_cols = [
            self.config.data.schema.date_column,
            self.config.data.schema.code_column,
            self.config.data.schema.target_column,
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]
        logger.info(f"✅ Auto-detected {len(feature_cols)} feature columns")

        return feature_cols

    def _get_target_columns(self) -> List[str]:
        """Get target column names."""
        horizons = self.config.data.time_series.prediction_horizons
        base_target = self.config.data.schema.target_column

        # Generate target column names for each horizon
        target_cols = []
        for h in horizons:
            target_cols.append(f"{base_target}_{h}d")

        return target_cols

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        if self.train_dataset is None:
            self.setup()

        # Use day batch sampler if enabled
        if self.config.data.get("use_day_batch_sampler", False):
            sampler = DayBatchSampler(
                dataset=self.train_dataset,
                batch_size=self.config.train.batch.train_batch_size,
                shuffle=True,
            )
            return DataLoader(
                self.train_dataset,
                batch_sampler=sampler,
                num_workers=self.config.train.batch.num_workers,
                pin_memory=self.config.train.batch.pin_memory,
                prefetch_factor=self.config.train.batch.prefetch_factor,
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.config.train.batch.train_batch_size,
                shuffle=True,
                num_workers=self.config.train.batch.num_workers,
                pin_memory=self.config.train.batch.pin_memory,
                prefetch_factor=self.config.train.batch.prefetch_factor,
            )

    def val_dataloader(self) -> Optional[DataLoader]:
        """Get validation dataloader."""
        if self.val_dataset is None:
            return None

        return DataLoader(
            self.val_dataset,
            batch_size=self.config.train.batch.val_batch_size,
            shuffle=False,
            num_workers=self.config.train.batch.num_workers,
            pin_memory=self.config.train.batch.pin_memory,
            prefetch_factor=self.config.train.batch.prefetch_factor,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """Get test dataloader."""
        if self.test_dataset is None:
            return None

        return DataLoader(
            self.test_dataset,
            batch_size=self.config.train.batch.val_batch_size,
            shuffle=False,
            num_workers=self.config.train.batch.num_workers,
            pin_memory=self.config.train.batch.pin_memory,
            prefetch_factor=self.config.train.batch.prefetch_factor,
        )

    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the data."""
        info = {
            "train_size": len(self.train_dataset) if self.train_dataset else 0,
            "val_size": len(self.val_dataset) if self.val_dataset else 0,
            "test_size": len(self.test_dataset) if self.test_dataset else 0,
            "num_features": len(self._get_feature_columns()),
            "num_targets": len(self._get_target_columns()),
            "sequence_length": self.config.data.time_series.sequence_length,
            "batch_size": self.config.train.batch.train_batch_size,
        }
        return info


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Custom collate function for batching."""
    features, targets = zip(*batch)
    features = torch.stack(features)
    targets = torch.stack(targets)
    return features, targets
