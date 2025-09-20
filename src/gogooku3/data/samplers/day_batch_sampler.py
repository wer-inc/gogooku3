"""
Day Batch Sampler for time-series data.

Groups samples by trading day to ensure temporal consistency in batches.
Critical for financial ML to prevent data leakage and maintain temporal structure.
"""

import logging
from collections import defaultdict
from typing import Iterator, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class DayBatchSampler(Sampler):
    """
    Sampler that groups samples by trading day.

    This sampler ensures that:
    1. All samples from the same day are in the same batch
    2. Days are processed chronologically
    3. Optional shuffling within days but not across days
    4. Support for multi-stock batching
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 256,
        shuffle_within_day: bool = True,
        shuffle_days: bool = False,
        drop_last: bool = False,
        date_column: str = "date",
        code_column: str = "code",
        min_samples_per_day: int = 10,
        seed: int = 42,
        # 互換性: 呼び出し元が `shuffle=` を渡す場合に受け付ける
        shuffle: Optional[bool] = None,
    ):
        """
        Initialize DayBatchSampler.

        Args:
            dataset: Dataset with date information
            batch_size: Maximum batch size
            shuffle_within_day: Whether to shuffle samples within each day
            shuffle_days: Whether to shuffle the order of days
            drop_last: Whether to drop the last incomplete batch
            date_column: Name of date column in dataset
            code_column: Name of stock code column in dataset
            min_samples_per_day: Minimum samples required per day
            seed: Random seed for shuffling
        """
        self.dataset = dataset
        self.batch_size = batch_size
        # 互換性対応: shuffle 引数が与えられたら優先
        self.shuffle_within_day = (
            shuffle if isinstance(shuffle, bool) else shuffle_within_day
        )
        self.shuffle_days = shuffle_days
        self.drop_last = drop_last
        self.date_column = date_column
        self.code_column = code_column
        self.min_samples_per_day = min_samples_per_day
        self.seed = seed

        # Build date-based indices
        self.date_indices = self._build_date_indices()
        self.batches = self._create_batches()

        logger.info(
            f"DayBatchSampler initialized: {len(self.date_indices)} days, "
            f"{len(self.batches)} batches"
        )

    def _build_date_indices(self) -> dict:
        """
        Build mapping from dates to sample indices.

        Returns:
            Dictionary mapping date -> list of indices
        """
        date_indices = defaultdict(list)

        # Fast-path: StreamingParquetDataset 等が sequence_dates を提供する場合
        try:
            seq_dates = getattr(self.dataset, "sequence_dates", None)
            if seq_dates is not None:
                dates_np = np.asarray(seq_dates)
                # 可能なら日次に正規化
                if dates_np.dtype.kind in ("U", "O"):
                    dates_np = np.array(dates_np, dtype="datetime64[D]")
                elif str(dates_np.dtype).startswith("datetime64[") is False:
                    # 予期しない dtype は日次にキャストを試みる
                    try:
                        dates_np = np.array(dates_np, dtype="datetime64[D]")
                    except Exception:
                        pass

                # 逆インデックスで高速グルーピング
                # unique はソート順（= 時系列昇順）を保つ
                unique, inv = np.unique(dates_np, return_inverse=True)
                order = np.argsort(inv, kind="stable")
                inv_sorted = inv[order]
                boundaries = np.flatnonzero(
                    np.r_[True, inv_sorted[1:] != inv_sorted[:-1]]
                )

                grouped: dict = {}
                for g_idx, start in enumerate(boundaries):
                    end = boundaries[g_idx + 1] if g_idx + 1 < len(boundaries) else len(order)
                    inds = order[start:end]
                    if inds.size >= self.min_samples_per_day:
                        grouped[str(unique[g_idx])] = inds.tolist()
                if grouped:
                    return grouped
        except Exception as e:
            logger.debug(f"sequence_dates fast-path failed: {e}")

        # Handle different dataset types（フォールバック）
        if hasattr(self.dataset, "data"):
            # Custom dataset with data attribute
            data = self.dataset.data
        elif hasattr(self.dataset, "df"):
            # DataFrame-based dataset
            data = self.dataset.df
        elif hasattr(self.dataset, "__getitem__"):
            # Try to extract dates from dataset items
            try:
                # Sample first item to check structure
                sample = self.dataset[0]
                if isinstance(sample, dict) and self.date_column in sample:
                    # Iterate through dataset to build indices
                    for idx in range(len(self.dataset)):
                        item = self.dataset[idx]
                        date = item[self.date_column]
                        date_indices[date].append(idx)
                    return dict(date_indices)
                else:
                    # Fallback: assume sequential indices
                    logger.warning(
                        "Could not extract dates from dataset. Using sequential batching."
                    )
                    return self._build_sequential_indices()
            except Exception as e:
                logger.warning(f"Error accessing dataset: {e}. Using sequential batching.")
                return self._build_sequential_indices()
        else:
            # Unknown dataset type - use sequential batching
            return self._build_sequential_indices()

        # Extract dates from data
        if isinstance(data, pd.DataFrame):
            if self.date_column in data.columns:
                for idx, date in enumerate(data[self.date_column]):
                    date_indices[date].append(idx)
            else:
                logger.warning(f"Date column '{self.date_column}' not found in data")
                return self._build_sequential_indices()
        elif hasattr(data, self.date_column):
            # Polars DataFrame or similar
            try:
                dates = data[self.date_column].to_list()
                for idx, date in enumerate(dates):
                    date_indices[date].append(idx)
            except Exception as e:
                logger.warning(f"Error extracting dates: {e}")
                return self._build_sequential_indices()
        else:
            return self._build_sequential_indices()

        # Filter days with too few samples
        filtered_indices = {}
        for date, indices in date_indices.items():
            if len(indices) >= self.min_samples_per_day:
                filtered_indices[date] = indices
            else:
                logger.debug(f"Dropping date {date} with only {len(indices)} samples")

        return filtered_indices

    def _build_sequential_indices(self) -> dict:
        """
        Build sequential indices when date information is not available.

        Returns:
            Dictionary with pseudo-dates mapping to indices
        """
        n_samples = len(self.dataset)
        n_days = max(1, n_samples // (self.batch_size * 4))  # Estimate number of days
        samples_per_day = n_samples // n_days

        date_indices = {}
        for day in range(n_days):
            start_idx = day * samples_per_day
            end_idx = min((day + 1) * samples_per_day, n_samples)
            if day == n_days - 1:
                end_idx = n_samples  # Include all remaining samples in last day

            date_indices[f"day_{day:04d}"] = list(range(start_idx, end_idx))

        return date_indices

    def _create_batches(self) -> List[List[int]]:
        """
        Create batches from date-based indices.

        Returns:
            List of batches, where each batch is a list of indices
        """
        batches = []

        # Get dates and optionally shuffle
        dates = list(self.date_indices.keys())
        if self.shuffle_days:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(dates)
        else:
            dates = sorted(dates)  # Ensure chronological order

        # Process each day
        for date in dates:
            day_indices = self.date_indices[date].copy()

            # Optionally shuffle within day
            if self.shuffle_within_day:
                rng = np.random.RandomState(self.seed + hash(date) % 100000)
                rng.shuffle(day_indices)

            # Create batches for this day
            for i in range(0, len(day_indices), self.batch_size):
                batch = day_indices[i:i + self.batch_size]

                # Handle drop_last
                if self.drop_last and len(batch) < self.batch_size:
                    continue

                batches.append(batch)

        return batches

    def __iter__(self) -> Iterator[List[int]]:
        """
        Iterate over batches.

        Yields:
            List of indices for each batch
        """
        # Optionally reshuffle for each epoch
        if self.shuffle_days or self.shuffle_within_day:
            self.batches = self._create_batches()

        for batch in self.batches:
            yield batch

    def __len__(self) -> int:
        """
        Get number of batches.

        Returns:
            Number of batches
        """
        return len(self.batches)

    def get_date_for_batch(self, batch_idx: int) -> Optional[str]:
        """
        Get the date associated with a batch.

        Args:
            batch_idx: Batch index

        Returns:
            Date string or None if not found
        """
        if batch_idx >= len(self.batches):
            return None

        batch = self.batches[batch_idx]
        if not batch:
            return None

        # Find which date this batch belongs to
        first_idx = batch[0]
        for date, indices in self.date_indices.items():
            if first_idx in indices:
                return str(date)

        return None

    def get_statistics(self) -> dict:
        """
        Get sampler statistics.

        Returns:
            Dictionary with sampler statistics
        """
        stats = {
            "n_days": len(self.date_indices),
            "n_batches": len(self.batches),
            "batch_size": self.batch_size,
            "total_samples": sum(len(indices) for indices in self.date_indices.values()),
            "avg_samples_per_day": np.mean([len(indices) for indices in self.date_indices.values()])
            if self.date_indices
            else 0,
            "min_samples_per_day": min([len(indices) for indices in self.date_indices.values()])
            if self.date_indices
            else 0,
            "max_samples_per_day": max([len(indices) for indices in self.date_indices.values()])
            if self.date_indices
            else 0,
        }

        return stats


class StratifiedDayBatchSampler(DayBatchSampler):
    """
    Stratified version of DayBatchSampler that ensures balanced representation.

    Useful for:
    - Balancing across sectors
    - Ensuring market cap diversity
    - Maintaining target distribution
    """

    def __init__(
        self,
        dataset,
        stratify_column: str,
        n_strata: int = 10,
        **kwargs
    ):
        """
        Initialize StratifiedDayBatchSampler.

        Args:
            dataset: Dataset with stratification information
            stratify_column: Column to use for stratification
            n_strata: Number of strata to create
            **kwargs: Arguments passed to DayBatchSampler
        """
        self.stratify_column = stratify_column
        self.n_strata = n_strata

        # Build strata before calling parent init
        self.strata = self._build_strata(dataset)

        super().__init__(dataset, **kwargs)

    def _build_strata(self, dataset) -> dict:
        """
        Build stratification groups.

        Args:
            dataset: Dataset to stratify

        Returns:
            Dictionary mapping stratum -> indices
        """
        # Implementation depends on dataset structure
        # This is a placeholder that should be customized
        logger.info(f"Building {self.n_strata} strata based on {self.stratify_column}")
        return {}

    def _create_batches(self) -> List[List[int]]:
        """
        Create stratified batches.

        Returns:
            List of batches with balanced strata
        """
        # Start with regular batches
        batches = super()._create_batches()

        # Apply stratification if available
        if self.strata:
            # Custom stratification logic here
            pass

        return batches


__all__ = ["DayBatchSampler", "StratifiedDayBatchSampler"]
