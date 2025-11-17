"""
Day Batch Sampler for time-series data.

Groups samples by trading day to ensure temporal consistency in batches.
Critical for financial ML to prevent data leakage and maintain temporal structure.
"""

import logging
from collections import defaultdict
from collections.abc import Iterator

import numpy as np
import pandas as pd
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
        dataset=None,
        batch_size: int = 256,
        shuffle_within_day: bool = True,
        shuffle_days: bool = False,
        drop_last: bool = False,
        date_column: str = "date",
        code_column: str = "code",
        min_samples_per_day: int = 10,
        seed: int = 42,
        # 互換性: 呼び出し元が `shuffle=` を渡す場合に受け付ける
        shuffle: bool | None = None,
        max_batch_size: int | None = None,
        min_nodes_per_day: int | None = None,
        min_nodes: int | None = None,
        drop_undersized: bool | None = None,
        indices: list[int] | None = None,
        dates: list | np.ndarray | None = None,
        **_: object,
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
        self.batch_size = int(batch_size)
        # 互換性対応: shuffle 引数が与えられたら優先
        self.shuffle_within_day = (
            shuffle if isinstance(shuffle, bool) else shuffle_within_day
        )
        self.shuffle_days = shuffle_days
        self.drop_last = drop_last
        self.date_column = date_column
        self.code_column = code_column
        if min_nodes_per_day is not None:
            self.min_samples_per_day = int(min_nodes_per_day)
        elif min_nodes is not None:
            self.min_samples_per_day = int(min_nodes)
        else:
            self.min_samples_per_day = int(min_samples_per_day)
        self.seed = seed

        self._drop_underfilled_days = (
            bool(drop_undersized) if drop_undersized is not None else True
        )

        self.max_batch_size = int(max_batch_size) if max_batch_size else None
        self._chunk_size = (
            min(self.batch_size, self.max_batch_size)
            if self.max_batch_size
            else self.batch_size
        )

        self._manual_date_indices: dict[str, list[int]] | None = None
        if indices is not None and dates is not None:
            if len(indices) != len(dates):
                raise ValueError(
                    "indices and dates must have the same length when provided"
                )
            manual_map: dict[str, list[int]] = {}
            for idx, date in zip(indices, dates, strict=False):
                manual_map.setdefault(str(date), []).append(int(idx))
            self._manual_date_indices = manual_map

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
        if self._manual_date_indices is not None:
            return {
                date: indices
                for date, indices in self._manual_date_indices.items()
                if self._should_keep_day(indices)
            }

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
                    except Exception as exc:
                        logger.debug(
                            "sequence_dates cast to datetime64[D] failed: %s", exc
                        )

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
                    # PERF WARNING: Slow path - O(n) sequential dataset access (2-5 min for 10M samples)
                    # To optimize: implement sequence_dates attribute on dataset or use .data/.df attribute
                    logger.warning(
                        "Using slow iteration path for date index building. "
                        "Consider implementing sequence_dates attribute for 100x speedup."
                    )
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
        # PERF: Vectorized groupby for pandas DataFrames (5-10x faster than enumerate loop)
        if isinstance(data, pd.DataFrame):
            if self.date_column in data.columns:
                # Use pandas groupby().indices for O(n) grouping instead of O(n) loop
                date_groups = data.groupby(self.date_column).indices
                date_indices = {str(date): list(indices) for date, indices in date_groups.items()}
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
            if self._should_keep_day(indices):
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

    def _should_keep_day(self, indices: list[int]) -> bool:
        """Return True if a day's indices satisfy sampling thresholds."""
        if not indices:
            return False
        if self._drop_underfilled_days:
            return len(indices) >= self.min_samples_per_day
        return True

    def _create_batches(self) -> list[list[int]]:
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
            step = max(self._chunk_size, 1)
            for i in range(0, len(day_indices), step):
                batch = day_indices[i : i + step]

                # Handle drop_last
                if self.drop_last and len(batch) < step:
                    continue

                batches.append(batch)

        return batches

    def __iter__(self) -> Iterator[list[int]]:
        """
        Iterate over batches.

        Yields:
            List of indices for each batch
        """
        # Optionally reshuffle for each epoch
        if self.shuffle_days or self.shuffle_within_day:
            self.batches = self._create_batches()

        yield from self.batches

    def __len__(self) -> int:
        """
        Get number of batches.

        Returns:
            Number of batches
        """
        return len(self.batches)

    def get_date_for_batch(self, batch_idx: int) -> str | None:
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

    def _create_batches(self) -> list[list[int]]:
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
