"""Walk-forward split generator for backtesting."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date

import numpy as np
import polars as pl


@dataclass
class Split:
    """Single train/validation split."""

    fold_id: int
    train_start: Date
    train_end: Date
    val_start: Date
    val_end: Date

    def __repr__(self) -> str:
        return (
            f"Split(fold={self.fold_id}, "
            f"train={self.train_start}~{self.train_end}, "
            f"val={self.val_start}~{self.val_end})"
        )


class WalkForwardSplitter:
    """
    Walk-forward cross-validation splitter.

    Generates rolling train/validation splits with configurable window sizes.
    Handles non-trading days and missing data gracefully.
    """

    def __init__(
        self,
        train_days: int = 252,
        val_days: int = 63,
        step_days: int = 21,
        min_train_days: int = 180,
        min_val_days: int = 20,
    ):
        """
        Initialize splitter.

        Args:
            train_days: Training window size in trading days (default: 252 = 1 year)
            val_days: Validation window size in trading days (default: 63 = 3 months)
            step_days: Step size between folds in trading days (default: 21 = 1 month)
            min_train_days: Minimum training days required (default: 180)
            min_val_days: Minimum validation days required (default: 20)
        """
        self.train_days = train_days
        self.val_days = val_days
        self.step_days = step_days
        self.min_train_days = min_train_days
        self.min_val_days = min_val_days

    def get_trading_dates(
        self,
        dataset: pl.DataFrame,
        date_col: str = "Date",
    ) -> list[Date]:
        """
        Extract unique sorted trading dates from dataset.

        Args:
            dataset: DataFrame with date column
            date_col: Name of date column

        Returns:
            List of trading dates (sorted)
        """
        dates = dataset[date_col].unique().sort().to_list()
        return [d if isinstance(d, Date) else d.date() for d in dates]

    def split(
        self,
        trading_dates: list[Date],
        start_date: Date | None = None,
        end_date: Date | None = None,
    ) -> list[Split]:
        """
        Generate walk-forward splits.

        Args:
            trading_dates: List of available trading dates
            start_date: Start date for splits (default: first date + train_days)
            end_date: End date for splits (default: last date)

        Returns:
            List of Split objects
        """
        if len(trading_dates) < self.min_train_days + self.min_val_days:
            raise ValueError(
                f"Insufficient data: {len(trading_dates)} dates, "
                f"need at least {self.min_train_days + self.min_val_days}"
            )

        # Filter by date range
        if start_date:
            trading_dates = [d for d in trading_dates if d >= start_date]
        if end_date:
            trading_dates = [d for d in trading_dates if d <= end_date]

        splits = []
        fold_id = 1

        # Initial training window
        train_start_idx = 0
        train_end_idx = min(self.train_days - 1, len(trading_dates) - self.val_days - 1)

        while train_end_idx < len(trading_dates) - self.min_val_days:
            # Validation window follows training window
            val_start_idx = train_end_idx + 1
            val_end_idx = min(val_start_idx + self.val_days - 1, len(trading_dates) - 1)

            # Skip if validation window too small
            if val_end_idx - val_start_idx + 1 < self.min_val_days:
                break

            # Create split
            split = Split(
                fold_id=fold_id,
                train_start=trading_dates[train_start_idx],
                train_end=trading_dates[train_end_idx],
                val_start=trading_dates[val_start_idx],
                val_end=trading_dates[val_end_idx],
            )
            splits.append(split)

            # Move to next fold
            fold_id += 1
            train_start_idx += self.step_days
            train_end_idx = min(
                train_start_idx + self.train_days - 1,
                len(trading_dates) - self.val_days - 1,
            )

            # Stop if training window too small
            if train_end_idx - train_start_idx + 1 < self.min_train_days:
                break

        return splits

    def split_from_dataset(
        self,
        dataset: pl.DataFrame,
        date_col: str = "Date",
        start_date: Date | None = None,
        end_date: Date | None = None,
    ) -> list[Split]:
        """
        Generate splits directly from dataset.

        Args:
            dataset: DataFrame with date column
            date_col: Name of date column
            start_date: Start date for splits
            end_date: End date for splits

        Returns:
            List of Split objects
        """
        trading_dates = self.get_trading_dates(dataset, date_col)
        return self.split(trading_dates, start_date, end_date)

    def get_date_masks(
        self,
        split: Split,
        dataset: pl.DataFrame,
        date_col: str = "Date",
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Get train/validation data masks for a split.

        Args:
            split: Split object
            dataset: Full dataset
            date_col: Name of date column

        Returns:
            (train_data, val_data) DataFrames
        """
        # Convert to polars date format if needed
        train_mask = (pl.col(date_col) >= split.train_start) & (pl.col(date_col) <= split.train_end)
        val_mask = (pl.col(date_col) >= split.val_start) & (pl.col(date_col) <= split.val_end)

        train_data = dataset.filter(train_mask)
        val_data = dataset.filter(val_mask)

        return train_data, val_data

    def summary(self, splits: list[Split]) -> dict:
        """
        Generate summary statistics for splits.

        Args:
            splits: List of Split objects

        Returns:
            Dict with summary statistics
        """
        if not splits:
            return {"num_splits": 0}

        train_sizes = [(split.train_end - split.train_start).days + 1 for split in splits]
        val_sizes = [(split.val_end - split.val_start).days + 1 for split in splits]

        return {
            "num_splits": len(splits),
            "date_range": {
                "start": str(splits[0].train_start),
                "end": str(splits[-1].val_end),
            },
            "train_window": {
                "target_days": self.train_days,
                "actual_days": {
                    "mean": np.mean(train_sizes),
                    "min": min(train_sizes),
                    "max": max(train_sizes),
                },
            },
            "val_window": {
                "target_days": self.val_days,
                "actual_days": {
                    "mean": np.mean(val_sizes),
                    "min": min(val_sizes),
                    "max": max(val_sizes),
                },
            },
            "step_size": self.step_days,
        }


def generate_splits_for_backtest(
    dataset_path: str,
    start_date: str | None = None,
    end_date: str | None = None,
    train_days: int = 252,
    val_days: int = 63,
    step_days: int = 21,
) -> list[Split]:
    """
    Convenience function to generate splits from dataset file.

    Args:
        dataset_path: Path to parquet dataset
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)
        train_days: Training window size
        val_days: Validation window size
        step_days: Step size between folds

    Returns:
        List of Split objects
    """
    from datetime import datetime

    dataset = pl.scan_parquet(dataset_path).select(["Date"]).collect()

    start = datetime.strptime(start_date, "%Y-%m-%d").date() if start_date else None
    end = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else None

    splitter = WalkForwardSplitter(
        train_days=train_days,
        val_days=val_days,
        step_days=step_days,
    )

    return splitter.split_from_dataset(dataset, start_date=start, end_date=end)


@dataclass(frozen=True)
class PurgeParams:
    """Bundle of lookback / horizon / embargo days used to compute purge gaps."""

    lookback_days: int
    max_horizon_days: int
    embargo_days: int = 0

    @property
    def purge_days(self) -> int:
        """Total trading-day purge applied around each test block."""
        return int(self.lookback_days + self.max_horizon_days + self.embargo_days)


class PurgedKFoldSplitter:
    """
    Time-ordered K-fold splitter with purge + embargo.

    The positional gap between any training day and any validation day is at least
    ``lookback + max_horizon + embargo`` trading days, eliminating feature/label
    leakage when using rolling windows.
    """

    def __init__(
        self,
        *,
        n_splits: int,
        params: PurgeParams,
        min_train_size: int = 60,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2 for PurgedKFoldSplitter")
        self.n_splits = int(n_splits)
        self.params = params
        self.min_train_size = int(min_train_size)

    def split(
        self,
        days: np.ndarray | list[int] | list[Date],
        fold_index: int,
    ):
        """
        Yield a single (train_idx, test_idx) pair for the requested fold.

        Args:
            days: Sorted unique trading-day sequence (positional order matters).
            fold_index: 1-based fold selector.
        """
        array = np.asarray(days)
        if array.ndim != 1:
            raise ValueError("days must be a 1-D sequence")
        if array.size < self.n_splits:
            raise ValueError(f"Insufficient days ({array.size}) for {self.n_splits}-fold split")
        if np.any(np.diff(array) < 0):
            raise ValueError("days must be sorted ascending")

        fold_index = int(fold_index)
        if not (1 <= fold_index <= self.n_splits):
            raise ValueError(f"fold_index must be within [1, {self.n_splits}], got {fold_index}")

        n = array.size
        block = n // self.n_splits
        if block == 0:
            raise ValueError("Too few samples per fold; increase dataset length")

        test_start = (fold_index - 1) * block
        test_end = (fold_index * block) if fold_index < self.n_splits else n

        test_mask = np.zeros(n, dtype=bool)
        test_mask[test_start:test_end] = True

        purge = self.params.purge_days
        gap_start = max(0, test_start - purge)
        gap_end = min(n, test_end + purge)

        gap_mask = np.zeros(n, dtype=bool)
        gap_mask[gap_start:gap_end] = True

        train_mask = ~gap_mask
        train_size = int(train_mask.sum())
        if train_size < self.min_train_size:
            raise RuntimeError(
                "Not enough training samples after purge/embargo: "
                f"{train_size} < {self.min_train_size} (purge_days={purge})"
            )

        train_idx = np.nonzero(train_mask)[0]
        test_idx = np.nonzero(test_mask)[0]

        if train_idx.size and test_idx.size:
            # Positional distance between any train/test index must exceed purge.
            min_dist = np.abs(train_idx[:, None] - test_idx[None, :]).min()
            assert min_dist >= purge, f"Leakage detected: min distance {min_dist} < purge {purge}"

        yield train_idx, test_idx
