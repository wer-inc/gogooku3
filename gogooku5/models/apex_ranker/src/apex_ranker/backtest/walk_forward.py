"""Walk-forward cross-validation utilities for backtesting."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import polars as pl


@dataclass
class WalkForwardFold:
    """Single walk-forward train/test split."""

    fold_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    train_days: int
    test_days: int

    def __repr__(self) -> str:
        return (
            f"Fold {self.fold_id}: "
            f"Train[{self.train_start} → {self.train_end}, {self.train_days}d] "
            f"Test[{self.test_start} → {self.test_end}, {self.test_days}d]"
        )


class WalkForwardSplitter:
    """
    Generate walk-forward train/test splits.

    Supports both rolling and expanding window strategies with optional gap days
    between train/test periods to avoid leakage.
    """

    def __init__(
        self,
        train_days: int = 252,
        test_days: int = 63,
        step_days: int = 21,
        mode: Literal["rolling", "expanding"] = "rolling",
        min_train_days: int = 126,
        gap_days: int = 0,
    ):
        if train_days < min_train_days:
            raise ValueError(
                f"train_days ({train_days}) must be >= min_train_days ({min_train_days})"
            )
        if test_days < 1:
            raise ValueError(f"test_days ({test_days}) must be >= 1")
        if step_days < 1:
            raise ValueError(f"step_days ({step_days}) must be >= 1")
        if gap_days < 0:
            raise ValueError(f"gap_days ({gap_days}) must be >= 0")

        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days
        self.mode = mode
        self.min_train_days = min_train_days
        self.gap_days = gap_days

    def _ensure_dates(self, dates: pl.Series | list[date]) -> list[date]:
        if isinstance(dates, pl.Series):
            return sorted(dates.to_list())
        return sorted(dates)

    def split(self, dates: pl.Series | list[date]) -> list[WalkForwardFold]:
        """Generate walk-forward train/test splits."""
        date_list = self._ensure_dates(dates)

        if not date_list:
            raise ValueError("dates cannot be empty")

        min_required = self.train_days + self.gap_days + self.test_days
        if len(date_list) < min_required:
            raise ValueError(
                "Insufficient dates: need at least "
                f"{min_required} (train={self.train_days} + gap={self.gap_days} "
                f"+ test={self.test_days}), got {len(date_list)}"
            )

        folds: list[WalkForwardFold] = []
        fold_id = 1
        train_start_idx = 0

        while True:
            if self.mode == "rolling":
                train_end_idx = train_start_idx + self.train_days - 1
            else:  # expanding window
                train_end_idx = (
                    train_start_idx
                    + self.train_days
                    + (fold_id - 1) * self.step_days
                    - 1
                )

            if train_end_idx >= len(date_list):
                break

            test_start_idx = train_end_idx + 1 + self.gap_days
            test_end_idx = test_start_idx + self.test_days - 1

            if test_end_idx >= len(date_list):
                break

            fold = WalkForwardFold(
                fold_id=fold_id,
                train_start=date_list[train_start_idx],
                train_end=date_list[train_end_idx],
                test_start=date_list[test_start_idx],
                test_end=date_list[test_end_idx],
                train_days=train_end_idx - train_start_idx + 1,
                test_days=test_end_idx - test_start_idx + 1,
            )
            folds.append(fold)

            train_start_idx += self.step_days
            fold_id += 1

        if not folds:
            raise ValueError(
                "Could not generate any folds. "
                f"Try reducing train_days ({self.train_days}) or test_days "
                f"({self.test_days}), or increasing available date range "
                f"(current: {len(date_list)} days)"
            )

        return folds

    def get_train_dates(
        self,
        fold: WalkForwardFold,
        dates: pl.Series | list[date],
    ) -> list[date]:
        """Return training dates for a fold."""
        date_list = self._ensure_dates(dates)
        return [d for d in date_list if fold.train_start <= d <= fold.train_end]

    def get_test_dates(
        self,
        fold: WalkForwardFold,
        dates: pl.Series | list[date],
    ) -> list[date]:
        """Return test dates for a fold."""
        date_list = self._ensure_dates(dates)
        return [d for d in date_list if fold.test_start <= d <= fold.test_end]

    def summary(self, dates: pl.Series | list[date]) -> dict:
        """Return summary statistics for generated folds."""
        folds = self.split(dates)
        date_list = self._ensure_dates(dates)

        return {
            "total_folds": len(folds),
            "mode": self.mode,
            "train_days_config": self.train_days,
            "test_days_config": self.test_days,
            "step_days": self.step_days,
            "gap_days": self.gap_days,
            "date_range": {
                "start": date_list[0],
                "end": date_list[-1],
                "total_days": len(date_list),
            },
            "first_fold": {
                "train": f"{folds[0].train_start} → {folds[0].train_end}",
                "test": f"{folds[0].test_start} → {folds[0].test_end}",
            },
            "last_fold": {
                "train": f"{folds[-1].train_start} → {folds[-1].train_end}",
                "test": f"{folds[-1].test_start} → {folds[-1].test_end}",
            },
            "coverage": {
                "train_min_days": min(f.train_days for f in folds),
                "train_max_days": max(f.train_days for f in folds),
                "test_min_days": min(f.test_days for f in folds),
                "test_max_days": max(f.test_days for f in folds),
            },
        }
