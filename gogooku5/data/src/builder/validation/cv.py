"""Time-series cross validation helpers with purge + embargo."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable, Iterator, List, Sequence, Tuple

try:
    import polars as pl
except ImportError:  # pragma: no cover - optional dependency
    pl = None  # type: ignore[assignment]


def _ensure_datetime(value: date | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    raise TypeError(f"Unsupported date type: {type(value)!r}")


def _extract_column(data: object, column: str) -> Sequence:
    if pl is not None and isinstance(data, pl.DataFrame):
        if column not in data.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame")
        return data[column].to_list()
    if hasattr(data, "__getitem__"):
        try:
            return data[column]
        except Exception as exc:  # pragma: no cover - defensive
            raise KeyError(f"Column '{column}' not found in container") from exc
    raise TypeError(
        "Unsupported container type. Pass a Polars DataFrame or an object implementing __getitem__ for columns."
    )


@dataclass
class PurgedGroupTimeSeriesSplit:
    """
    Time-series splitter that removes leakage via purge + embargo windows.

    Based on López de Prado (2018) PurgedKFold. Each split uses contiguous
    validation blocks in chronological order while:

    - Purging `purge_days` immediately before the validation start.
    - Applying an embargo of `embargo_days` after the validation end.
    - Removing any training rows that share the same group (code) as the validation set.
    """

    n_splits: int
    purge_days: int = 0
    embargo_days: int = 20
    time_column: str = "date"
    group_column: str = "code"

    def split(
        self,
        X: object,
        y: Sequence | None = None,  # parity with sklearn API
        groups: Sequence | None = None,
    ) -> Iterator[Tuple[List[int], List[int]]]:
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")

        dates_seq = _extract_column(X, self.time_column)
        if groups is None:
            groups = _extract_column(X, self.group_column)

        if len(dates_seq) != len(groups):
            raise ValueError("dates and groups must have equal length")

        row_dates = [_ensure_datetime(d) for d in dates_seq]
        row_groups = list(groups)

        unique_dates = sorted(dict.fromkeys(row_dates))
        if self.n_splits > len(unique_dates):
            raise ValueError("n_splits cannot exceed the number of unique dates")

        fold_boundaries = _split_indices(len(unique_dates), self.n_splits)

        for start_idx, end_idx in fold_boundaries:
            fold_dates = unique_dates[start_idx:end_idx]
            if not fold_dates:
                continue
            val_start = fold_dates[0]
            val_end = fold_dates[-1]
            val_date_set = set(fold_dates)

            val_idx = [i for i, dt in enumerate(row_dates) if dt in val_date_set]
            val_groups = {row_groups[i] for i in val_idx}

            purge_start = val_start - timedelta(days=self.purge_days)
            embargo_end = val_end + timedelta(days=self.embargo_days)

            train_idx = [
                i
                for i, dt in enumerate(row_dates)
                if (
                    (dt < purge_start or dt > embargo_end)
                    and row_groups[i] not in val_groups
                )
            ]

            if not train_idx or not val_idx:
                raise ValueError(
                    "Empty train/validation split encountered; "
                    "adjust n_splits or reduce purge/embargo windows."
                )

            yield train_idx, val_idx

    def get_n_splits(self) -> int:
        return self.n_splits


def _split_indices(n: int, n_splits: int) -> List[Tuple[int, int]]:
    """Return start/end index tuples that partition ``range(n)`` into ``n_splits`` contiguous blocks."""
    base = n // n_splits
    remainder = n % n_splits
    boundaries: List[Tuple[int, int]] = []
    start = 0
    for fold in range(n_splits):
        extra = 1 if fold < remainder else 0
        end = start + base + extra
        boundaries.append((start, end))
        start = end
    return boundaries
