from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Fold:
    train_idx: np.ndarray
    val_idx: np.ndarray


def purged_kfold_indices(
    dates: Sequence[np.datetime64] | Sequence[str],
    *,
    n_splits: int = 5,
    embargo_days: int = 20,
) -> List[Fold]:
    """Time-ordered KFold with an embargo around validation windows.

    Parameters
    ----------
    dates : sequence of dates (ISO strings or numpy datetime64)
        One date per row; assumed pre-sorted by time within each instrument panel.
    n_splits : int
        Number of folds.
    embargo_days : int
        Number of calendar days to exclude on both sides of validation windows.
    """
    # Convert to days ordinal for simple arithmetic
    d_ord = np.array(dates, dtype="datetime64[D]").astype("datetime64[D]")
    unique_days = np.unique(d_ord)
    splits = np.array_split(unique_days, n_splits)
    folds: List[Fold] = []
    for val_days in splits:
        if val_days.size == 0:
            continue
        start = val_days[0] - np.timedelta64(embargo_days, "D")
        end = val_days[-1] + np.timedelta64(embargo_days, "D")
        val_mask = np.isin(d_ord, val_days)
        embargo_mask = (d_ord >= start) & (d_ord <= end)
        train_mask = ~embargo_mask & ~val_mask
        train_idx = np.nonzero(train_mask)[0]
        val_idx = np.nonzero(val_mask)[0]
        folds.append(Fold(train_idx=train_idx, val_idx=val_idx))
    return folds


__all__ = ["Fold", "purged_kfold_indices"]

