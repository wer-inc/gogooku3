from __future__ import annotations

import numpy as np

from apex_ranker.backtest.splitter import PurgeParams, PurgedKFoldSplitter


def test_purged_kfold_splitter_ensures_gap() -> None:
    days = np.arange(1000)
    params = PurgeParams(lookback_days=180, max_horizon_days=20, embargo_days=5)
    splitter = PurgedKFoldSplitter(n_splits=5, params=params, min_train_size=50)
    purge = params.purge_days

    for fold in range(1, 6):
        train_idx, test_idx = next(splitter.split(days, fold))
        assert set(train_idx).isdisjoint(set(test_idx))
        min_distance = np.abs(train_idx[:, None] - test_idx[None, :]).min()
        assert (
            min_distance >= purge
        ), f"Fold {fold} leaked with min distance {min_distance} < {purge}"


def test_purged_kfold_splitter_min_train_guard() -> None:
    days = np.arange(220)
    params = PurgeParams(lookback_days=180, max_horizon_days=20, embargo_days=5)
    splitter = PurgedKFoldSplitter(n_splits=5, params=params, min_train_size=10)

    try:
        next(splitter.split(days, 1))
    except RuntimeError:
        return

    raise AssertionError("Expected RuntimeError when training window collapses")
