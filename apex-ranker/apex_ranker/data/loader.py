"""Dataset loading helpers for backtesting and inference."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl


def load_backtest_frame(
    data_path: Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    feature_cols: list[str] | None = None,
    lookback: int = 0,
) -> pl.DataFrame:
    """Load parquet dataset with required columns for inference/backtests."""

    print(f"[Backtest] Loading dataset: {data_path}")

    required_cols: set[str] = {
        "Date",
        "Code",
        "Close",
        "Volume",
        "TurnoverValue",
        "returns_1d",
        "returns_5d",
        "returns_10d",
        "returns_20d",
    }

    if feature_cols:
        required_cols.update(feature_cols)

    frame = pl.read_parquet(str(data_path), columns=list(required_cols))
    frame = frame.sort(["Date", "Code"])

    start_dt = np.datetime64(start_date) if start_date else None
    end_dt = np.datetime64(end_date) if end_date else None

    buffer_start = None
    if lookback > 0 and start_dt is not None:
        all_dates = frame["Date"].unique().sort().to_numpy()
        idx = np.searchsorted(all_dates, start_dt)
        if idx == 0:
            buffer_idx = 0
        else:
            buffer_idx = max(0, idx - lookback)
        if all_dates.size > 0:
            buffer_start = all_dates[buffer_idx]

    lower_bound = buffer_start if buffer_start is not None else start_dt
    if lower_bound is not None:
        frame = frame.filter(pl.col("Date") >= lower_bound)
    if end_dt is not None:
        frame = frame.filter(pl.col("Date") <= end_dt)

    print(f"[Backtest] Loaded {len(frame):,} rows")
    print(
        "[Backtest] Date span:",
        frame["Date"].min(),
        "→",
        frame["Date"].max(),
    )
    print(f"[Backtest] Unique stocks: {frame['Code'].n_unique()}")

    return frame
