from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


@dataclass
class PanelCache:
    """Pre-computed panel cache for efficient DayPanelDataset creation."""

    date_ints: list[int]
    date_to_codes: dict[int, list[str]]
    codes: dict[str, Mapping[str, np.ndarray]]
    lookback: int
    min_stocks: int
    target_columns: list[str]


def _date_series_to_int(series: pl.Series) -> np.ndarray:
    """Convert a Polars date/datetime series to ``datetime64[D]`` ints."""
    values = series.to_numpy()
    dates = values.astype("datetime64[D]")
    return dates.astype("int64")


def build_panel_cache(
    frame: pl.DataFrame,
    *,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    mask_cols: Sequence[str],
    date_col: str,
    code_col: str,
    lookback: int,
    min_stocks_per_day: int,
) -> PanelCache:
    """Construct reusable cache for panel datasets."""

    if lookback <= 0:
        raise ValueError("lookback must be positive")

    sorted_df = frame.sort([date_col, code_col])
    codes_data: dict[str, dict[str, np.ndarray]] = {}

    for code_df in sorted_df.partition_by(code_col, maintain_order=True):
        code = code_df[0, code_col]
        if isinstance(code, bytes):
            code = code.decode("utf-8")
        dates_int = _date_series_to_int(code_df[date_col])
        feat_arr = code_df.select(feature_cols).to_numpy()
        targ_arr = code_df.select(target_cols).to_numpy() if target_cols else None
        mask_arr = code_df.select(mask_cols).to_numpy() if mask_cols else None

        codes_data[str(code)] = {
            "dates": dates_int,
            "features": feat_arr.astype(np.float32, copy=False),
            "targets": None if targ_arr is None else targ_arr.astype(np.float32, copy=False),
            "masks": None if mask_arr is None else mask_arr.astype(np.float32, copy=False),
        }

    unique_dates = sorted_df.select(pl.col(date_col).unique()).to_series().to_numpy().astype("datetime64[D]")
    unique_date_ints = np.unique(unique_dates.astype("int64"))

    date_to_codes: dict[int, list[str]] = {}
    for date_int in unique_date_ints:
        eligible: list[str] = []
        for code, payload in codes_data.items():
            dates = payload["dates"]
            idx = np.searchsorted(dates, date_int)
            if idx == len(dates) or dates[idx] != date_int:
                continue
            start = idx - lookback + 1
            if start < 0:
                continue
            window = payload["features"][start : idx + 1]

            # Check for NaN in features
            if np.isnan(window).any():
                continue

            # Check for NaN in targets (only if targets exist)
            if payload["targets"] is not None:
                targets = payload["targets"][idx]
                if np.isnan(targets).any():
                    continue
            if payload["masks"] is not None:
                if np.any(payload["masks"][idx] == 0):
                    continue
            eligible.append(code)

        if len(eligible) >= min_stocks_per_day:
            date_to_codes[int(date_int)] = eligible

    cached_dates = sorted(date_to_codes.keys())
    return PanelCache(
        date_ints=cached_dates,
        date_to_codes=date_to_codes,
        codes=codes_data,
        lookback=lookback,
        min_stocks=min_stocks_per_day,
        target_columns=list(target_cols),
    )


class DayPanelDataset(Dataset):
    """Dataset returning day-level panels for ranking losses."""

    def __init__(
        self,
        cache: PanelCache,
        *,
        feature_cols: Sequence[str],
        mask_cols: Sequence[str],
        target_cols: Sequence[str],
        dates_subset: Iterable[int] | None = None,
    ) -> None:
        self.cache = cache
        self.feature_cols = list(feature_cols)
        self.mask_cols = list(mask_cols)
        self.target_cols = list(target_cols)

        if dates_subset is None:
            self.dates = cache.date_ints
        else:
            subset = [int(d) for d in dates_subset if int(d) in cache.date_to_codes]
            self.dates = sorted(subset)

    def __len__(self) -> int:
        return len(self.dates)

    def __getitem__(self, index: int) -> dict | None:
        date_int = self.dates[index]
        codes = self.cache.date_to_codes.get(date_int, [])
        if not codes:
            return None

        X_list: list[np.ndarray] = []
        Y_list: list[np.ndarray] = []
        valid_codes: list[str] = []

        for code in codes:
            payload = self.cache.codes[code]
            dates = payload["dates"]
            idx = np.searchsorted(dates, date_int)
            if idx == len(dates) or dates[idx] != date_int:
                continue
            start = idx - self.cache.lookback + 1
            if start < 0:
                continue

            window = payload["features"][start : idx + 1]
            targets = payload["targets"][idx]

            if np.isnan(window).any() or np.isnan(targets).any():
                continue

            if payload["masks"] is not None:
                mask_row = payload["masks"][idx]
                if np.any(mask_row == 0):
                    continue

            X_list.append(window)
            Y_list.append(targets)
            valid_codes.append(code)

        if not X_list:
            return None

        features = np.stack(X_list, axis=0).astype(np.float32, copy=False)
        targets = np.stack(Y_list, axis=0).astype(np.float32, copy=False)

        return {
            "X": torch.from_numpy(features),
            "y": torch.from_numpy(targets),
            "codes": valid_codes,
            "date_int": date_int,
        }


def collate_day_batch(batch: Sequence[dict | None]) -> dict | None:
    """Collate function for day-level datasets."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    sample = batch[0]
    return sample


def panel_cache_key(
    dataset_path: Path,
    *,
    lookback: int,
    feature_cols: Sequence[str],
    version: str = "v1",
    extra_salt: str | None = None,
) -> str:
    """Generate a deterministic key for panel cache persistence."""
    resolved = Path(dataset_path).resolve()
    payload = {
        "dataset": str(resolved),
        "lookback": int(lookback),
        "feature_count": len(feature_cols),
        "features": list(feature_cols),
        "version": version,
    }
    if extra_salt:
        payload["salt"] = extra_salt

    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    digest = hashlib.sha1(encoded).hexdigest()
    stem = resolved.stem or resolved.name
    return f"{stem}_lb{lookback}_f{len(feature_cols)}_{digest[:10]}"


def save_panel_cache(cache: PanelCache, path: str | Path) -> None:
    """Persist a ``PanelCache`` to disk using pickle serialization."""
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as fh:
        pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_panel_cache(path: str | Path) -> PanelCache:
    """Load a ``PanelCache`` previously saved with :func:`save_panel_cache`."""
    cache_path = Path(path)
    with cache_path.open("rb") as fh:
        cache = pickle.load(fh)
    if not isinstance(cache, PanelCache):
        raise TypeError(f"Loaded object is not a PanelCache: {type(cache)!r}")
    return cache
