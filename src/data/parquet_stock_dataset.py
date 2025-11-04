from __future__ import annotations

"""
Streaming Parquet dataset utilities used by Phase 1 refresh.

`ParquetStockIterableDataset` reads row-groups lazily, assembles sliding windows
per銘柄 (Code) and yields samples as dictionaries that mirror the ATFT training
pipeline expectations (`features`, `targets`, `date`, `code`, など)。

It relies on `OnlineRobustScaler` to approximate per-feature median/MAD via
reservoirサンプリング so that正規化コストを一定に保てる。
"""

import logging
from collections.abc import Generator, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import IterableDataset, get_worker_info

try:  # pragma: no cover - optional dependency for row-group streaming
    import pyarrow.parquet as pq

    _HAS_PYARROW = True
except Exception:  # pragma: no cover
    pq = None  # type: ignore
    _HAS_PYARROW = False

NUMPY_EPS = np.finfo(np.float32).eps

# P0-3: Non-numeric column exclusion for safe tensor conversion
_ALWAYS_EXCLUDE_COLS = {
    "Code", "LocalCode", "Section", "MarketCode",
    "Date", "date", "trade_date",
    "dmi_published_date", "dmi_application_date",
    "dmi_publish_reason",
    "sector17_code", "sector17_name", "sector33_code", "sector33_name",
    "section_norm",  # String type
}

logger = logging.getLogger(__name__)


def _select_numeric_columns_polars(
    df: pl.DataFrame,
    candidate_cols: list[str],
) -> tuple[list[str], dict]:
    """
    Filter candidate columns to only numeric types (int, float, bool).
    Exclude datetime, string, struct types and meta columns.

    Returns:
        (selected_cols, diagnostics_dict)
    """
    schema = df.schema
    selected = []
    dropped_non_numeric = []
    casted_bool = []

    for col in candidate_cols:
        if col in _ALWAYS_EXCLUDE_COLS:
            continue

        if col not in schema:
            dropped_non_numeric.append(col)
            continue

        dtype = schema[col]

        # Accept numeric types
        if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                     pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                     pl.Float32, pl.Float64):
            selected.append(col)
        # Accept boolean (will be cast to 0/1)
        elif dtype == pl.Boolean:
            selected.append(col)
            casted_bool.append(col)
        # Reject datetime, string, struct, etc.
        else:
            dropped_non_numeric.append(col)

    diag = {
        "selected": selected,
        "dropped_non_numeric": dropped_non_numeric,
        "casted_bool_to_float": casted_bool,
        "excluded_meta": sorted(_ALWAYS_EXCLUDE_COLS),
    }

    return selected, diag


class OnlineRobustScaler:
    """
    Online approximation of median / MAD using reservoir sampling.

    For large Parquet shards this avoids materialising全サンプル while still提供
    stable正規化パラメータ.
    """

    def __init__(self, max_samples: int = 200_000, random_state: int | None = 42):
        self.max_samples = max_samples
        self._rng = np.random.default_rng(random_state)
        self._count = 0
        self._reservoir: np.ndarray | None = None
        self.median_: np.ndarray | None = None
        self.mad_: np.ndarray | None = None
        self._random_state = random_state

    def partial_fit(self, batch: np.ndarray) -> None:
        if batch.size == 0:
            return
        batch = batch.reshape(-1, batch.shape[-1])
        batch = np.nan_to_num(batch, nan=0.0, posinf=0.0, neginf=0.0)
        if self._reservoir is None:
            take = min(batch.shape[0], self.max_samples)
            self._reservoir = batch[:take].copy()
            self._count = take
            start = take
        else:
            start = 0
        remaining = self.max_samples - self._reservoir.shape[0]
        if remaining > 0:
            add = min(remaining, batch[start:].shape[0])
            if add > 0:
                self._reservoir = np.vstack(
                    [self._reservoir, batch[start : start + add]]
                )
                start += add
                self._count += add
        for row in batch[start:]:
            j = self._rng.integers(0, self._count + 1)
            if j < self.max_samples:
                self._reservoir[j] = row
            self._count += 1

    def finalise(self) -> None:
        if self._reservoir is None or self._reservoir.size == 0:
            raise RuntimeError(
                "OnlineRobustScaler received no data. Call partial_fit first."
            )
        arr = self._reservoir
        self.median_ = np.median(arr, axis=0)
        mad = np.median(np.abs(arr - self.median_), axis=0) * 1.4826
        mad = np.where(mad < NUMPY_EPS, 1.0, mad)
        self.mad_ = mad
        # Release reservoir to free memory once stats computed
        self._reservoir = None

    def transform(self, batch: np.ndarray) -> np.ndarray:
        if self.median_ is None or self.mad_ is None:
            raise RuntimeError("Scaler not fitted. Call finalise() first.")
        return np.clip((batch - self.median_) / self.mad_, -5.0, 5.0)

    @property
    def is_fitted(self) -> bool:
        return self.median_ is not None and self.mad_ is not None

    def clone(self) -> OnlineRobustScaler:
        """
        Create an immutable copy carrying over fitted statistics.

        The clone shares no mutable buffers with the source scaler to avoid
        accidental cross-dataset mutations (e.g. during validation).
        """
        clone = OnlineRobustScaler(
            max_samples=self.max_samples,
            random_state=self._random_state,
        )
        if self.median_ is not None:
            clone.median_ = self.median_.copy()
        if self.mad_ is not None:
            clone.mad_ = self.mad_.copy()
        clone._count = self._count
        clone._reservoir = None
        return clone


@dataclass
class Sample:
    features: torch.Tensor
    targets: dict[str, torch.Tensor]
    code: str
    date: str


class ParquetStockIterableDataset(IterableDataset):
    """
    IterableDataset that streams Parquet row-groups, yields sliding windows and
    applies online robust正規化.
    """

    def __init__(
        self,
        file_paths: Sequence[Path],
        feature_columns: Sequence[str],
        target_columns: Sequence[str],
        code_column: str = "Code",
        date_column: str = "Date",
        sequence_length: int = 60,
        scaler: OnlineRobustScaler | None = None,
    ):
        super().__init__()
        self.file_paths = [Path(p) for p in file_paths]
        self.feature_columns = list(feature_columns)
        self.target_columns = list(target_columns)
        self.code_column = code_column
        self.date_column = date_column
        self.sequence_length = sequence_length
        self._scaler = scaler or OnlineRobustScaler()
        self._fitted = False
        self._columns_needed = list(
            dict.fromkeys(
                [self.code_column, self.date_column]
                + self.feature_columns
                + self.target_columns
            )
        )

    # ------------------------------------------------------------------ Fitting
    def fit(
        self,
        max_samples: int = 200_000,
        *,
        max_files: int | None = None,
    ) -> None:
        scaler = self._scaler
        if max_files is not None and max_files > 0:
            candidate_files = self.file_paths[: max_files]
        else:
            candidate_files = self.file_paths
        for window in self._stream_windows(candidate_files, yield_windows=True):
            numeric_cols, diag = _select_numeric_columns_polars(window, self.feature_columns)
            if not hasattr(self, '_filter_logged'):
                logger.info(f"[P0-3 FeatureFilter] selected={len(numeric_cols)} "
                           f"dropped={len(diag['dropped_non_numeric'])} "
                           f"bool->float={len(diag['casted_bool_to_float'])}")
                self._filter_logged = True
            features = window.select(numeric_cols).to_numpy().astype(np.float32)
            scaler.partial_fit(features)
            if scaler._count >= max_samples:
                break
        scaler.finalise()
        self._fitted = True

    def export_fitted_scaler(self) -> OnlineRobustScaler:
        """
        Return a cloned copy of the fitted scaler.

        Raises:
            RuntimeError: if the dataset has not been fitted yet.
        """
        if not self._fitted:
            raise RuntimeError("Dataset must be fitted before exporting scaler.")
        return self._scaler.clone()

    def apply_fitted_scaler(self, scaler: OnlineRobustScaler) -> None:
        """
        Attach externally fitted scaler statistics to this dataset.

        Used by validation / test splits to reuse the training statistics.
        """
        if not scaler.is_fitted:
            raise RuntimeError("Supplied scaler is not fitted.")
        self._scaler = scaler
        self._fitted = True

    # ------------------------------------------------------------------ Iteration
    def __iter__(self) -> Iterator[dict]:
        if not self._fitted:
            self.fit()
        worker = get_worker_info()
        if worker is None:
            assigned = self.file_paths
        else:
            assigned = self.file_paths[worker.id :: worker.num_workers]
        return self._stream_samples(assigned)

    # ---------------------------------------------------------------- Helpers
    def _stream_samples(self, files: Iterable[Path]) -> Iterator[dict]:
        for window in self._stream_windows(files, yield_windows=True):
            yield self._window_to_sample(window)

    def _stream_windows(
        self, files: Iterable[Path], *, yield_windows: bool
    ) -> Iterator[pl.DataFrame]:
        buffers: dict[str, pl.DataFrame] = {}
        for file_path in files:
            if _HAS_PYARROW:
                pf = pq.ParquetFile(file_path)
                for rg_idx in range(pf.metadata.num_row_groups):
                    table = pf.read_row_groups(
                        [rg_idx],
                        columns=self._columns_needed,
                        use_threads=False,
                    )
                    chunk = pl.from_arrow(table)
                    yield from self._emit_windows(chunk, buffers)
            else:  # pragma: no cover
                chunk = pl.read_parquet(file_path, columns=self._columns_needed)
                yield from self._emit_windows(chunk, buffers)

    def _emit_windows(
        self, chunk: pl.DataFrame, buffers: dict[str, pl.DataFrame]
    ) -> Generator[pl.DataFrame, None, None]:
        if chunk.is_empty():
            return
        chunk = chunk.sort([self.code_column, self.date_column])
        partitions = chunk.partition_by(self.code_column, maintain_order=True)
        for subdf in partitions:
            code = subdf[self.code_column][0]
            prev = buffers.get(code)
            if prev is not None and not prev.is_empty():
                subdf = pl.concat([prev, subdf], how="vertical_relaxed")

            if subdf.height >= self.sequence_length:
                max_start = subdf.height - self.sequence_length + 1
                for start in range(max_start):
                    yield subdf.slice(start, self.sequence_length)
            # Carry tail forward
            tail_len = min(self.sequence_length - 1, subdf.height)
            buffers[code] = subdf.slice(-tail_len, tail_len) if tail_len > 0 else subdf

    # ---------------------------------------------------------------- Conversion
    def _window_to_sample(self, window: pl.DataFrame) -> dict:
        numeric_cols, _ = _select_numeric_columns_polars(window, self.feature_columns)
        features = window.select(numeric_cols).to_numpy().astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features = self._scaler.transform(features)
        features_tensor = torch.tensor(features, dtype=torch.float32)

        targets: dict[str, torch.Tensor] = {}
        for col in self.target_columns:
            if col not in window.columns:
                continue
            key = self._canonical_target_key(col)
            series = (
                window[col]
                .tail(1)
                .to_numpy()
                .astype(np.float32, copy=False)
                .reshape(-1)
            )
            targets[key] = torch.tensor(series, dtype=torch.float32)

        code = str(window[self.code_column][self.sequence_length - 1])
        date = str(window[self.date_column][self.sequence_length - 1])

        return {
            "features": features_tensor,
            "targets": targets,
            "code": code,
            "date": date,
        }

    @staticmethod
    def _canonical_target_key(col: str) -> str:
        stripped = col.replace("feat_ret_", "").replace("target_", "")
        stripped = stripped.replace("d", "")
        if stripped.isdigit():
            return f"horizon_{stripped}"
        return col
