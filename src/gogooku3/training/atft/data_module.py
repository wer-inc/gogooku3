"""
ATFT Data Module
データローディングとバッチ処理を管理するモジュール
"""

import logging
import multiprocessing as mp
import os
from bisect import bisect_right
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, Sampler

try:  # pragma: no cover - optional acceleration path
    import pyarrow.parquet as pq

    _HAS_PQ = True
except Exception:  # pragma: no cover - keep lightweight fallback path
    pq = None
    _HAS_PQ = False

# Prefer the project DayBatchSampler implementation; fall back to internal placeholder
try:  # pragma: no cover - runtime import guard
    from gogooku3.data.samplers.day_batch_sampler import (
        DayBatchSampler as ExtDayBatchSampler,
    )
    _USE_EXT_SAMPLER = True
except Exception:  # pragma: no cover - fallback for environments without full package
    _USE_EXT_SAMPLER = False

logger = logging.getLogger(__name__)


def _resolve_dl_params(config: DictConfig) -> dict[str, Any]:
    """Derive safe DataLoader parameters from config/env values."""

    def _get(path: str, default: Any) -> Any:
        try:
            value = OmegaConf.select(config, path)
            return default if value is None else value
        except Exception:
            return default

    # Base values (config takes priority, fall back to env, finally default)
    num_workers = _get("train.batch.num_workers", None)
    if num_workers is None:
        num_workers = int(os.getenv("NUM_WORKERS", "0"))
    else:
        num_workers = int(num_workers)

    prefetch_factor = _get("train.batch.prefetch_factor", None)
    if prefetch_factor is None:
        env_pf = os.getenv("PREFETCH_FACTOR", None)
        if env_pf is not None and env_pf.lower() not in ("none", "null", ""):
            prefetch_factor = int(env_pf)

    pin_memory = _get("train.batch.pin_memory", None)
    if pin_memory is None:
        pin_memory = os.getenv("PIN_MEMORY", "0").lower() in ("1", "true", "yes")

    persistent_workers = _get("train.batch.persistent_workers", None)
    if persistent_workers is None:
        persistent_workers = os.getenv("PERSISTENT_WORKERS", "0").lower() in (
            "1",
            "true",
            "yes",
        )

    if num_workers <= 0:
        num_workers = 0
        persistent_workers = False
        prefetch_factor = None
        pin_memory = False if isinstance(pin_memory, bool) else False

    params: dict[str, Any] = {
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(persistent_workers),
    }

    # Global loader-guard (mirrors scripts/train_atft.py):
    # Unless ALLOW_UNSAFE_DATALOADER=1, force single-process mode to avoid
    # sporadic worker aborts seen with multi-process parquet loading.
    allow_multi = os.getenv("ALLOW_UNSAFE_DATALOADER", "0").lower() in ("1", "true", "yes")
    try:
        requested = int(OmegaConf.select(config, "train.batch.num_workers") or 0)
    except Exception:
        requested = 0
    effective = max(int(params.get("num_workers", 0) or 0), requested)
    if not allow_multi and (effective > 0 or bool(params.get("persistent_workers", False))):
        logger.warning(
            "[loader-guard] Enforcing single-process DataLoader (num_workers=0). "
            "Set ALLOW_UNSAFE_DATALOADER=1 to opt in to multi-process."
        )
        params["num_workers"] = 0
        params["persistent_workers"] = False
        params["prefetch_factor"] = None
        params["pin_memory"] = False
        os.environ["NUM_WORKERS"] = "0"
        os.environ["PERSISTENT_WORKERS"] = "0"
        os.environ.setdefault("PIN_MEMORY", "0")
        # Explicitly neutralize prefetch
        os.environ["PREFETCH_FACTOR"] = "0"

    _apply_thread_cap_env(params["num_workers"])
    params["multiprocessing_context"] = _resolve_mp_context(params["num_workers"])

    return params


def _apply_thread_cap_env(num_workers: int) -> None:
    """Clamp thread-heavy libraries so multi-worker loaders stay stable."""

    if num_workers <= 0:
        return

    # Single-thread the parquet readers inside each worker.
    single_thread_targets = {
        "POLARS_MAX_THREADS": 1,
        "RAYON_NUM_THREADS": 1,
        "ARROW_NUM_THREADS": 1,
        "PYARROW_NUM_THREADS": 1,
    }

    for env_key, limit in single_thread_targets.items():
        current = os.getenv(env_key)
        try:
            needs_update = current is None or int(current) <= 0 or int(current) > limit
        except (TypeError, ValueError):
            needs_update = True
        if needs_update:
            os.environ[env_key] = str(limit)

    # Keep BLAS/NumExpr thread counts reasonable to avoid oversubscription.
    cpu_count = os.cpu_count() or 1
    safe_threads = max(1, min(8, cpu_count // max(1, num_workers)))
    blas_targets = {
        "OMP_NUM_THREADS": safe_threads,
        "MKL_NUM_THREADS": safe_threads,
        "OPENBLAS_NUM_THREADS": safe_threads,
        "NUMEXPR_NUM_THREADS": safe_threads,
    }

    for env_key, limit in blas_targets.items():
        current = os.getenv(env_key)
        try:
            needs_update = current is None or int(current) <= 0
            if not needs_update:
                needs_update = int(current) > limit
        except (TypeError, ValueError):
            needs_update = True
        if needs_update:
            os.environ[env_key] = str(limit)

    logger.debug(
        "[loader-guard] Applied thread caps for %d worker(s): polars/arrow->1, BLAS family->%d",
        num_workers,
        safe_threads,
    )


def _resolve_mp_context(num_workers: int) -> mp.context.BaseContext | None:
    """Pick a safe multiprocessing context for DataLoader workers."""

    if num_workers <= 0:
        return None

    preferred = (os.getenv("MP_START_METHOD") or "").strip().lower()
    for candidate in (preferred, "spawn", "forkserver"):
        if not candidate:
            continue
        try:
            return mp.get_context(candidate)
        except (ValueError, RuntimeError):
            continue
    return None


def _build_loader_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    """Compose DataLoader keyword arguments from resolved parameters."""

    kwargs: dict[str, Any] = {
        "num_workers": params.get("num_workers", 0),
        "pin_memory": params.get("pin_memory", False),
    }

    num_workers = kwargs["num_workers"]
    prefetch = params.get("prefetch_factor")
    if num_workers > 0 and prefetch is not None:
        kwargs["prefetch_factor"] = prefetch

    persistent = params.get("persistent_workers", False)
    if num_workers > 0 and persistent:
        kwargs["persistent_workers"] = persistent

    mp_ctx = params.get("multiprocessing_context")
    if mp_ctx is not None:
        kwargs["multiprocessing_context"] = mp_ctx

    return kwargs


class StreamingParquetDataset(Dataset):
    """Streaming dataset for large parquet files."""

    def __init__(
        self,
        file_paths: list[Path],
        feature_columns: list[str],
        target_columns: list[str],
        code_column: str,
        date_column: str,
        sequence_length: int = 60,
        normalize_online: bool = True,
        cache_size: int = 10000,
    ):
        """
        Initialize streaming parquet dataset.

        Args:
            file_paths: List of parquet file paths
            feature_columns: Feature column names
            target_columns: Target column names
            sequence_length: Sequence length for time series
            normalize_online: Apply online normalization
            cache_size: Number of samples to cache in memory
        """
        cleaned_paths: list[Path] = []
        missing_paths: list[str] = []
        for raw_path in file_paths:
            path_obj = Path(raw_path)
            if path_obj.exists():
                cleaned_paths.append(path_obj)
            else:
                missing_paths.append(str(path_obj))

        if missing_paths:
            sample_path = missing_paths[0]
            if len(missing_paths) > 1:
                sample_path = f"{sample_path} (+{len(missing_paths) - 1} more)"
            logger.warning(
                "Skipped %d missing parquet files during dataset init (e.g. %s)",
                len(missing_paths),
                sample_path,
            )

        if not cleaned_paths:
            raise FileNotFoundError("No existing parquet files found for dataset")

        self.file_paths = cleaned_paths
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.sequence_length = sequence_length
        self.normalize_online = normalize_online
        self.cache_size = cache_size
        try:
            self.feature_clip_value = float(os.getenv("FEATURE_CLIP_VALUE", "0"))
        except Exception:
            self.feature_clip_value = 0.0
        self._clip_logged = False
        if self.feature_clip_value <= 0:
            logger.warning(
                "FEATURE_CLIP_VALUE is 0; set a positive bound to enable preprocessing clip and avoid overflow"
            )

        sample_columns = self._detect_columns(cleaned_paths[0])
        self.code_column = self._resolve_column_name(code_column, sample_columns)
        self.date_column = self._resolve_column_name(date_column, sample_columns)
        logger.debug(
            "Resolved schema columns: code=%s, date=%s (available=%s)",
            self.code_column,
            self.date_column,
            sample_columns[:10],
        )
        # 末尾時点（日次）の日付メタデータ（各ウィンドウに1つ）
        # numpy datetime64[D] に統一して軽量に保持する
        self.sequence_dates: np.ndarray | None = None

        # Initialize cache (int -> sample dict)
        self._cache: dict[int, dict[str, Any]] = {}
        self._cache_indices: list[int] = []

        # Pre-compute per-file window offsets for fast index resolution
        self._columns_needed = list(
            dict.fromkeys(
                list(self.feature_columns)
                + list(self.target_columns)
                + [self.code_column, self.date_column]
            )
        )
        self._file_window_counts: list[int] = []
        self._cumulative_windows: list[int] = []
        self._length = 0
        self._build_index()

    @staticmethod
    def _detect_columns(sample_path: Path) -> list[str]:
        try:
            return pl.scan_parquet(sample_path).columns
        except Exception as exc:
            logger.warning("Failed to detect columns from %s: %s", sample_path, exc)
            return []

    @staticmethod
    def _resolve_column_name(original: str, available: list[str]) -> str:
        if original in available:
            return original
        lower_map = {col.lower(): col for col in available}
        resolved = lower_map.get(original.lower())
        if resolved is None:
            logger.warning(
                "Column '%s' not found in parquet (available=%s); downstream access may fail",
                original,
                available[:10],
            )
            return original
        return resolved

    def _build_index(self) -> None:
        """Pre-compute cumulative window counts for each parquet file."""
        total_length = 0
        skipped_files = 0

        self._file_window_counts.clear()
        self._cumulative_windows.clear()
        # 各ファイルに対するウィンドウ末尾日付を溜める（最終的に連結）
        sequence_dates_list: list[np.datetime64] = []

        for file_path in self.file_paths:
            num_rows = self._get_file_num_rows(file_path)
            windows = max(0, num_rows - self.sequence_length + 1)

            if windows <= 0:
                skipped_files += 1

            self._file_window_counts.append(windows)
            total_length += windows
            self._cumulative_windows.append(total_length)

            # ウィンドウ数がある場合のみ日付メタデータを抽出
            if windows > 0:
                try:
                    # 日付列のみをストリーミングで取得
                    df_dates = (
                        pl.scan_parquet(file_path)
                        .select(pl.col(self.date_column))
                        .collect(streaming=True)
                    )
                    if self.date_column not in df_dates.columns:
                        # 想定外: 日付列がない場合はスキップ（サンプラーは後で順序推定にフォールバック）
                        continue

                    # 末尾時点（sequence_length-1 オフセット）の日付を使用
                    # Polars -> Python/NumPy へ変換し、numpy.datetime64[D] へ正規化
                    series = df_dates[self.date_column]
                    # to_list() は dtype に依存せず安全
                    all_dates_list = series.to_list()
                    if len(all_dates_list) >= self.sequence_length:
                        end_dates = all_dates_list[self.sequence_length - 1 :]
                        # numpy.datetime64[D] に変換（文字列/日付/日時のいずれでも受け付ける）
                        end_dates_np = np.array(end_dates, dtype="datetime64[D]")
                        # windows と長さを安全側で一致させる
                        if end_dates_np.shape[0] > windows:
                            end_dates_np = end_dates_np[:windows]
                        sequence_dates_list.extend(end_dates_np.tolist())
                except Exception as exc:
                    # ここでの失敗は致命的ではないのでログのみ
                    logger.debug(
                        "Failed to build sequence_dates for %s due to: %s",
                        file_path,
                        exc,
                    )

        self._length = total_length

        # 収集した sequence_dates を numpy 配列へ確定（長さが一致する場合のみ設定）
        try:
            if sequence_dates_list and len(sequence_dates_list) == self._length:
                self.sequence_dates = np.array(sequence_dates_list, dtype="datetime64[D]")
                logger.info(
                    "Built sequence_dates metadata: %d windows across %d files",
                    len(self.sequence_dates),
                    len(self.file_paths),
                )
            else:
                # 一致しない場合は未設定のまま（サンプラー側でフォールバック）
                if sequence_dates_list:
                    logger.debug(
                        "sequence_dates length mismatch (got %d, expected %d); skipping",
                        len(sequence_dates_list),
                        self._length,
                    )
        except Exception as exc:
            logger.debug("sequence_dates finalization failed: %s", exc)

        if skipped_files:
            logger.warning(
                "Skipped %d parquet files shorter than sequence_length=%d",
                skipped_files,
                self.sequence_length,
            )

    def _get_file_num_rows(self, file_path: Path) -> int:
        """Read the row count from parquet metadata (fast path)."""
        if _HAS_PQ:
            try:
                return int(pq.ParquetFile(file_path).metadata.num_rows)
            except Exception as exc:  # pragma: no cover - fall back if metadata read fails
                logger.debug(
                    "Falling back to Polars row count for %s due to: %s",
                    file_path,
                    exc,
                )

        df = (
            pl.scan_parquet(file_path)
            .select(pl.len().alias("row_count"))
            .collect(streaming=True)
        )
        return int(df[0, 0])

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample."""
        # Check cache
        if idx in self._cache:
            return self._cache[idx]

        # Load sample
        sample = self._load_sample(idx)

        # Update cache
        if len(self._cache) >= self.cache_size:
            # Remove oldest cached item
            oldest_idx = self._cache_indices.pop(0)
            del self._cache[oldest_idx]

        self._cache[idx] = sample
        self._cache_indices.append(idx)

        return sample

    def _canonical_target_key(self, column: str) -> str:
        """Convert dataset target column name to canonical horizon key."""
        if column.startswith("target_") and column.endswith("d"):
            horizon = column[len("target_") : -1]
            if horizon.isdigit():
                return f"horizon_{int(horizon)}d"
        if column.startswith("target_") and column[len("target_") :].isdigit():
            horizon = column[len("target_") :]
            return f"horizon_{int(horizon)}d"
        return column

    @staticmethod
    def _to_python_scalar(value: Any) -> Any:
        """Convert Polars scalar-like objects to native Python types."""
        try:
            if isinstance(value, pl.Series):
                values = value.to_list()
                return values[0] if values else None
            if hasattr(value, "item"):
                return value.item()
        except Exception as exc:
            logger.debug("Failed to convert value %r to scalar: %s", value, exc)
        return value

    def _load_sample(self, idx: int) -> dict[str, Any]:
        """Load a single sample from disk."""
        file_idx, relative_idx = self._resolve_sample_location(idx)
        file_path = self.file_paths[file_idx]

        window = (
            pl.scan_parquet(file_path)
            .slice(relative_idx, self.sequence_length)
            .select(self._columns_needed)
            .collect(streaming=True)
        )

        if window.height < self.sequence_length:
            raise IndexError(
                f"Index {idx} produced truncated window (file: {file_path}, rows: {window.height})"
            )

        features = window.select(self.feature_columns).to_numpy()
        features = features.astype(np.float32, copy=False)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        if self.feature_clip_value > 0:
            np.clip(features, -self.feature_clip_value, self.feature_clip_value, out=features)
            if not self._clip_logged:
                logger.info(
                    "[feature-clip] Applied feature clipping at ±%.2f (set FEATURE_CLIP_VALUE to adjust)",
                    self.feature_clip_value,
                )
                self._clip_logged = True

        # Apply online normalization if needed
        if self.normalize_online:
            features = self._normalize(features)

        targets_window = window.select(self.target_columns).to_numpy()
        targets_row = targets_window[-1]
        targets_row = np.nan_to_num(targets_row, nan=0.0, posinf=0.0, neginf=0.0)

        code_series = window[self.code_column] if self.code_column in window.columns else None
        date_series = window[self.date_column] if self.date_column in window.columns else None

        code_value: Any | None = code_series[-1] if code_series is not None else None
        date_value: Any | None = date_series[-1] if date_series is not None else None

        feature_tensor = torch.tensor(features, dtype=torch.float32)

        target_dict: dict[str, torch.Tensor] = {}
        for col_name, value in zip(self.target_columns, targets_row, strict=False):
            canon = self._canonical_target_key(col_name)
            scalar = float(np.asarray(value).reshape(-1)[0])
            target_dict[canon] = torch.tensor([scalar], dtype=torch.float32)

        sample = {
            "features": feature_tensor,
            "targets": target_dict,
            "code": self._to_python_scalar(code_value),
            "date": self._to_python_scalar(date_value),
        }

        if sample["code"] is not None:
            sample["code"] = str(sample["code"])
        if sample["date"] is not None:
            sample["date"] = str(sample["date"])

        return sample

    def _resolve_sample_location(self, idx: int) -> tuple[int, int]:
        """Map a global sample index to (file_index, offset within file)."""
        if self._length == 0:
            raise IndexError("Dataset is empty")

        if idx < 0:
            idx += self._length

        if idx < 0 or idx >= self._length:
            raise IndexError(f"Index {idx} out of range for dataset with length {self._length}")

        file_idx = bisect_right(self._cumulative_windows, idx)
        prev_cumulative = 0 if file_idx == 0 else self._cumulative_windows[file_idx - 1]
        relative_idx = idx - prev_cumulative

        # Guard against empty files that contribute zero windows
        while file_idx < len(self._file_window_counts) and self._file_window_counts[file_idx] == 0:
            file_idx += 1
            prev_cumulative = self._cumulative_windows[file_idx - 1]
            relative_idx = idx - prev_cumulative

        if file_idx >= len(self.file_paths):
            raise IndexError(f"Index {idx} could not be resolved to a file")

        return file_idx, relative_idx

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """Apply online normalization."""
        # Work in float64 to avoid overflow on large magnitudes, then return float32 for torch
        work_array = features.astype(np.float64, copy=False)
        mean = work_array.mean(axis=0, keepdims=True)
        std = work_array.std(axis=0, keepdims=True)
        std = np.clip(std, 1e-8, None)
        normalized = (work_array - mean) / std
        return normalized.astype(np.float32, copy=False)


class _InternalDayBatchSampler(Sampler):
    """Sampler that groups samples by day for batch processing."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        date_indices: dict[str, list[int]] | None = None,
        shuffle: bool = True,
    ):
        """
        Initialize day batch sampler.

        Args:
            dataset: Dataset to sample from
            batch_size: Batch size
            date_indices: Mapping from dates to sample indices
            shuffle: Whether to shuffle dates
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.date_indices = date_indices or self._build_date_indices()
        self.shuffle = shuffle

    def _build_date_indices(self) -> dict[str, list[int]]:
        """Build mapping from dates to indices (placeholder)."""
        # This would need actual implementation based on dataset structure
        logger.warning("Using placeholder date indices - implement actual date grouping")
        return {"default": list(range(len(self.dataset)))}

    def __iter__(self):
        """Iterate over batches grouped by day."""
        dates = list(self.date_indices.keys())

        if self.shuffle:
            np.random.shuffle(dates)

        for date in dates:
            indices = self.date_indices[date]

            if self.shuffle:
                np.random.shuffle(indices)

            # Yield batches for this date
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                if batch_indices:  # Skip empty batches
                    yield batch_indices

    def __len__(self) -> int:
        """Return number of batches."""
        total_batches = 0
        for indices in self.date_indices.values():
            total_batches += (len(indices) + self.batch_size - 1) // self.batch_size
        return total_batches


# Bind the sampler symbol used below
if _USE_EXT_SAMPLER:
    DayBatchSampler = ExtDayBatchSampler  # type: ignore
else:  # pragma: no cover - keep internal placeholder
    DayBatchSampler = _InternalDayBatchSampler  # type: ignore


class ProductionDataModuleV2:
    """Production data module for ATFT training."""

    def __init__(self, config: DictConfig):
        """
        Initialize data module.

        Args:
            config: Data configuration
        """
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self) -> None:
        """Set up datasets."""
        data_dir = Path(self.config.data.source.data_dir)

        # Find train/val/test files
        train_files = sorted(data_dir.glob("train/*.parquet"))
        val_files = sorted(data_dir.glob("val/*.parquet"))
        test_files = sorted(data_dir.glob("test/*.parquet"))

        if not train_files:
            raise FileNotFoundError(f"No training files found in {data_dir}/train/")

        logger.info(f"📂 Found {len(train_files)} train, {len(val_files)} val, {len(test_files)} test files")

        # Get feature and target columns
        feature_columns = self._get_feature_columns()
        target_columns = self._get_target_columns()

        # Create datasets
        self.train_dataset = StreamingParquetDataset(
            file_paths=train_files,
            feature_columns=feature_columns,
            target_columns=target_columns,
            code_column=self.config.data.schema.code_column,
            date_column=self.config.data.schema.date_column,
            sequence_length=self.config.data.time_series.sequence_length,
            normalize_online=self.config.normalization.online_normalization.enabled,
        )

        if val_files:
            self.val_dataset = StreamingParquetDataset(
                file_paths=val_files,
                feature_columns=feature_columns,
                target_columns=target_columns,
                code_column=self.config.data.schema.code_column,
                date_column=self.config.data.schema.date_column,
                sequence_length=self.config.data.time_series.sequence_length,
                normalize_online=self.config.normalization.online_normalization.enabled,
            )

        if test_files:
            self.test_dataset = StreamingParquetDataset(
                file_paths=test_files,
                feature_columns=feature_columns,
                target_columns=target_columns,
                code_column=self.config.data.schema.code_column,
                date_column=self.config.data.schema.date_column,
                sequence_length=self.config.data.time_series.sequence_length,
                normalize_online=self.config.normalization.online_normalization.enabled,
            )

        logger.info(f"✅ Datasets created: train={len(self.train_dataset)} samples")

    def _get_feature_columns(self) -> list[str]:
        """Get feature column names."""
        if self.config.data.schema.feature_columns:
            return self.config.data.schema.feature_columns

        # Auto-detect from first file
        data_dir = Path(self.config.data.source.data_dir)
        first_file = next(data_dir.glob("**/*.parquet"), None)

        if not first_file:
            raise FileNotFoundError("No parquet files found for column detection")

        df = pl.scan_parquet(first_file).head(1).collect()
        exclude_cols = [
            self.config.data.schema.date_column,
            self.config.data.schema.code_column,
            self.config.data.schema.target_column,
        ]

        numeric_dtypes = {
            pl.Float64,
            pl.Float32,
            pl.Int64,
            pl.Int32,
            pl.Int16,
            pl.Int8,
            pl.UInt64,
            pl.UInt32,
            pl.UInt16,
            pl.UInt8,
        }

        feature_cols = [
            col
            for col in df.columns
            if col not in exclude_cols and df.schema[col] in numeric_dtypes
        ]
        logger.info(f"✅ Auto-detected {len(feature_cols)} feature columns")

        return feature_cols

    def _get_target_columns(self) -> list[str]:
        """Get target column names."""
        horizons = self.config.data.time_series.prediction_horizons
        base_target = self.config.data.schema.target_column

        # Generate target column names for each horizon
        target_cols = []
        for h in horizons:
            target_cols.append(f"{base_target}_{h}d")

        return target_cols

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        if self.train_dataset is None:
            self.setup()

        dl_params = _resolve_dl_params(self.config)

        # Use day batch sampler if enabled
        if self.config.data.get("use_day_batch_sampler", False):
            sampler = DayBatchSampler(
                dataset=self.train_dataset,
                batch_size=self.config.train.batch.train_batch_size,
                shuffle=True,
            )
            return DataLoader(
                self.train_dataset,
                batch_sampler=sampler,
                **_build_loader_kwargs(dl_params),
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.config.train.batch.train_batch_size,
                shuffle=True,
                **_build_loader_kwargs(dl_params),
            )

    def val_dataloader(self) -> DataLoader | None:
        """Get validation dataloader."""
        if self.val_dataset is None:
            return None

        dl_params = _resolve_dl_params(self.config)

        return DataLoader(
            self.val_dataset,
            batch_size=self.config.train.batch.val_batch_size,
            shuffle=False,
            **_build_loader_kwargs(dl_params),
        )

    def test_dataloader(self) -> DataLoader | None:
        """Get test dataloader."""
        if self.test_dataset is None:
            return None

        dl_params = _resolve_dl_params(self.config)

        return DataLoader(
            self.test_dataset,
            batch_size=self.config.train.batch.val_batch_size,
            shuffle=False,
            **_build_loader_kwargs(dl_params),
        )

    def get_data_info(self) -> dict[str, Any]:
        """Get information about the data."""
        info = {
            "train_size": len(self.train_dataset) if self.train_dataset else 0,
            "val_size": len(self.val_dataset) if self.val_dataset else 0,
            "test_size": len(self.test_dataset) if self.test_dataset else 0,
            "num_features": len(self._get_feature_columns()),
            "num_targets": len(self._get_target_columns()),
            "sequence_length": self.config.data.time_series.sequence_length,
            "batch_size": self.config.train.batch.train_batch_size,
        }
        return info


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Custom collate function for batching."""
    features, targets = zip(*batch, strict=False)
    features = torch.stack(features)
    targets = torch.stack(targets)
    return features, targets
