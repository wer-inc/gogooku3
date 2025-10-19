"""
ATFT Data Module
データローディングとバッチ処理を管理するモジュール
"""

import logging
import multiprocessing as mp
import os
import pickle
import time
from bisect import bisect_right
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, Sampler
import pandas as pd

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

    # CRITICAL: Safe mode (FORCE_SINGLE_PROCESS=1) ALWAYS overrides multi-worker settings
    # This must be checked BEFORE any other guards to ensure proper single-process operation
    is_safe_mode = os.getenv("FORCE_SINGLE_PROCESS", "0").lower() in ("1", "true", "yes")
    if is_safe_mode:
        logger.info(
            "[SAFE MODE] Enforcing single-process DataLoader (num_workers=0) due to FORCE_SINGLE_PROCESS=1"
        )
        params["num_workers"] = 0
        params["persistent_workers"] = False
        params["prefetch_factor"] = None
        params["pin_memory"] = False
        os.environ["NUM_WORKERS"] = "0"
        os.environ["PERSISTENT_WORKERS"] = "0"
        os.environ["PIN_MEMORY"] = "0"
        os.environ["PREFETCH_FACTOR"] = "0"

        # CRITICAL FIX (2025-10-14): Limit PyTorch internal thread pool to prevent deadlock
        # Root cause: PyTorch uses 128 threads by default, causing contention with Parquet I/O
        # This leads to deadlock when iterating DataLoader in training loop
        try:
            import torch
            torch.set_num_threads(1)
            logger.info("[SAFE MODE] Limited PyTorch threads to 1 (prevents 128-thread deadlock)")
        except Exception as e:
            logger.warning(f"[SAFE MODE] Failed to limit PyTorch threads: {e}")

        # Limit all parallel computation libraries
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        os.environ["POLARS_MAX_THREADS"] = "1"

        # Skip remaining guards - Safe mode takes absolute priority
        _apply_thread_cap_env(0)
        params["multiprocessing_context"] = None
        return params

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
        reader_engine: str | None = None,
        exposure_columns: list[str] | None = None,
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
            exposure_columns: Columns to use for exposure features (e.g. market_cap, beta, sector_code)
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
        # Engine selection: default to pyarrow for multi-worker safety if unspecified
        self.reader_engine = (reader_engine or os.getenv("PARQUET_READER_ENGINE") or "").strip().lower()
        if not self.reader_engine:
            # Heuristic: prefer pyarrow in multi-worker scenarios for stability
            try:
                _nw = int(os.getenv("NUM_WORKERS", "0"))
            except Exception:
                _nw = 0
            self.reader_engine = "pyarrow" if _nw > 0 else "polars"
        if self.reader_engine not in ("polars", "pyarrow"):
            logger.warning("Unknown reader_engine=%s; falling back to 'polars'", self.reader_engine)
            self.reader_engine = "polars"
        try:
            self.feature_clip_value = float(os.getenv("FEATURE_CLIP_VALUE", "0"))
        except Exception:
            self.feature_clip_value = 0.0
        self._clip_logged = False
        if self.feature_clip_value <= 0:
            logger.warning(
                "FEATURE_CLIP_VALUE is 0; set a positive bound to enable preprocessing clip and avoid overflow"
            )

        # CRITICAL FIX (2025-10-03): Global normalization statistics
        # Compute median/MAD from training data for proper cross-sample normalization
        self._global_median: np.ndarray | None = None
        self._global_mad: np.ndarray | None = None
        self._stats_computed = False

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

        # Lazy cache for pyarrow row-group metadata per file (file_idx -> (offsets, lengths))
        self._rg_meta: dict[int, tuple[list[int], list[int]]] = {}

        # Phase 2: 露出中立・回転率・整合性ロスのための拡張フィールド
        self.use_exposure_features = os.getenv("USE_EXPOSURE_FEATURES", "0") == "1"
        if self.use_exposure_features:
            # 露出特徴量の列設定
            if exposure_columns is None:
                exposure_cols_env = os.getenv("EXPOSURE_COLUMNS", "market_cap,beta,sector_code")
                exposure_columns = exposure_cols_env.split(",")

            # Filter to only columns that actually exist in the data
            self.exposure_columns = []
            for col in exposure_columns:
                if col in sample_columns:
                    self.exposure_columns.append(col)
                    logger.debug(f"[Phase2] Found exposure column: {col}")
                else:
                    logger.debug(f"[Phase2] Exposure column not found in data: {col}")

            if self.exposure_columns:
                logger.info(f"[Phase2] Exposure features enabled with available columns: {self.exposure_columns}")
            else:
                logger.warning("[Phase2] No exposure columns found in data, using placeholders")
                # Keep the requested columns for placeholder generation
                self.exposure_columns = exposure_columns

            # マッピング辞書の初期化
            self.date_to_group_id: dict[str, int] = {}
            self.code_to_sid: dict[str, int] = {}
            self.sector_to_id: dict[str, int] = {}
            self._next_group_id = 0
            self._next_sid = 0
            self._next_sector_id = 0
        else:
            self.exposure_columns = []
            logger.debug("[Phase2] Exposure features disabled (USE_EXPOSURE_FEATURES=0)")

        # Pre-compute per-file window offsets for fast index resolution
        # Only add exposure columns that actually exist in the data
        exposure_cols_to_add = []
        if self.use_exposure_features and self.exposure_columns:
            for col in self.exposure_columns:
                if col in sample_columns:
                    exposure_cols_to_add.append(col)

        self._columns_needed = list(
            dict.fromkeys(
                list(self.feature_columns)
                + list(self.target_columns)
                + [self.code_column, self.date_column]
                + exposure_cols_to_add
            )
        )
        self._file_window_counts: list[int] = []
        self._cumulative_windows: list[int] = []
        self._length = 0
        self._build_index()

    @staticmethod
    def _detect_columns(sample_path: Path) -> list[str]:
        try:
            # Use collect_schema().names() instead of .columns to avoid schema resolution warnings
            return pl.scan_parquet(sample_path).collect_schema().names()
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

    def _get_cache_path(self) -> Path:
        """Get the path to the metadata cache file."""
        if not self.file_paths:
            return Path(".")  / ".metadata_cache.pkl"
        # Place cache in the split directory (train/val/test specific)
        # e.g., output/atft_data/train/xxx.parquet -> output/atft_data/train/.metadata_cache.pkl
        return self.file_paths[0].parent / ".metadata_cache.pkl"

    def _validate_cache(self, cached: dict) -> bool:
        """Validate that cached metadata is still valid."""
        try:
            # Check if file count matches
            if cached.get('file_count') != len(self.file_paths):
                logger.debug("Cache invalid: file count mismatch")
                return False

            # Check if cache is not too old (7 days)
            cache_age_days = (time.time() - cached.get('created_at', 0)) / 86400
            if cache_age_days > 7:
                logger.debug("Cache invalid: too old (%.1f days)", cache_age_days)
                return False

            return True
        except Exception as exc:
            logger.debug("Cache validation failed: %s", exc)
            return False

    def _load_metadata_cache(self) -> bool:
        """Load metadata from cache if available and valid."""
        cache_path = self._get_cache_path()

        if not cache_path.exists():
            logger.debug("No metadata cache found at %s", cache_path)
            return False

        try:
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)

            if not self._validate_cache(cached):
                return False

            # Load cached metadata
            self._file_window_counts = cached['window_counts']
            self._cumulative_windows = cached['cumulative_windows']
            self.sequence_dates = cached.get('sequence_dates')
            self._length = cached['total_length']

            logger.info("✅ Loaded metadata from cache (%d windows)", self._length)
            return True

        except Exception as exc:
            logger.warning("Failed to load metadata cache: %s", exc)
            return False

    def _save_metadata_cache(self) -> None:
        """Save metadata to cache."""
        cache_path = self._get_cache_path()

        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            cache_data = {
                'window_counts': self._file_window_counts,
                'cumulative_windows': self._cumulative_windows,
                'sequence_dates': self.sequence_dates,
                'total_length': self._length,
                'file_count': len(self.file_paths),
                'created_at': time.time()
            }

            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)

            logger.debug("Saved metadata cache to %s", cache_path)

        except Exception as exc:
            logger.warning("Failed to save metadata cache: %s", exc)

    def _build_index(self) -> None:
        """Pre-compute cumulative window counts for each parquet file."""
        # Try to load from cache first
        start_time = time.time()
        if self._load_metadata_cache():
            logger.info("Metadata loading took %.2fs (from cache)", time.time() - start_time)
            return

        # Cache miss or invalid - build from scratch
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

        # Save metadata to cache for next run
        self._save_metadata_cache()
        logger.info("Metadata building took %.2fs (from scratch)", time.time() - start_time)

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

    def _date_to_group_id_fn(self, date_value: Any) -> int:
        """Convert date to group ID for daily batching."""
        if date_value is None:
            return 0
        date_str = str(date_value)[:10]  # YYYY-MM-DD format
        if date_str not in self.date_to_group_id:
            self.date_to_group_id[date_str] = self._next_group_id
            self._next_group_id += 1
        return self.date_to_group_id[date_str]

    def _code_to_sid_fn(self, code_value: Any) -> int:
        """Convert stock code to stock ID."""
        if code_value is None:
            return 0
        code_str = str(code_value)
        if code_str not in self.code_to_sid:
            self.code_to_sid[code_str] = self._next_sid
            self._next_sid += 1
        return self.code_to_sid[code_str]

    def _sector_to_onehot(self, sector_value: Any) -> list[float]:
        """Convert sector code to one-hot vector."""
        if sector_value is None:
            # Return zero vector for unknown sectors
            return [0.0] * max(1, len(self.sector_to_id))

        sector_str = str(sector_value)
        if sector_str not in self.sector_to_id:
            self.sector_to_id[sector_str] = self._next_sector_id
            self._next_sector_id += 1

        # Create one-hot vector
        onehot = [0.0] * max(1, self._next_sector_id)
        onehot[self.sector_to_id[sector_str]] = 1.0
        return onehot

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

        # Select reader path
        if self.reader_engine == "pyarrow" and _HAS_PQ:
            window = self._load_window_pyarrow(file_idx, file_path, relative_idx)
        else:
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

        # Phase 2: 拡張フィールドの追加
        if self.use_exposure_features:
            # 1. group_day: 日付グループID (torch.long)
            sample["group_day"] = torch.tensor(
                self._date_to_group_id_fn(sample["date"]), dtype=torch.long
            )

            # 2. sid: 銘柄ID (torch.long)
            sample["sid"] = torch.tensor(
                self._code_to_sid_fn(sample["code"]), dtype=torch.long
            )

            # 3. exposures: 露出特徴量 (torch.float)
            exposures = []

            # market_cap (対数変換) - only if column exists
            if "market_cap" in self.exposure_columns:
                if "market_cap" in window.columns:
                    mkt_cap = window["market_cap"][-1]
                    mkt_cap_val = self._to_python_scalar(mkt_cap)
                    if mkt_cap_val is not None and mkt_cap_val > 0:
                        exposures.append(np.log(float(mkt_cap_val)))
                    else:
                        exposures.append(0.0)
                else:
                    # Column requested but not available - use placeholder
                    exposures.append(0.0)

            # beta - only if column exists
            if "beta" in self.exposure_columns:
                if "beta" in window.columns:
                    beta_val = window["beta"][-1]
                    beta_scalar = self._to_python_scalar(beta_val)
                    exposures.append(float(beta_scalar) if beta_scalar is not None else 0.0)
                else:
                    # Column requested but not available - use placeholder
                    exposures.append(0.0)

            # sector_code (One-Hot) - only if column exists
            if "sector_code" in self.exposure_columns:
                if "sector_code" in window.columns:
                    sector_val = window["sector_code"][-1]
                    sector_scalar = self._to_python_scalar(sector_val)
                    sector_onehot = self._sector_to_onehot(sector_scalar)
                    exposures.extend(sector_onehot)
                else:
                    # Column requested but not available - use placeholder one-hot
                    exposures.extend([0.0])

            # Convert to tensor
            if exposures:
                sample["exposures"] = torch.tensor(exposures, dtype=torch.float32)
            else:
                # Empty exposures if no features found
                sample["exposures"] = torch.zeros(1, dtype=torch.float32)

        return sample

    # -----------------------
    # PyArrow reader backend
    # -----------------------
    def _ensure_rg_meta(self, file_idx: int, file_path: Path) -> tuple[list[int], list[int]]:
        """Build (offsets, lengths) for row-groups of a parquet file lazily."""
        if file_idx in self._rg_meta:
            return self._rg_meta[file_idx]
        if not _HAS_PQ:
            raise RuntimeError("PyArrow is not available but reader_engine='pyarrow' was selected")
        pf = pq.ParquetFile(file_path)
        md = pf.metadata
        rg_lens: list[int] = [md.row_group(i).num_rows for i in range(md.num_row_groups)]
        offsets: list[int] = [0]
        for n in rg_lens[:-1]:
            offsets.append(offsets[-1] + n)
        self._rg_meta[file_idx] = (offsets, rg_lens)
        return self._rg_meta[file_idx]

    def _load_window_pyarrow(self, file_idx: int, file_path: Path, relative_idx: int) -> pl.DataFrame:
        """Read a [relative_idx : relative_idx+sequence_length) window via PyArrow.

        Safety choices:
        - use_threads=False (avoid internal parallelism in worker)
        - memory_map=False (avoid OS mmaps across many processes)
        """
        offsets, lengths = self._ensure_rg_meta(file_idx, file_path)

        # Locate covering row-groups
        start = relative_idx
        end = relative_idx + self.sequence_length  # exclusive

        # Find start RG
        rg_start = bisect_right(offsets, start) - 1
        if rg_start < 0:
            rg_start = 0
        # Find end RG (inclusive)
        # last start offset <= end-1
        rg_end = bisect_right(offsets, end - 1) - 1
        if rg_end < rg_start:
            rg_end = rg_start

        row_groups = list(range(rg_start, rg_end + 1))

        # Read minimal set of row-groups
        pf = pq.ParquetFile(file_path)
        table = pf.read_row_groups(
            row_groups,
            columns=self._columns_needed,
            use_threads=False,
        )

        # Slice inside the concatenated row-groups to the exact window
        start_in_table = start - offsets[rg_start]
        # If multiple RGs, the concatenated table starts at offsets[rg_start]
        # Adjust for cumulative lengths
        if rg_start != 0:
            # We need cumulative length of preceding row-groups among selected ones
            start_in_table = start - offsets[rg_start]
        end_in_table = start_in_table + self.sequence_length

        # Convert to pandas (cheap for small slices), then slice
        pdf: pd.DataFrame = table.to_pandas(types_mapper=pd.ArrowDtype)
        window_pdf = pdf.iloc[start_in_table:end_in_table]
        # Convert to Polars for downstream uniformity
        window_pl = pl.from_pandas(window_pdf.reset_index(drop=True))
        return window_pl

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

    def _canonical_target_key(self, col_name: str) -> str:
        """
        Convert target column names to canonical horizon_N format.

        Converts:
            target_1d -> horizon_1
            target_5d -> horizon_5
            feat_ret_1d -> horizon_1
            feat_ret_20d -> horizon_20
        """
        import re

        # Match target_Nd, feat_ret_Nd patterns
        match = re.search(r'(?:target|feat_ret)_(\d+)d?', col_name)
        if match:
            horizon_num = match.group(1)
            return f'horizon_{horizon_num}'

        # Return original if no pattern matches
        return col_name

    def _compute_global_statistics(self, max_samples: int = 1000) -> None:
        """
        Compute global median/MAD from a subset of training data.

        CRITICAL FIX (2025-10-03): Use global statistics instead of per-sample stats
        to preserve cross-sample information.

        Args:
            max_samples: Maximum number of samples to use for statistics estimation
        """
        if self._stats_computed:
            return

        logger.info("Computing global normalization statistics from training data...")

        # Sample features from multiple files
        all_features = []
        samples_per_file = max(1, max_samples // len(self.file_paths))

        for file_path in self.file_paths[: min(50, len(self.file_paths))]:  # Expanded from 5 to 50 for better statistics
            try:
                df = pl.scan_parquet(file_path).select(self.feature_columns).head(samples_per_file).collect(streaming=True)
                features = df.to_numpy().astype(np.float64)
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                # Clip before computing statistics
                np.clip(features, -1e8, 1e8, out=features)
                all_features.append(features)

                if sum(len(f) for f in all_features) >= max_samples:
                    break
            except Exception as exc:
                logger.warning("Failed to load features from %s: %s", file_path, exc)
                continue

        if not all_features:
            logger.warning("Failed to compute global statistics, falling back to per-sample normalization")
            self._stats_computed = True
            return

        # Concatenate all features
        combined = np.vstack(all_features)

        # Compute robust statistics (per feature, across all samples)
        self._global_median = np.median(combined, axis=0, keepdims=False)
        mad = np.median(np.abs(combined - self._global_median), axis=0, keepdims=False)
        self._global_mad = mad * 1.4826  # Convert MAD to std
        self._global_mad = np.clip(self._global_mad, 1e-8, None)

        self._stats_computed = True

        logger.info(
            "Global statistics computed from %d samples: median range [%.2e, %.2e], MAD range [%.2e, %.2e]",
            len(combined),
            self._global_median.min(),
            self._global_median.max(),
            self._global_mad.min(),
            self._global_mad.max(),
        )

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """
        Apply robust normalization to features using global statistics.

        CRITICAL FIX (2025-10-03): Previous implementation used per-sample normalization,
        which loses cross-sample information and causes prediction collapse.

        New implementation:
        1. Compute global median/MAD once from training data
        2. Use global statistics to normalize all samples consistently
        3. This preserves cross-sample relationships
        """
        # Compute global statistics on first call (lazy initialization)
        if not self._stats_computed:
            self._compute_global_statistics()

        # Work in float64 to avoid overflow
        work_array = features.astype(np.float64, copy=False)

        # Clip extreme values
        np.clip(work_array, -1e8, 1e8, out=work_array)

        # Apply global normalization if statistics available
        if self._global_median is not None and self._global_mad is not None:
            # Broadcast global stats to (seq_len, n_features) shape
            normalized = (work_array - self._global_median) / self._global_mad
        else:
            # Fallback to per-sample normalization (not ideal but better than nothing)
            median = np.median(work_array, axis=0, keepdims=True)
            mad = np.median(np.abs(work_array - median), axis=0, keepdims=True)
            mad = mad * 1.4826
            mad = np.clip(mad, 1e-8, None)
            normalized = (work_array - median) / mad
            logger.warning("Using fallback per-sample normalization (global stats not available)")

        # Final clipping to ±5 sigma
        np.clip(normalized, -5.0, 5.0, out=normalized)

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
        """Build mapping from dates to indices using dataset metadata when available.

        Prefer `dataset.sequence_dates` (numpy datetime64[D] for each sample window).
        Falls back to a single default bucket if unavailable or mismatched.
        """
        try:
            seq_dates = getattr(self.dataset, "sequence_dates", None)
            total_len = len(self.dataset)
            if seq_dates is None or len(seq_dates) != total_len:
                logger.warning(
                    "DayBatchSampler: sequence_dates unavailable or length mismatch (%s vs %s); using single bucket",
                    0 if seq_dates is None else len(seq_dates),
                    total_len,
                )
                return {"default": list(range(total_len))}

            # Build buckets: ISO string keys to keep dict small and deterministic
            buckets: dict[str, list[int]] = {}
            # Convert once to ISO strings for hashing; numpy datetime64 -> str
            # This is efficient as it avoids per-batch parquet scans
            for idx, d in enumerate(seq_dates):
                # Safe conversion to YYYY-MM-DD
                try:
                    key = str(d)[:10]
                except Exception:
                    key = str(d)
                if key not in buckets:
                    buckets[key] = []
                buckets[key].append(idx)

            # Optionally drop degenerate very small buckets by merging
            # Keep as-is for transparency
            logger.info("DayBatchSampler: built %d day buckets", len(buckets))
            return buckets
        except Exception as exc:
            logger.warning("DayBatchSampler: fallback to single bucket due to: %s", exc)
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

        # Phase 2: 露出特徴量の設定
        exposure_columns = None
        if os.getenv("USE_EXPOSURE_FEATURES", "0") == "1":
            exposure_cols_env = os.getenv("EXPOSURE_COLUMNS", "market_cap,beta,sector_code")
            exposure_columns = exposure_cols_env.split(",")

        # CRITICAL FIX (2025-10-04): Resolve cache_size based on num_workers
        # Root cause: cache_size=10000 × num_workers × prefetch_factor = memory explosion
        # Solution: Small cache (256) for multi-worker, large cache (10000) for single-worker
        cache_size = self._resolve_cache_size()

        # Decide parquet reader engine (stability first)
        reader_engine = os.getenv("PARQUET_READER_ENGINE", "").strip().lower()
        try:
            _nw_req = int(self.config.train.batch.get("num_workers", 0))
        except Exception:
            _nw_req = 0
        if not reader_engine:
            reader_engine = "pyarrow" if _nw_req > 0 else "polars"
        logger.info("Parquet reader engine: %s", reader_engine)

        # Create datasets
        self.train_dataset = StreamingParquetDataset(
            file_paths=train_files,
            feature_columns=feature_columns,
            target_columns=target_columns,
            code_column=self.config.data.schema.code_column,
            date_column=self.config.data.schema.date_column,
            sequence_length=self.config.data.time_series.sequence_length,
            normalize_online=self.config.normalization.online_normalization.enabled,
            exposure_columns=exposure_columns,
            cache_size=cache_size,
            reader_engine=reader_engine,
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
                exposure_columns=exposure_columns,
                cache_size=cache_size,
                reader_engine=reader_engine,
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
                exposure_columns=exposure_columns,
                cache_size=cache_size,
                reader_engine=reader_engine,
            )

        logger.info(f"✅ Datasets created: train={len(self.train_dataset)} samples")

        # CRITICAL FIX (2025-10-04): Pre-compute normalization stats in main process
        # Root cause: Each DataLoader worker independently computes statistics (lazy init)
        # This causes parallel file I/O from 8 workers → resource contention → crash
        # Solution: Compute once in main process, share with all datasets
        logger.info("Pre-computing global normalization statistics in main process...")
        self.train_dataset._compute_global_statistics()

        # Share train statistics with val/test datasets (train-only stats for data safety)
        if hasattr(self, 'val_dataset') and self.val_dataset:
            self.val_dataset._global_median = self.train_dataset._global_median
            self.val_dataset._global_mad = self.train_dataset._global_mad
            self.val_dataset._stats_computed = True

        if hasattr(self, 'test_dataset') and self.test_dataset:
            self.test_dataset._global_median = self.train_dataset._global_median
            self.test_dataset._global_mad = self.train_dataset._global_mad
            self.test_dataset._stats_computed = True

        logger.info("✅ Global statistics computed and shared across datasets")

    def _resolve_cache_size(self) -> int:
        """
        Resolve cache_size based on num_workers configuration.

        Memory calculation:
        - Single sample: ~73KB (60 × 306 features × 4 bytes)
        - Multi-worker (8 workers × 10000 cache): ~5.8GB → Memory explosion → Crash
        - Multi-worker (4 workers × 256 cache): ~72MB → Safe
        - Single-worker (1 process × 10000 cache): ~0.7GB → Safe

        Returns:
            int: Resolved cache size
        """
        # Get num_workers from config
        batch_cfg = self.config.train.batch
        num_workers = batch_cfg.get('num_workers', 0)

        # Get dataset config if available
        dataset_cfg = batch_cfg.get('dataset', {})

        if num_workers == 0:
            # Single worker: Large cache is safe in main process
            cache_size = dataset_cfg.get('cache_size_single_worker', 10000)
            logger.info(f"🔧 Single-worker mode: cache_size={cache_size}")
        else:
            # Multi-worker: Small cache to avoid memory explosion
            # Memory: num_workers × cache_size × 73KB/sample
            cache_size = dataset_cfg.get('cache_size', 256)
            total_memory_mb = num_workers * cache_size * 73 / 1024
            logger.info(
                f"🔧 Multi-worker mode (workers={num_workers}): "
                f"cache_size={cache_size} (est. {total_memory_mb:.1f}MB total)"
            )

        return cache_size

    def _get_feature_columns(self) -> list[str]:
        """Get feature column names."""
        if self.config.data.schema.feature_columns:
            cols = list(self.config.data.schema.feature_columns)
            selected_path = os.getenv("SELECTED_FEATURES_JSON", "").strip()
            if selected_path:
                try:
                    import json
                    from pathlib import Path as _P
                    data = json.loads(_P(selected_path).read_text())
                    selected = set(data.get("selected_features", []))
                    if selected:
                        cols = [c for c in cols if c in selected]
                        logger.info("[feature-selection] Applied selected features (%d)", len(cols))
                except Exception as _e:
                    logger.warning("[feature-selection] failed to apply %s: %s", selected_path, _e)
            return cols

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

        # Exclude target columns (target_1d, target_5d, etc.) as well
        feature_cols = [
            col for col in df.columns
            if col not in exclude_cols
            and not col.startswith('target_')  # Exclude all target columns
            and df.schema[col] in numeric_dtypes
        ]
        # Optional: intersect with externally selected features (JSON list)
        selected_path = os.getenv("SELECTED_FEATURES_JSON", "").strip()
        if selected_path:
            try:
                import json
                data = json.loads(Path(selected_path).read_text())
                selected = set(data.get("selected_features", []))
                if selected:
                    feature_cols = [c for c in feature_cols if c in selected]
                    logger.info("[feature-selection] Applied selected features (%d)", len(feature_cols))
            except Exception as _e:
                logger.warning("[feature-selection] failed to apply %s: %s", selected_path, _e)
        logger.info(f"✅ Auto-detected {len(feature_cols)} feature columns")

        return feature_cols

    def _get_target_columns(self) -> list[str]:
        """Get target column names."""
        # First, check if explicit target_columns are provided in config
        if hasattr(self.config.data.schema, 'target_columns') and self.config.data.schema.target_columns:
            target_cols = list(self.config.data.schema.target_columns)
            logger.info(f"✅ Using explicit target columns from config: {target_cols}")
            return target_cols

        # Fallback: Generate from prediction_horizons
        horizons = self.config.data.time_series.prediction_horizons
        base_target = self.config.data.schema.target_column

        # Generate target column names for each horizon
        target_cols = []
        for h in horizons:
            target_cols.append(f"{base_target}_{h}d")

        logger.info(f"✅ Generated target columns from horizons: {target_cols}")
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
