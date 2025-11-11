"""Cache management helpers with Arrow IPC support.

This module provides high-performance caching with:
- Arrow IPC format for 3-5x faster reads (zero-copy mmap)
- Automatic fallback to Parquet for compatibility
- Dual-format support (IPC + Parquet)
- TTL-based cache validation
- Thread-safe operations with POSIX file locking

Performance gains:
- IPC cache reads: 3-5x faster than Parquet
- Backward compatible: Parquet-only mode available
- Disk usage: +10-20% for IPC cache (configurable)
"""
from __future__ import annotations

import fcntl
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Literal, Optional, Tuple

import polars as pl

from ..config import DatasetBuilderSettings, get_settings
from .lazy_io import lazy_load
from .logger import get_logger

LOGGER = get_logger("cache")


@dataclass
class CacheManager:
    """Manage cache metadata stored alongside dataset artifacts."""

    settings: DatasetBuilderSettings = field(default_factory=get_settings)

    def __post_init__(self) -> None:
        self.cache_dir = self.settings.data_cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.settings.default_cache_index_path

    def load_index(self) -> Dict[str, Any]:
        """Load cache metadata from disk if it exists."""

        if not self._index_path.exists():
            LOGGER.debug("Cache index not found at %s", self._index_path)
            return {}
        with self._index_lock(shared=True):
            try:
                return json.loads(self._index_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                LOGGER.warning("Failed to decode cache index: %s", exc)
                return {}

    def save_index(self, index: Dict[str, Any]) -> None:
        """Persist cache metadata to disk."""

        with self._index_lock(shared=False):
            self._write_index(index)
        LOGGER.debug("Wrote cache index with %d keys", len(index))

    def cache_file(self, key: str, format: Literal["parquet", "ipc"] = "parquet") -> Path:
        """Return a deterministic cache file path for a key.

        Args:
            key: Cache key identifier
            format: File format ("parquet" or "ipc"). Defaults to "parquet".

        Returns:
            Path to cache file with appropriate extension.

        Example:
            >>> cache_mgr.cache_file("features_2023", format="ipc")
            PosixPath('output/cache/features_2023.arrow')
        """
        ext = ".arrow" if format == "ipc" else ".parquet"
        filename = f"{key}{ext}"
        return self.cache_dir / filename

    def load_dataframe(self, key: str, prefer_ipc: bool = True) -> Optional[pl.DataFrame]:
        """Load a cached dataframe if present.

        Strategy:
            1. Try IPC first (if prefer_ipc=True and .arrow exists) → 3-5x faster
            2. Fallback to Parquet
            3. Return None if neither exists

        Args:
            key: Cache key identifier
            prefer_ipc: Try Arrow IPC format first. Defaults to True.

        Returns:
            Cached DataFrame or None if not found.

        Performance:
            - IPC read: Zero-copy mmap, 3-5x faster than Parquet
            - Parquet read: Universal compatibility

        Example:
            >>> df = cache_mgr.load_dataframe("features_2023")
            >>> # Tries: features_2023.arrow → features_2023.parquet
        """
        # Use lazy_load for optimal performance (IPC-first with fallback)
        # Try IPC first (faster)
        if prefer_ipc:
            ipc_path = self.cache_file(key, format="ipc")
            if ipc_path.exists():
                try:
                    LOGGER.debug("Loading from IPC cache (fast): %s", ipc_path)
                    # Use lazy_load for predicate pushdown and column pruning support
                    return lazy_load(ipc_path, prefer_ipc=True)
                except Exception as exc:
                    LOGGER.warning("Failed to read IPC cache %s: %s. Trying Parquet.", ipc_path, exc)

        # Fallback to Parquet (compatible)
        parquet_path = self.cache_file(key, format="parquet")
        if parquet_path.exists():
            LOGGER.debug("Loading from Parquet cache: %s", parquet_path)
            # Use lazy_load for optimal performance
            return lazy_load(parquet_path, prefer_ipc=False)

        # Neither format exists
        return None

    def save_dataframe(
        self,
        key: str,
        df: pl.DataFrame,
        format: Literal["parquet", "ipc"] = "ipc",
        dual_format: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Store a dataframe in the cache.

        Args:
            key: Cache key identifier
            df: DataFrame to cache
            format: Primary format ("parquet" or "ipc"). Defaults to "ipc".
            dual_format: Also save in secondary format. Defaults to True.
                If format="ipc" and dual_format=True, saves both .arrow and .parquet.
                If format="parquet" and dual_format=True, saves both .parquet and .arrow.

        Returns:
            Path to primary format file.

        Performance:
            - IPC write: Similar speed to Parquet, but 3-5x faster reads
            - Dual format: +10-20% write time, optimal read performance
            - Disk usage: +10-20% for second format

        Example:
            >>> # Save with dual format (recommended)
            >>> path = cache_mgr.save_dataframe("features", df, format="ipc", dual_format=True)
            >>> # Creates: features.arrow (primary) + features.parquet (backup)
        """
        # Write primary format
        primary_path = self.cache_file(key, format=format)
        if format == "ipc":
            df.write_ipc(primary_path, compression="lz4")
            LOGGER.debug("Saved IPC cache: %s (%d rows, %d cols)", primary_path, df.height, df.width)
        else:
            df.write_parquet(primary_path)
            LOGGER.debug("Saved Parquet cache: %s (%d rows, %d cols)", primary_path, df.height, df.width)

        # Optionally write secondary format
        if dual_format:
            secondary_format = "parquet" if format == "ipc" else "ipc"
            secondary_path = self.cache_file(key, format=secondary_format)
            try:
                if secondary_format == "ipc":
                    df.write_ipc(secondary_path, compression="lz4")
                else:
                    df.write_parquet(secondary_path)
                LOGGER.debug("Saved secondary format: %s", secondary_path)
            except Exception as exc:
                LOGGER.warning("Failed to save secondary format %s: %s", secondary_path, exc)

        # Update cache index with format information
        def _mutator(idx: Dict[str, Any]) -> None:
            idx[key] = {
                "rows": df.height,
                "format": format,
                "dual_format": dual_format,
                "updated_at": datetime.utcnow().isoformat(),
            }
            if metadata:
                idx[key]["metadata"] = metadata

        self._update_index(_mutator)
        return primary_path

    def has_cache(self, key: str, any_format: bool = True) -> bool:
        """Check if cache data exists for the given key.

        Args:
            key: Cache key identifier
            any_format: If True, checks for either IPC or Parquet. Defaults to True.
                If False, only checks for Parquet (backward compatible).

        Returns:
            True if cache file exists in at least one format.

        Example:
            >>> cache_mgr.has_cache("features_2023")  # Checks .arrow or .parquet
            True
        """
        if any_format:
            # Check either format
            ipc_path = self.cache_file(key, format="ipc")
            parquet_path = self.cache_file(key, format="parquet")
            exists = ipc_path.exists() or parquet_path.exists()
        else:
            # Check only Parquet (backward compatible)
            parquet_path = self.cache_file(key, format="parquet")
            exists = parquet_path.exists()

        LOGGER.debug("Cache %s exists=%s (any_format=%s)", key, exists, any_format)
        return exists

    def invalidate(self, key: Optional[str] = None) -> None:
        """Remove cache files (both IPC and Parquet if present).

        Args:
            key: Cache key to invalidate. If None, clears entire cache directory.

        Note:
            When invalidating a specific key, removes both .arrow and .parquet files.
        """
        if key is None:
            LOGGER.info("Invalidating entire cache directory at %s", self.cache_dir)
            for file in self.cache_dir.glob("*"):
                if file.is_file():
                    file.unlink()
            return

        # Remove both formats if they exist
        for format in ("ipc", "parquet"):
            path = self.cache_file(key, format=format)
            if path.exists():
                LOGGER.info("Removing %s cache file: %s", format.upper(), path)
                path.unlink()

    # ------------------------------------------------------------------
    # Extended helpers
    # ------------------------------------------------------------------
    def is_valid(self, key: str, ttl_days: int) -> bool:
        """Return True if cached entry exists and is within TTL."""

        if ttl_days < 0:
            ttl_days = 0

        if not self.has_cache(key):
            return False

        index = self.load_index()
        entry = index.get(key)
        if entry is None:
            return False

        if ttl_days == 0:
            return True

        updated_at = entry.get("updated_at")
        if not updated_at:
            return False
        try:
            updated = datetime.fromisoformat(str(updated_at))
        except ValueError:
            return False

        age = datetime.utcnow() - updated
        return age <= timedelta(days=ttl_days)

    def get_or_fetch_dataframe(
        self,
        key: str,
        fetch_fn: Callable[[], pl.DataFrame],
        *,
        ttl_days: Optional[int] = None,
        prefer_ipc: bool = True,
        save_format: Literal["parquet", "ipc"] = "ipc",
        dual_format: bool = True,
        allow_empty: bool = False,
        force_refresh: bool = False,
        enable_read: bool = True,
        enable_write: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pl.DataFrame, bool]:
        """Return cached dataframe or fetch and persist a fresh copy.

        Args:
            key: Cache key identifier
            fetch_fn: Function to call if cache miss (no arguments, returns DataFrame)
            ttl_days: Time-to-live in days. None uses default from settings.
            prefer_ipc: Try IPC format first when loading. Defaults to True.
            save_format: Primary format for new cache. Defaults to "ipc".
            dual_format: Save both formats. Defaults to True.
            allow_empty: If False, empty dataframes are treated as failures (no caching).

        Returns:
            Tuple of (dataframe, cache_hit).
                - dataframe: The cached or freshly fetched data
                - cache_hit: True if loaded from cache, False if fetched

        Performance:
            - Cache hit with IPC: 3-5x faster than Parquet
            - Cache miss: Saves in both formats (if dual_format=True)

        Example:
            >>> def fetch_macro_data():
            ...     return pl.DataFrame(...)
            >>>
            >>> df, hit = cache_mgr.get_or_fetch_dataframe(
            ...     "macro_2023",
            ...     fetch_macro_data,
            ...     ttl_days=7,
            ...     prefer_ipc=True
            ... )
            >>> print(f"Cache hit: {hit}")
        """
        ttl = self.settings.cache_ttl_days_default if ttl_days is None else ttl_days
        if enable_read and not force_refresh and self.is_valid(key, ttl):
            cached = self.load_dataframe(key, prefer_ipc=prefer_ipc)
            if cached is not None:
                if not allow_empty and cached.height == 0:
                    LOGGER.warning("Cache %s contained empty dataframe; treating as miss", key)
                else:
                    LOGGER.debug("Cache hit for %s (ttl=%d days)", key, ttl)
                    return cached, True

        LOGGER.debug("Cache miss for %s (ttl=%d days); fetching...", key, ttl)
        df = fetch_fn()
        if not allow_empty and df.height == 0:
            raise ValueError(f"Cache key {key} produced empty dataframe")
        if enable_write:
            self.save_dataframe(key, df, format=save_format, dual_format=dual_format, metadata=metadata)
        return df, False

    @contextmanager
    def _index_lock(self, *, shared: bool) -> Iterator[None]:
        """Context manager providing a POSIX advisory lock around index operations."""

        lock_path = self._index_path.with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _update_index(self, mutator: Callable[[Dict[str, Any]], None]) -> None:
        """Atomically mutate cache index with exclusive locking."""

        with self._index_lock(shared=False):
            try:
                current = json.loads(self._index_path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                current = {}
            except json.JSONDecodeError as exc:
                LOGGER.warning("Failed to decode cache index during update: %s", exc)
                current = {}
            mutator(current)
            self._write_index(current)

    def _write_index(self, index: Dict[str, Any]) -> None:
        """Write cache index atomically via temp file replace while lock is held."""

        tmp_path = self._index_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self._index_path)


def ensure_cache_dir(path: Path) -> Path:
    """Ensure the cache directory exists and return the path."""

    path.mkdir(parents=True, exist_ok=True)
    return path
