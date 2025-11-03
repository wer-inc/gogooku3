"""Cache management helpers."""
from __future__ import annotations

import fcntl
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import polars as pl

from ..config import DatasetBuilderSettings, get_settings
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

    def cache_file(self, key: str) -> Path:
        """Return a deterministic cache file path for a key."""

        filename = f"{key}.parquet"
        return self.cache_dir / filename

    def load_dataframe(self, key: str) -> Optional[pl.DataFrame]:
        """Load a cached dataframe if present."""

        path = self.cache_file(key)
        if not path.exists():
            return None
        return pl.read_parquet(path)

    def save_dataframe(self, key: str, df: pl.DataFrame) -> Path:
        """Store a dataframe in the cache."""

        path = self.cache_file(key)
        df.write_parquet(path)
        LOGGER.debug("Saved dataframe with %d rows to cache key %s", df.height, key)

        def _mutator(idx: Dict[str, Any]) -> None:
            idx[key] = {
                "rows": df.height,
                "updated_at": datetime.utcnow().isoformat(),
            }

        self._update_index(_mutator)
        return path

    def has_cache(self, key: str) -> bool:
        """Check if cache data exists for the given key."""

        path = self.cache_file(key)
        exists = path.exists()
        LOGGER.debug("Cache %s exists=%s", key, exists)
        return exists

    def invalidate(self, key: Optional[str] = None) -> None:
        """Remove cache files.

        If `key` is None, clears the entire cache directory while preserving structure.
        """

        if key is None:
            LOGGER.info("Invalidating entire cache directory at %s", self.cache_dir)
            for file in self.cache_dir.glob("*"):
                if file.is_file():
                    file.unlink()
            return
        path = self.cache_file(key)
        if path.exists():
            LOGGER.info("Removing cache file %s", path)
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
    ) -> Tuple[pl.DataFrame, bool]:
        """Return cached dataframe or fetch and persist a fresh copy.

        Returns:
            (dataframe, cache_hit)
        """

        ttl = self.settings.cache_ttl_days_default if ttl_days is None else ttl_days
        if self.is_valid(key, ttl):
            cached = self.load_dataframe(key)
            if cached is not None:
                LOGGER.debug("Cache hit for %s (ttl=%d days)", key, ttl)
                return cached, True

        LOGGER.debug("Cache miss for %s (ttl=%d days); fetching...", key, ttl)
        df = fetch_fn()
        self.save_dataframe(key, df)
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
