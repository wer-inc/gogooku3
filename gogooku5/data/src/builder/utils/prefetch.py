"""Thread-based prefetch helpers to overlap I/O heavy data fetches."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Generic, TypeVar

T = TypeVar("T")

LOGGER = logging.getLogger(__name__)


class DataSourcePrefetcher(Generic[T]):
    """Small thread pool that prefetches expensive data source calls."""

    def __init__(self, max_workers: int = 0, *, logger: logging.Logger | None = None) -> None:
        self._max_workers = max_workers if max_workers is not None else 0
        self._logger = logger or LOGGER
        self._executor = (
            ThreadPoolExecutor(max_workers=self._max_workers, thread_name_prefix="prefetch")
            if self._max_workers > 0
            else None
        )
        self._futures: dict[str, Future[T]] = {}
        self._lock = threading.Lock()

    def __enter__(self) -> "DataSourcePrefetcher[T]":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def enabled(self) -> bool:
        return self._executor is not None

    def schedule(self, key: str, fn: Callable[[], T]) -> bool:
        """Submit a callable if the prefetcher is enabled and key unused."""

        if self._executor is None:
            return False
        with self._lock:
            if key in self._futures:
                return False
            self._futures[key] = self._executor.submit(self._run_with_timing, key, fn)
        return True

    def resolve(self, key: str, fallback_fn: Callable[[], T] | None = None) -> T:
        """Return prefetched result or execute fallback synchronously."""

        future: Future[T] | None = None
        with self._lock:
            future = self._futures.pop(key, None)

        if future is not None:
            return future.result()

        if fallback_fn is None:
            raise KeyError(f"No prefetched result or fallback for key='{key}'")

        return fallback_fn()

    def close(self) -> None:
        """Shut down executor without blocking ongoing tasks."""

        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=False)
            self._executor = None
        self._futures.clear()

    def _run_with_timing(self, key: str, fn: Callable[[], T]) -> T:
        start = time.perf_counter()
        try:
            return fn()
        finally:
            elapsed = time.perf_counter() - start
            self._logger.info("[PREFETCH] %s completed in %.2fs", key, elapsed)
