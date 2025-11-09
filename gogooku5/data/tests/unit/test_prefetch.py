from __future__ import annotations

import time

import polars as pl
import pytest

from builder.utils.prefetch import DataSourcePrefetcher


def test_prefetch_schedule_and_resolve() -> None:
    with DataSourcePrefetcher(max_workers=2) as prefetcher:
        prefetcher.schedule("one", lambda: pl.DataFrame({"v": [1]}))
        result = prefetcher.resolve("one")
        assert isinstance(result, pl.DataFrame)
        assert result.select("v").item(0, 0) == 1


def test_prefetch_fallback_when_disabled() -> None:
    with DataSourcePrefetcher(max_workers=0) as prefetcher:
        result = prefetcher.resolve("missing", lambda: "fallback")
        assert result == "fallback"


def test_prefetch_propagates_exceptions() -> None:
    with DataSourcePrefetcher(max_workers=1) as prefetcher:
        prefetcher.schedule("fail", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        with pytest.raises(RuntimeError, match="boom"):
            prefetcher.resolve("fail")


def test_prefetch_logs_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class _Logger:
        def info(self, message: str, key: str, elapsed: float) -> None:
            captured["msg"] = message

    with DataSourcePrefetcher(max_workers=1, logger=_Logger()) as prefetcher:
        prefetcher.schedule("sleepy", lambda: time.sleep(0.01) or 42)
        value = prefetcher.resolve("sleepy")
        assert value == 42
