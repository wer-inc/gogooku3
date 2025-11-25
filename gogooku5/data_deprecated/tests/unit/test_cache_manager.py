from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from builder.utils.cache import CacheManager


def test_cache_manager_dataframe_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JQUANTS_AUTH_EMAIL", "user@example.com")
    monkeypatch.setenv("JQUANTS_AUTH_PASSWORD", "secret")
    monkeypatch.setenv("DATA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("DATA_CACHE_DIR", str(tmp_path / "cache"))

    manager = CacheManager()

    df = pl.DataFrame({"value": [1, 2, 3]})
    manager.save_dataframe("sample", df)
    assert manager.has_cache("sample") is True

    reloaded = manager.load_dataframe("sample")
    assert reloaded is not None
    assert reloaded.select("value").to_series().to_list() == [1, 2, 3]

    index = manager.load_index()
    index["sample"] = {"rows": 3}
    manager.save_index(index)
    assert manager.load_index()["sample"]["rows"] == 3

    manager.invalidate("sample")
    assert manager.has_cache("sample") is False

    # Ensure global invalidation works
    manager.save_dataframe("sample", df)
    manager.invalidate()
    assert manager.has_cache("sample") is False


def test_cache_get_or_fetch_respects_ttl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JQUANTS_AUTH_EMAIL", "user@example.com")
    monkeypatch.setenv("JQUANTS_AUTH_PASSWORD", "secret")
    monkeypatch.setenv("DATA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("DATA_CACHE_DIR", str(tmp_path / "cache"))

    manager = CacheManager()

    calls = {"count": 0}

    def _fetch() -> pl.DataFrame:
        calls["count"] += 1
        return pl.DataFrame({"value": [calls["count"]]})

    df, hit = manager.get_or_fetch_dataframe("ttl_sample", _fetch, ttl_days=1)
    assert hit is False
    assert df.select("value").item(0, 0) == 1
    assert calls["count"] == 1

    df2, hit2 = manager.get_or_fetch_dataframe("ttl_sample", _fetch, ttl_days=1)
    assert hit2 is True
    assert df2.select("value").item(0, 0) == 1
    assert calls["count"] == 1

    # Force expiry by manipulating index timestamp
    index = manager.load_index()
    entry = index["ttl_sample"]
    entry["updated_at"] = "2000-01-01T00:00:00"
    manager.save_index(index)

    df3, hit3 = manager.get_or_fetch_dataframe("ttl_sample", _fetch, ttl_days=1)
    assert hit3 is False
    assert df3.select("value").item(0, 0) == 2
    assert calls["count"] == 2
