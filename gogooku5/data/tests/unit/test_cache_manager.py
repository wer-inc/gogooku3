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
