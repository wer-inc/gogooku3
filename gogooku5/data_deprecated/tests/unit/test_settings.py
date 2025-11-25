from __future__ import annotations

from pathlib import Path

import pytest
from builder.config import DatasetBuilderSettings


def test_dataset_builder_settings_creates_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JQUANTS_AUTH_EMAIL", "user@example.com")
    monkeypatch.setenv("JQUANTS_AUTH_PASSWORD", "secret")
    monkeypatch.setenv("JQUANTS_PLAN_TIER", "premium")
    monkeypatch.setenv("DATA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("DATA_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "55")

    settings = DatasetBuilderSettings()

    assert settings.jquants_auth_email == "user@example.com"
    assert settings.jquants_plan_tier == "premium"
    assert settings.data_output_dir.exists()
    assert settings.data_cache_dir.exists()
    assert settings.request_timeout_seconds == 55
    assert settings.latest_dataset_path.parent == settings.data_output_dir
    assert settings.default_cache_index_path.parent == settings.data_cache_dir
