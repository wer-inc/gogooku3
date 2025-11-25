from __future__ import annotations

from pathlib import Path

import pytest
from builder.utils.env import ensure_env_loaded, load_local_env, require_env_var


def test_load_local_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("FOO=bar\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    loaded = load_local_env(env_file)

    assert loaded is True
    assert require_env_var("FOO") == "bar"


def test_require_env_var_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING", raising=False)

    with pytest.raises(RuntimeError):
        require_env_var("MISSING")


def test_ensure_env_loaded_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    ensure_env_loaded()
    # Should not raise even when .env missing
