from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict

import pytest


def _load_script(module_name: str) -> Any:
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    path = scripts_dir / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"builder.scripts.{module_name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


@pytest.fixture(autouse=True)
def _minimal_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("JQUANTS_AUTH_EMAIL", "user@example.com")
    monkeypatch.setenv("JQUANTS_AUTH_PASSWORD", "secret")
    monkeypatch.chdir(tmp_path)


def test_build_cli_invokes_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script("build")

    captured: Dict[str, Any] = {}

    def fake_pipeline(*, start: str, end: str, refresh_listed: bool) -> None:
        captured.update({"start": start, "end": end, "refresh": refresh_listed})

    monkeypatch.setattr(module, "run_full_pipeline", fake_pipeline)

    exit_code = module.main(["--start", "2024-01-01", "--end", "2024-01-31", "--refresh-listed"])

    assert exit_code == 0
    assert captured == {"start": "2024-01-01", "end": "2024-01-31", "refresh": True}


def test_build_optimized_cli_invokes_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script("build_optimized")

    captured: Dict[str, Any] = {}

    def fake_pipeline(*, start: str, end: str, cache_only: bool) -> None:
        captured.update({"start": start, "end": end, "cache_only": cache_only})

    monkeypatch.setattr(module, "run_optimized_pipeline", fake_pipeline)

    exit_code = module.main(["--start", "2024-01-01", "--end", "2024-01-31", "--cache-only"])

    assert exit_code == 0
    assert captured == {"start": "2024-01-01", "end": "2024-01-31", "cache_only": True}
