from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import List

import pytest
from builder.chunks import ChunkSpec


@pytest.fixture()
def cli_module():
    module = import_module("scripts.build_chunks")
    return module


@pytest.fixture()
def fake_specs(tmp_path: Path) -> List[ChunkSpec]:
    output_dir = tmp_path / "output" / "chunks" / "2019Q1"
    return [
        ChunkSpec(
            chunk_id="2019Q1",
            input_start="2018-09-01",
            input_end="2019-03-31",
            output_start="2019-01-01",
            output_end="2019-03-31",
            output_dir=output_dir,
        )
    ]


def test_build_chunks_dry_run(monkeypatch, capsys, fake_specs, cli_module):
    created = []

    class FakePlanner:
        def __init__(self, *, warmup_days=85, months_per_chunk=3, **_kwargs) -> None:
            self.warmup_days = warmup_days
            self.months_per_chunk = months_per_chunk
            created.append(self)

        def plan(self, *, start: str, end: str):
            return fake_specs

    monkeypatch.setattr(cli_module, "ChunkPlanner", FakePlanner)
    monkeypatch.setattr(cli_module, "ensure_env_loaded", lambda: None)

    calls: list = []

    class FakeBuilder:
        def build_chunk(self, spec, *, refresh_listed=False):
            calls.append((spec.chunk_id, refresh_listed))

    monkeypatch.setattr(cli_module, "DatasetBuilder", lambda: FakeBuilder())

    rc = cli_module.main(["--start", "2019-01-01", "--end", "2019-03-31", "--dry-run"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "Chunk build plan" in captured
    assert calls == []
    assert created and created[0].months_per_chunk == 3


def test_build_chunks_resume_skips_completed(monkeypatch, tmp_path, fake_specs, cli_module):
    created = []

    class FakePlanner:
        def __init__(self, *, warmup_days=85, months_per_chunk=3, **_kwargs) -> None:
            self.warmup_days = warmup_days
            self.months_per_chunk = months_per_chunk
            created.append(self)

        def plan(self, *, start: str, end: str):
            return fake_specs

    monkeypatch.setattr(cli_module, "ChunkPlanner", FakePlanner)
    monkeypatch.setattr(cli_module, "ensure_env_loaded", lambda: None)

    calls: list = []

    class FakeBuilder:
        def build_chunk(self, spec, *, refresh_listed=False):
            calls.append((spec.chunk_id, refresh_listed))

    monkeypatch.setattr(cli_module, "DatasetBuilder", lambda: FakeBuilder())

    spec = fake_specs[0]
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    spec.status_path.write_text(json.dumps({"state": "completed"}), encoding="utf-8")

    rc = cli_module.main(["--start", "2019-01-01", "--end", "2019-03-31", "--resume"])
    assert rc == 0
    assert calls == []
    assert created and created[0].months_per_chunk == 3


def test_build_chunks_handles_warmup_failure(monkeypatch, tmp_path, fake_specs, cli_module, capfd):
    created = []

    class FakePlanner:
        def __init__(
            self,
            *,
            warmup_days=85,
            settings=None,
            output_root=None,
            months_per_chunk=3,
            **_kwargs,
        ):
            self.warmup_days = warmup_days
            self.settings = settings
            self.output_root = output_root
            self.months_per_chunk = months_per_chunk
            created.append(self)

        def plan(self, *, start: str, end: str):
            if self.warmup_days == 85:
                raise RuntimeError("holiday calendar unavailable")
            return fake_specs

    calls: list = []

    class FakeBuilder:
        def build_chunk(self, spec, *, refresh_listed=False):
            calls.append((spec.chunk_id, refresh_listed))

    monkeypatch.setattr(cli_module, "ChunkPlanner", FakePlanner)
    monkeypatch.setattr(cli_module, "DatasetBuilder", lambda: FakeBuilder())
    monkeypatch.setattr(cli_module, "ensure_env_loaded", lambda: None)

    spec = fake_specs[0]
    spec.output_dir.mkdir(parents=True, exist_ok=True)

    rc = cli_module.main(["--start", "2019-01-01", "--end", "2019-03-31", "--no-isolation"])

    assert rc == 0
    # Two planner instances should be created (initial + fallback)
    assert len(created) == 2
    assert created[0].warmup_days == 85
    assert created[1].warmup_days == 0
    assert created[0].months_per_chunk == created[1].months_per_chunk == 3
    assert calls == [("2019Q1", False)]


def test_build_chunks_chunk_months_flag(monkeypatch, capsys, fake_specs, cli_module):
    months_seen: list[int] = []

    class FakePlanner:
        def __init__(self, *, warmup_days=85, months_per_chunk=3, **_kwargs):
            self.warmup_days = warmup_days
            self.months_per_chunk = months_per_chunk
            months_seen.append(months_per_chunk)

        def plan(self, *, start: str, end: str):
            return fake_specs

    monkeypatch.setattr(cli_module, "ChunkPlanner", FakePlanner)
    monkeypatch.setattr(cli_module, "ensure_env_loaded", lambda: None)

    rc = cli_module.main(
        [
            "--start",
            "2019-01-01",
            "--end",
            "2019-01-31",
            "--chunk-months",
            "1",
            "--dry-run",
        ]
    )
    assert rc == 0
    assert months_seen == [1]
