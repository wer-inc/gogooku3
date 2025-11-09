from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from builder.chunks import ChunkPlanner
from builder.chunks import ChunkSpec
from builder.pipelines.dataset_builder import DatasetBuilder
from data.tests.helpers import make_settings


def test_plan_quarter_chunks(monkeypatch, tmp_path):
    calls = []

    def fake_shift(date_str: str, days: int) -> str:
        calls.append((date_str, days))
        return f"{date_str}-warmup"

    monkeypatch.setattr("builder.chunks.planner.shift_trading_days", fake_shift)

    settings = make_settings(tmp_path)
    planner = ChunkPlanner(settings=settings, warmup_days=85, output_root=settings.data_output_dir / "chunks")

    specs = planner.plan(start="2019-01-15", end="2019-06-10")

    assert [spec.chunk_id for spec in specs] == ["2019Q1", "2019Q2"]

    first = specs[0]
    assert first.output_start == "2019-01-15"
    assert first.output_end == "2019-03-31"
    assert first.input_start == "2019-01-15-warmup"
    assert first.input_end == "2019-03-31"
    assert first.output_dir == settings.data_output_dir / "chunks" / "2019Q1"
    assert first.dataset_path.name == "ml_dataset.parquet"

    second = specs[1]
    assert second.output_start == "2019-04-01"
    assert second.output_end == "2019-06-10"
    assert second.input_start == "2019-04-01-warmup"

    assert calls == [("2019-01-15", -85), ("2019-04-01", -85)]


def test_plan_invalid_range(tmp_path):
    settings = make_settings(tmp_path)
    planner = ChunkPlanner(settings=settings, warmup_days=0)

    with pytest.raises(ValueError):
        planner.plan(start="2020-01-10", end="2020-01-05")


def test_plan_warmup_fallback(monkeypatch, tmp_path, caplog):
    def raising_shift(date_str: str, days: int) -> str:
        raise RuntimeError("holiday calendar unavailable")

    monkeypatch.setattr("builder.chunks.planner.shift_trading_days", raising_shift)

    settings = make_settings(tmp_path)
    planner = ChunkPlanner(settings=settings, warmup_days=85, output_root=settings.data_output_dir / "chunks")

    with caplog.at_level("WARNING"):
        specs = planner.plan(start="2019-01-15", end="2019-02-15")

    assert specs[0].input_start == specs[0].output_start == "2019-01-15"
    assert any("Falling back to chunk output start" in record.message for record in caplog.records)


def test_persist_chunk_dataset(tmp_path):
    settings = make_settings(tmp_path)
    builder = DatasetBuilder.__new__(DatasetBuilder)
    builder.settings = settings
    builder._run_meta = {"notes": "unit-test"}

    chunk_spec = ChunkSpec(
        chunk_id="2019Q1",
        input_start="2018-09-01",
        input_end="2019-03-31",
        output_start="2019-01-01",
        output_end="2019-03-31",
        output_dir=settings.data_output_dir / "chunks" / "2019Q1",
    )

    df = pl.DataFrame({"code": ["1001"], "date": ["2019-01-02"], "feature": [1.23]})
    path = builder._persist_chunk_dataset(df, chunk_spec)
    assert path.exists()

    metadata = json.loads(chunk_spec.metadata_path.read_text(encoding="utf-8"))
    assert metadata["chunk_id"] == "2019Q1"
    assert metadata["rows"] == 1
    assert metadata["builder_meta"]["notes"] == "unit-test"

    status = json.loads(chunk_spec.status_path.read_text(encoding="utf-8"))
    assert status["state"] == "completed"
    assert status["rows"] == 1

    builder._write_chunk_status(chunk_spec, state="failed", error="boom")
    status = json.loads(chunk_spec.status_path.read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert status["error"] == "boom"

