from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import polars as pl

from builder.chunks import ChunkPlanner
from builder.pipelines.dataset_builder import DatasetBuilder
from data.tests.helpers import make_settings


def _fake_writer(settings: object, outputs: dict):
    class FakeWriter:
        def __init__(self, *, settings):
            self.settings = settings

        def write(self, df: pl.DataFrame, *, start_date: str | None, end_date: str | None, extra_metadata: dict | None):
            parquet_path = self.settings.data_output_dir / "ml_dataset_latest.parquet"
            metadata_path = self.settings.data_output_dir / "ml_dataset_latest_metadata.json"
            df.write_parquet(parquet_path)
            metadata = {"start": start_date, "end": end_date, **(extra_metadata or {})}
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            outputs["parquet"] = parquet_path
            outputs["metadata"] = metadata_path
            return SimpleNamespace(
                parquet_path=parquet_path,
                metadata_path=metadata_path,
                latest_symlink=parquet_path,
                tagged_symlink=parquet_path,
                latest_feature_index_symlink=parquet_path,
                feature_index_path=metadata_path,
            )

    return FakeWriter


def test_chunk_pipeline_end_to_end(monkeypatch, tmp_path):
    settings = make_settings(tmp_path)
    output_root = settings.data_output_dir / "chunks"

    # Avoid dependency on holiday libraries in tests.
    monkeypatch.setattr("builder.chunks.planner.shift_trading_days", lambda start, delta: start)

    planner = ChunkPlanner(settings=settings, output_root=output_root)
    specs = planner.plan(start="2020-01-01", end="2020-06-30")
    assert [spec.chunk_id for spec in specs] == ["2020Q1", "2020Q2"]

    # Use DatasetBuilder helpers to persist synthetic chunk data.
    builder = DatasetBuilder.__new__(DatasetBuilder)
    builder.settings = settings
    builder._run_meta = {"test": "chunk-pipeline"}

    for spec in specs:
        values = pl.arange(0, 2).to_list()
        df = pl.DataFrame(
            {
                "Code": ["1001"] * 2,
                "Date": [spec.output_start, spec.output_end],
                "feature": values,
            }
        )
        builder._persist_chunk_dataset(df, spec)
        status = json.loads(spec.status_path.read_text(encoding="utf-8"))
        assert status["state"] == "completed"

    merge_module = import_module("tools.merge_chunks")

    outputs: dict[str, Path] = {}

    class FakeStorage:
        def __init__(self, *_, **__):
            pass

        def ensure_remote_symlink(self, *, target: str):
            outputs["symlink"] = Path(target)

    monkeypatch.setattr(merge_module, "ensure_env_loaded", lambda: None)
    monkeypatch.setattr(merge_module, "get_settings", lambda: settings)
    monkeypatch.setattr(merge_module, "DatasetArtifactWriter", _fake_writer(settings, outputs))
    monkeypatch.setattr(merge_module, "StorageClient", FakeStorage)

    rc = merge_module.main(
        [
            "--chunks-dir",
            str(output_root),
            "--output-dir",
            str(settings.data_output_dir),
        ]
    )
    assert rc == 0

    merged = pl.read_parquet(outputs["parquet"])
    assert merged.shape == (4, 3)
    metadata = json.loads(outputs["metadata"].read_text(encoding="utf-8"))
    assert len(metadata["chunks"]) == 2
    assert metadata["chunks"][0]["rows"] == 2

