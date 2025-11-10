from __future__ import annotations

import json
from datetime import datetime, timedelta
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from tests.helpers import make_settings


@pytest.fixture()
def merge_module():
    return import_module("tools.merge_chunks")


def _write_chunk(directory: Path, *, chunk_id: str, start: str, end: str, rows: int = 2) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if rows <= 2:
        dates = [start, end]
    else:
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
        current = start_dt
        dates: list[str] = []
        while current <= end_dt and len(dates) < rows:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        if not dates:
            dates = [start, end]
    df = pl.DataFrame(
        {
            "Date": dates,
            "Code": ["1001"] * len(dates),
            "value": list(range(len(dates))),
        }
    )
    df.write_parquet(directory / "ml_dataset.parquet")
    metadata = {
        "chunk_id": chunk_id,
        "input_start": start,
        "input_end": end,
        "output_start": start,
        "output_end": end,
        "rows": len(dates),
        "paths": {"parquet": str(directory / "ml_dataset.parquet"), "ipc": None},
    }
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (directory / "status.json").write_text(json.dumps({"state": "completed", "rows": len(dates)}), encoding="utf-8")


def test_collect_chunks(tmp_path, merge_module):
    chunk_dir = tmp_path / "chunks" / "2019Q1"
    _write_chunk(chunk_dir, chunk_id="2019Q1", start="2019-01-01", end="2019-03-31")

    records = merge_module._collect_chunks(tmp_path / "chunks")
    assert [record.chunk_id for record in records] == ["2019Q1"]
    assert records[0].dataset_path.exists()


def test_clip_chunk_to_range(tmp_path, merge_module):
    chunk_dir = tmp_path / "chunks" / "2019Q1"
    _write_chunk(chunk_dir, chunk_id="2019Q1", start="2019-01-01", end="2019-03-31")
    record = merge_module._collect_chunks(tmp_path / "chunks")[0]
    df = pl.read_parquet(record.dataset_path)
    out = merge_module._clip_chunk_to_range(df, record)
    assert out.height == df.height

    dict_rows = df.to_dicts()
    dict_rows[0]["Date"] = "2018-12-31"
    bad_df = pl.DataFrame(dict_rows)
    trimmed = merge_module._clip_chunk_to_range(bad_df, record)
    assert trimmed.height == df.height - 1
    assert trimmed.select(pl.col("Date").min()).item() >= record.output_start
    assert trimmed.select(pl.col("Date").max()).item() == record.output_end


def test_merge_chunks_main(monkeypatch, tmp_path, merge_module):
    settings = make_settings(tmp_path)
    chunks_root = settings.data_output_dir / "chunks"
    _write_chunk(
        chunks_root / "2019Q1",
        chunk_id="2019Q1",
        start="2019-01-01",
        end="2019-03-31",
        rows=3,
    )
    _write_chunk(
        chunks_root / "2019Q2",
        chunk_id="2019Q2",
        start="2019-04-01",
        end="2019-06-30",
        rows=3,
    )

    outputs: dict[str, Path] = {}

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

    class FakeStorage:
        def __init__(self, *_, **__):
            pass

        def ensure_remote_symlink(self, *, target: str):
            outputs["symlink_target"] = Path(target)

    monkeypatch.setattr(merge_module, "ensure_env_loaded", lambda: None)
    monkeypatch.setattr(merge_module, "get_settings", lambda: settings)
    monkeypatch.setattr(merge_module, "DatasetArtifactWriter", FakeWriter)
    monkeypatch.setattr(merge_module, "StorageClient", FakeStorage)

    rc = merge_module.main(
        [
            "--chunks-dir",
            str(chunks_root),
            "--output-dir",
            str(settings.data_output_dir),
        ]
    )
    assert rc == 0
    merged = pl.read_parquet(outputs["parquet"])
    assert merged.height == 6
    metadata = json.loads(outputs["metadata"].read_text(encoding="utf-8"))
    assert metadata["chunks"][0]["id"] == "2019Q1"


def test_merge_chunks_fails_on_incomplete(monkeypatch, tmp_path, merge_module):
    settings = make_settings(tmp_path)
    chunks_root = settings.data_output_dir / "chunks"
    _write_chunk(chunks_root / "2019Q1", chunk_id="2019Q1", start="2019-01-01", end="2019-03-31")
    chunk_dir = chunks_root / "2019Q2"
    _write_chunk(chunk_dir, chunk_id="2019Q2", start="2019-04-01", end="2019-06-30")
    (chunk_dir / "status.json").write_text(json.dumps({"state": "failed"}), encoding="utf-8")

    monkeypatch.setattr(merge_module, "ensure_env_loaded", lambda: None)
    monkeypatch.setattr(merge_module, "get_settings", lambda: settings)

    rc = merge_module.main(["--chunks-dir", str(chunks_root)])
    assert rc == 1


def test_merge_chunks_allow_partial(monkeypatch, tmp_path, merge_module):
    settings = make_settings(tmp_path)
    chunks_root = settings.data_output_dir / "chunks"
    _write_chunk(
        chunks_root / "2019Q1",
        chunk_id="2019Q1",
        start="2019-01-01",
        end="2019-03-31",
        rows=2,
    )
    chunk_dir = chunks_root / "2019Q2"
    _write_chunk(
        chunk_dir,
        chunk_id="2019Q2",
        start="2019-04-01",
        end="2019-06-30",
        rows=2,
    )
    (chunk_dir / "status.json").write_text(json.dumps({"state": "failed"}), encoding="utf-8")

    outputs: dict[str, Path] = {}

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

    class FakeStorage:
        def __init__(self, *_, **__):
            pass

        def ensure_remote_symlink(self, *, target: str):
            outputs["symlink_target"] = Path(target)

    monkeypatch.setattr(merge_module, "ensure_env_loaded", lambda: None)
    monkeypatch.setattr(merge_module, "get_settings", lambda: settings)
    monkeypatch.setattr(merge_module, "DatasetArtifactWriter", FakeWriter)
    monkeypatch.setattr(merge_module, "StorageClient", FakeStorage)

    rc = merge_module.main(
        [
            "--chunks-dir",
            str(chunks_root),
            "--output-dir",
            str(settings.data_output_dir),
            "--allow-partial",
        ]
    )
    assert rc == 0
    merged = pl.read_parquet(outputs["parquet"])
    # Only one completed chunk should be merged.
    assert merged.height == 2
