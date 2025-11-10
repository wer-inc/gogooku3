"""Merge completed chunk datasets into the latest full dataset."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import polars as pl
from builder.config.settings import get_settings
from builder.utils import ensure_env_loaded
from builder.utils.artifacts import DatasetArtifactWriter
from builder.utils.lazy_io import lazy_load
from builder.utils.logger import get_logger
from builder.utils.storage import StorageClient

LOGGER = get_logger("merge_chunks")


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    input_start: str
    input_end: str
    output_start: str
    output_end: str
    rows: int
    dataset_path: Path
    metadata_path: Path
    status_path: Path
    directory: Path

    @property
    def status(self) -> dict:
        try:
            return json.loads(self.status_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge dataset chunks into a full dataset")
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        help="Directory containing chunk subdirectories (default: <DATA_OUTPUT_DIR>/chunks)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory for merged dataset (default: DATA_OUTPUT_DIR)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any chunk is missing or incomplete instead of skipping",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow merging even if some chunks are incomplete (default: fail).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_env_loaded()
    settings = get_settings()
    chunks_root = args.chunks_dir or (settings.data_output_dir / "chunks")
    if args.output_dir:
        settings.data_output_dir = args.output_dir

    if not chunks_root.exists():
        LOGGER.error("Chunks directory does not exist: %s", chunks_root)
        return 1

    chunks = _collect_chunks(chunks_root)
    if not chunks:
        LOGGER.error("No chunk metadata found in %s", chunks_root)
        return 1

    completed = [record for record in chunks if record.status.get("state") == "completed"]
    incomplete = [record for record in chunks if record.status.get("state") != "completed"]
    if incomplete and not args.allow_partial:
        missing = ", ".join(record.chunk_id for record in incomplete)
        LOGGER.error(
            "Incomplete chunks detected (%s). Use --allow-partial to merge anyway or resolve the failed chunks.",
            missing,
        )
        return 1
    if args.strict and incomplete:
        missing = ", ".join(record.chunk_id for record in incomplete)
        LOGGER.error("Strict mode: incomplete chunks detected: %s", missing)
        return 1
    if incomplete:
        LOGGER.warning(
            "Proceeding with partial merge after skipping incomplete chunks: %s",
            ", ".join(record.chunk_id for record in incomplete),
        )

    if not completed:
        LOGGER.error("No completed chunks to merge in %s", chunks_root)
        return 1

    completed.sort(key=lambda record: (record.output_start, record.chunk_id))

    LOGGER.info("Merging %d chunks from %s", len(completed), chunks_root)
    dfs: List[pl.DataFrame] = []
    schema_reference: dict[str, pl.DataType] | None = None
    chunk_summaries: List[dict[str, object]] = []

    for record in completed:
        LOGGER.info("📦 Loading chunk %s", record.chunk_id)
        df = lazy_load(record.dataset_path, prefer_ipc=True)
        df = _clip_chunk_to_range(df, record)
        dfs.append(df)
        chunk_summaries.append(
            {
                "id": record.chunk_id,
                "input_start": record.input_start,
                "input_end": record.input_end,
                "output_start": record.output_start,
                "output_end": record.output_end,
                "rows": df.height,
                "dataset_path": str(record.dataset_path),
                "metadata_path": str(record.metadata_path),
            }
        )
        if schema_reference is None:
            schema_reference = df.schema
        else:
            if df.schema != schema_reference:
                raise ValueError(f"Schema mismatch for chunk {record.chunk_id}")

    merged = pl.concat(dfs, how="vertical_relaxed", rechunk=True)
    LOGGER.info("✅ Concatenated dataset: %d rows × %d columns", merged.height, merged.width)

    start_date = chunk_summaries[0]["output_start"]
    end_date = chunk_summaries[-1]["output_end"]
    extra_meta = {"chunks": chunk_summaries}

    writer = DatasetArtifactWriter(settings=settings)
    artifact = writer.write(
        merged,
        start_date=start_date,
        end_date=end_date,
        extra_metadata=extra_meta,
    )
    LOGGER.info("📁 Merged dataset stored at %s", artifact.parquet_path)

    storage = StorageClient(settings=settings)
    storage.ensure_remote_symlink(target=str(artifact.latest_symlink))

    LOGGER.info("📄 Latest metadata: %s", artifact.metadata_path)
    return 0


def _collect_chunks(root: Path) -> List[ChunkRecord]:
    records: List[ChunkRecord] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        metadata_path = path / "metadata.json"
        status_path = path / "status.json"
        dataset_path = path / "ml_dataset.parquet"
        if not metadata_path.exists() or not dataset_path.exists():
            LOGGER.warning("Skipping %s (missing metadata or dataset)", path)
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        record = ChunkRecord(
            chunk_id=metadata["chunk_id"],
            input_start=metadata["input_start"],
            input_end=metadata["input_end"],
            output_start=metadata["output_start"],
            output_end=metadata["output_end"],
            rows=int(metadata.get("rows", 0)),
            dataset_path=dataset_path,
            metadata_path=metadata_path,
            status_path=status_path,
            directory=path,
        )
        records.append(record)
    return records


def _clip_chunk_to_range(df: pl.DataFrame, record: ChunkRecord) -> pl.DataFrame:
    """Ensure chunk rows align with the declared output range (inclusive)."""

    date_col = next((col for col in ("date", "Date") if col in df.columns), None)
    if date_col is None:
        LOGGER.warning("Chunk %s has no date column; skipping date validation", record.chunk_id)
        return df

    date_expr = pl.col(date_col).cast(pl.Utf8, strict=False)
    try:
        min_date, max_date = df.select([date_expr.min().alias("min_date"), date_expr.max().alias("max_date")]).row(0)
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning("Failed to inspect date bounds for chunk %s: %s", record.chunk_id, exc)
        return df

    if min_date is None or max_date is None:
        LOGGER.warning("Chunk %s date column %s contains nulls only; skipping trim", record.chunk_id, date_col)
        return df

    if min_date == record.output_start and max_date == record.output_end:
        return df

    LOGGER.warning(
        "Chunk %s date range mismatch: expected %s→%s, actual %s→%s. Trimming to declared bounds.",
        record.chunk_id,
        record.output_start,
        record.output_end,
        min_date,
        max_date,
    )

    trimmed = (
        df.with_columns(date_expr.alias("__merge_date"))
        .filter((pl.col("__merge_date") >= record.output_start) & (pl.col("__merge_date") <= record.output_end))
        .drop("__merge_date")
    )
    if trimmed.is_empty():
        raise ValueError(
            f"Chunk {record.chunk_id} has no rows within declared range "
            f"{record.output_start}→{record.output_end}. Rebuild the chunk."
        )
    return trimmed


if __name__ == "__main__":
    raise SystemExit(main())
