"""Dagster assets that wrap gogooku5 dataset operations."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from builder.chunks import ChunkPlanner, ChunkSpec
from builder.config.settings import DatasetBuilderSettings
from builder.pipelines.dataset_builder import DatasetBuilder
from dagster import Failure, Field, asset

REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass
class _ChunkIds:
    output_dir: Path
    chunks_dir: Path
    chunk_ids: List[str]

    def as_jsonable(self) -> Dict[str, str | List[str]]:
        return {
            "output_dir": str(self.output_dir),
            "chunks_dir": str(self.chunks_dir),
            "chunk_ids": self.chunk_ids,
        }


@asset(
    name="g5_dataset_chunks",
    group_name="gogooku5_dataset",
    compute_kind="DatasetBuilder",
    io_manager_key="io_manager",
    required_resource_keys={"dataset_builder"},
    config_schema={
        "start": Field(str, description="Chunk build start date (YYYY-MM-DD)"),
        "end": Field(str, description="Chunk build end date (YYYY-MM-DD)"),
        "chunk_months": Field(int, default_value=3, description="Number of calendar months per chunk"),
        "latest_only": Field(bool, default_value=False, description="Only build the latest chunk"),
        "resume": Field(bool, default_value=False, description="Skip completed chunks based on status.json"),
        "force": Field(bool, default_value=False, description="Force rebuild chunks even if completed"),
        "refresh_listed": Field(bool, default_value=False, description="Refresh listed metadata before first chunk"),
    },
)
def build_dataset_chunks(
    context,
):
    """
    Build dataset chunks for the requested date range.
    """

    config = context.op_config or {}
    dataset_builder: DatasetBuilder = context.resources.dataset_builder
    settings: DatasetBuilderSettings = dataset_builder.settings
    planner = ChunkPlanner(
        months_per_chunk=int(config["chunk_months"]),
        output_root=settings.data_output_dir / "chunks",
    )

    chunk_specs: List[ChunkSpec] = planner.plan(start=config["start"], end=config["end"])
    if config.get("latest_only") and chunk_specs:
        chunk_specs = [chunk_specs[-1]]

    executed: List[ChunkSpec] = []
    refresh_flag = bool(config.get("refresh_listed")) or getattr(dataset_builder, "_dagster_refresh_listed", False)
    for idx, spec in enumerate(chunk_specs, start=1):
        if _should_skip_chunk(spec, bool(config.get("resume")), bool(config.get("force"))):
            context.log.info("⏭️  Skipping completed chunk %s", spec.chunk_id)
            continue

        try:
            context.log.info(
                "🚧 Building chunk %s (%d/%d): %s → %s",
                spec.chunk_id,
                idx,
                len(chunk_specs),
                spec.output_start,
                spec.output_end,
            )
            dataset_builder.build_chunk(spec, refresh_listed=refresh_flag)
            refresh_flag = False
            executed.append(spec)
            context.log.info("✅ Chunk %s completed", spec.chunk_id)
        except Exception as exc:  # pragma: no cover - execution failure handled in Dagster
            raise Failure(f"Chunk {spec.chunk_id} failed: {exc}") from exc

    if not executed:
        context.log.warning("No chunks executed (all skipped). Returning existing metadata.")

    result = _ChunkIds(
        output_dir=settings.data_output_dir,
        chunks_dir=settings.data_output_dir / "chunks",
        chunk_ids=[spec.chunk_id for spec in executed],
    )

    context.log.info(
        "Chunk build summary: output_dir=%s, chunks_dir=%s, chunk_ids=%s",
        result.output_dir,
        result.chunks_dir,
        result.chunk_ids,
    )

    return result.as_jsonable()


@asset(
    name="g5_dataset_full",
    group_name="gogooku5_dataset",
    compute_kind="merge",
    deps=[build_dataset_chunks],
    config_schema={
        "allow_partial": Field(bool, default_value=False, description="Allow merge even if some chunks incomplete"),
        "strict": Field(bool, default_value=False, description="Fail if any chunk is missing"),
    },
)
def merge_latest_dataset(
    context,
    g5_dataset_chunks,
):
    """
    Merge completed chunks into the latest dataset artifacts.
    """

    chunks_dir = Path(g5_dataset_chunks["chunks_dir"])  # type: ignore[arg-type]
    output_dir = Path(g5_dataset_chunks["output_dir"])  # type: ignore[arg-type]
    config = context.op_config or {}
    chunk_ids = g5_dataset_chunks.get("chunk_ids", [])

    if not chunk_ids:
        context.log.warning("merge_latest_dataset: no new chunks detected, running merge anyway")

    merge_script = REPO_ROOT / "data" / "tools" / "merge_chunks.py"
    cmd = [
        sys.executable,
        str(merge_script),
        "--chunks-dir",
        str(chunks_dir),
        "--output-dir",
        str(output_dir),
    ]
    if config.get("allow_partial"):
        cmd.append("--allow-partial")
    if config.get("strict"):
        cmd.append("--strict")

    context.log.info("🔗 Running merge command: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    context.log.info(result.stdout)
    if result.returncode != 0:
        raise Failure(f"merge_chunks failed (exit_code={result.returncode})")

    latest = output_dir / "ml_dataset_latest_full.parquet"
    context.log.info("✅ Merge complete: %s", latest)
    return {"latest_parquet": str(latest)}


def _should_skip_chunk(spec: ChunkSpec, resume: bool, force: bool) -> bool:
    if force or not resume:
        return False
    try:
        payload = json.loads(spec.status_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False
    except json.JSONDecodeError:
        return False
    return payload.get("state") == "completed"
