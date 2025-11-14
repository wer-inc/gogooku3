"""Dagster assets that wrap gogooku5 dataset operations."""

from __future__ import annotations

import json
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from builder.chunks import ChunkPlanner, ChunkSpec
from builder.config.settings import DatasetBuilderSettings
from builder.pipelines.dataset_builder import DatasetBuilder
from builder.utils.schema_validator import SchemaValidator
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
    tracker = getattr(dataset_builder, "mlflow_tracker", None)
    dagster_run_id = getattr(context, "run_id", None)
    run_params = {
        "start": config["start"],
        "end": config["end"],
        "chunk_months": int(config["chunk_months"]),
        "latest_only": bool(config.get("latest_only")),
        "resume": bool(config.get("resume")),
        "force": bool(config.get("force")),
        "refresh_listed": bool(config.get("refresh_listed")),
    }

    run_meta = getattr(dataset_builder, "_run_meta", None)
    if isinstance(run_meta, dict):
        run_meta["dagster_run_id"] = dagster_run_id
    else:
        dataset_builder._run_meta = {"dagster_run_id": dagster_run_id}
    planner = ChunkPlanner(
        months_per_chunk=int(config["chunk_months"]),
        output_root=settings.data_output_dir / "chunks",
    )

    chunk_specs: List[ChunkSpec] = planner.plan(start=config["start"], end=config["end"])
    if config.get("latest_only") and chunk_specs:
        chunk_specs = [chunk_specs[-1]]

    executed: List[ChunkSpec] = []
    refresh_flag = bool(config.get("refresh_listed")) or getattr(dataset_builder, "_dagster_refresh_listed", False)
    mlflow_cm = (
        tracker.start_run(
            stage="dataset_chunks",
            dagster_run_id=dagster_run_id,
            params=run_params,
            tags={"asset": "g5_dataset_chunks"},
        )
        if tracker and tracker.enabled
        else nullcontext()
    )

    with mlflow_cm:
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

                if tracker and tracker.enabled:
                    rows_logged = None
                    if spec.metadata_path.exists():
                        try:
                            meta = json.loads(spec.metadata_path.read_text(encoding="utf-8"))
                            rows_logged = meta.get("rows")
                        except json.JSONDecodeError:
                            rows_logged = None
                        tracker.log_artifact(str(spec.metadata_path), artifact_path=f"chunks/{spec.chunk_id}")
                    tracker.log_metrics(
                        {
                            "chunks_completed": float(len(executed)),
                            "chunk_rows": float(rows_logged or 0),
                        },
                        step=len(executed),
                    )
            except Exception as exc:  # pragma: no cover - execution failure handled in Dagster
                raise Failure(f"Chunk {spec.chunk_id} failed: {exc}") from exc

        if tracker and tracker.enabled:
            tracker.log_metrics(
                {
                    "chunks_total": float(len(chunk_specs)),
                    "chunks_completed": float(len(executed)),
                }
            )
            if executed:
                tracker.log_params({"completed_chunks": ",".join(spec.chunk_id for spec in executed)})

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
    name="g5_schema_gate",
    group_name="gogooku5_dataset",
    compute_kind="validation",
    deps=[build_dataset_chunks],
    required_resource_keys={"dataset_builder"},
)
def validate_chunk_schemas(context, g5_dataset_chunks):
    """
    Ensure all completed chunks share the manifest schema hash before merge.
    """

    chunks_dir = Path(g5_dataset_chunks["chunks_dir"])  # type: ignore[arg-type]
    manifest_version = None
    try:
        validator = SchemaValidator()
        manifest_hash = validator.manifest_hash
        manifest_version = validator.manifest_version
    except FileNotFoundError as exc:
        raise Failure(f"Schema manifest missing: {exc}") from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise Failure(f"Failed to initialize schema validator: {exc}") from exc

    if not chunks_dir.exists():
        context.log.warning("g5_schema_gate: chunks directory %s does not exist", chunks_dir)
        return {"schema_hash": None, "validated_chunks": []}

    completed_hashes: Dict[str, List[str]] = {}
    schema_failed: List[str] = []
    completed_count = 0

    for chunk_dir in sorted(chunks_dir.iterdir()):
        if not chunk_dir.is_dir():
            continue

        status_path = chunk_dir / "status.json"
        try:
            status_payload = json.loads(status_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        except json.JSONDecodeError:
            context.log.warning("Malformed status.json for chunk %s", chunk_dir.name)
            continue

        state = status_payload.get("state")
        if state == "failed_schema_mismatch":
            schema_failed.append(chunk_dir.name)
            continue
        if state != "completed":
            continue

        metadata_path = chunk_dir / "metadata.json"
        schema_hash = None
        if metadata_path.exists():
            try:
                metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                schema_hash = metadata_payload.get("feature_schema_hash")
            except json.JSONDecodeError:
                schema_hash = None

        if schema_hash is None:
            result = validator.validate_parquet(chunk_dir / "ml_dataset.parquet")
            schema_hash = result.schema_hash

        completed_hashes.setdefault(schema_hash, []).append(chunk_dir.name)
        completed_count += 1

    if schema_failed:
        raise Failure(
            f"{len(schema_failed)} chunks marked failed_schema_mismatch: {', '.join(schema_failed[:5])}"
            + (" ..." if len(schema_failed) > 5 else "")
        )

    if not completed_hashes:
        context.log.warning("g5_schema_gate: no completed chunks found in %s", chunks_dir)
        return {"schema_hash": None, "validated_chunks": []}

    if len(completed_hashes) > 1:
        detail = "; ".join(f"{hash_val}: {ids[:3]}" for hash_val, ids in completed_hashes.items())
        raise Failure(f"Schema hash mismatch across chunks: {detail}")

    schema_hash = next(iter(completed_hashes))
    if schema_hash != manifest_hash:
        raise Failure(
            f"Chunk schema hash {schema_hash} does not match manifest hash {manifest_hash}. "
            "Rebuild chunks to align with the manifest."
        )

    context.log.info(
        "Schema gate passed: %d completed chunks match manifest version %s (hash=%s)",
        completed_count,
        manifest_version,
        schema_hash,
    )
    return {
        "schema_hash": schema_hash,
        "validated_chunks": completed_hashes[schema_hash],
        "manifest_version": manifest_version,
    }


@asset(
    name="g5_dataset_full",
    group_name="gogooku5_dataset",
    compute_kind="merge",
    deps=[build_dataset_chunks, validate_chunk_schemas],
    required_resource_keys={"dataset_builder"},
    config_schema={
        "allow_partial": Field(bool, default_value=False, description="Allow merge even if some chunks incomplete"),
        "strict": Field(bool, default_value=False, description="Fail if any chunk is missing"),
    },
)
def merge_latest_dataset(
    context,
    g5_dataset_chunks,
    g5_schema_gate,
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

    dataset_builder: DatasetBuilder = context.resources.dataset_builder
    tracker = getattr(dataset_builder, "mlflow_tracker", None)
    dagster_run_id = getattr(context, "run_id", None)
    run_params = {
        "allow_partial": bool(config.get("allow_partial")),
        "strict": bool(config.get("strict")),
        "chunk_count": len(g5_dataset_chunks.get("chunk_ids", [])),
    }

    mlflow_cm = (
        tracker.start_run(
            stage="dataset_merge",
            dagster_run_id=dagster_run_id,
            params=run_params,
            tags={"asset": "g5_dataset_full"},
        )
        if tracker and tracker.enabled
        else nullcontext()
    )

    with mlflow_cm:
        merge_script = REPO_ROOT / "data" / "tools" / "merge_chunks.py"
        cmd = [
            sys.executable,
            str(merge_script),
            "--chunks-dir",
            str(chunks_dir),
            "--output-dir",
            str(output_dir),
        ]
        required_hash = g5_schema_gate.get("schema_hash") if isinstance(g5_schema_gate, dict) else None
        if required_hash:
            cmd.extend(["--require-schema-hash", required_hash])
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
        metadata_path = latest.with_name(latest.stem + "_metadata.json")
        dataset_metadata = {}
        if metadata_path.exists():
            try:
                dataset_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                context.log.warning("Failed to parse dataset metadata: %s", metadata_path)

        context.log.info("✅ Merge complete: %s", latest)
        dataset_hash = dataset_metadata.get("dataset_hash")
        schema_version = dataset_metadata.get("feature_schema_version")
        if dataset_hash:
            context.log.info("dataset_hash=%s", dataset_hash)
        if schema_version:
            context.log.info("feature_schema_version=%s", schema_version)

        if tracker and tracker.enabled:
            tracker.log_params(
                {
                    "latest_parquet": str(latest),
                    "dataset_hash": dataset_hash,
                    "feature_schema_version": schema_version,
                }
            )
            if dataset_metadata:
                rows = dataset_metadata.get("rows")
                if rows is not None:
                    tracker.log_metrics({"merged_rows": float(rows or 0)})
            if metadata_path.exists():
                tracker.log_artifact(str(metadata_path), artifact_path="merge")

        return {
            "latest_parquet": str(latest),
            "dataset_hash": dataset_hash,
            "feature_schema_version": schema_version,
        }


def _should_skip_chunk(spec: ChunkSpec, resume: bool, force: bool) -> bool:
    """Check if chunk should be skipped based on status and file integrity.

    Args:
        spec: Chunk specification
        resume: Whether resume mode is enabled
        force: Whether force rebuild is enabled

    Returns:
        True if chunk should be skipped, False if it should be built

    Enhanced checks:
    - Verifies status.json exists and is valid JSON
    - Checks state is "completed" (not "failed_schema_mismatch")
    - Validates parquet file exists
    - Validates parquet file is readable and has expected row count
    """
    import logging
    import polars as pl

    logger = logging.getLogger(__name__)

    # Force rebuild overrides all checks
    if force:
        return False

    # Resume mode disabled → rebuild everything
    if not resume:
        return False

    # Check status.json exists and is valid
    try:
        payload = json.loads(spec.status_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.debug("[RESUME] No status.json for %s → rebuild", spec.chunk_id)
        return False
    except json.JSONDecodeError:
        logger.warning("[RESUME] Malformed status.json for %s → rebuild", spec.chunk_id)
        return False

    state = payload.get("state")

    # Only skip "completed" chunks (not "failed_schema_mismatch")
    # Failed schema chunks should be retried after schema fixes
    if state != "completed":
        logger.debug("[RESUME] State=%s for %s → rebuild", state, spec.chunk_id)
        return False

    # Check parquet file exists
    parquet_path = spec.output_dir / "ml_dataset.parquet"
    if not parquet_path.exists():
        logger.warning(
            "[RESUME] Status shows completed but parquet missing for %s → rebuild", spec.chunk_id
        )
        return False

    # Validate parquet file integrity and row count
    try:
        # Read just the schema to verify file is readable
        df_scan = pl.scan_parquet(parquet_path)
        actual_rows = df_scan.select(pl.count()).collect().item()

        # Check row count matches status.json
        expected_rows = payload.get("rows")
        if expected_rows is not None and actual_rows != expected_rows:
            logger.warning(
                "[RESUME] Row count mismatch for %s (expected %d, got %d) → rebuild",
                spec.chunk_id,
                expected_rows,
                actual_rows,
            )
            return False

        logger.info(
            "[RESUME] ✅ Valid completed chunk %s (%d rows) → skip", spec.chunk_id, actual_rows
        )
        return True

    except Exception as exc:
        logger.warning(
            "[RESUME] Parquet validation failed for %s: %s → rebuild", spec.chunk_id, exc
        )
        return False
