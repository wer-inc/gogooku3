#!/usr/bin/env python
"""Chunk health inspector for gogooku5 dataset builds."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

try:
    from builder.utils.schema_validator import SchemaValidator, SchemaValidationResult
    SCHEMA_VALIDATION_AVAILABLE = True
except ImportError:
    SCHEMA_VALIDATION_AVAILABLE = False
    SchemaValidator = None
    SchemaValidationResult = None


@dataclass
class ChunkStatus:
    chunk_id: str
    state: str | None = None
    error: str | None = None
    rows: int | None = None
    input_start: str | None = None
    input_end: str | None = None
    output_start: str | None = None
    output_end: str | None = None
    missing_status: bool = False
    missing_metadata: bool = False
    missing_parquet: bool = False
    warnings: List[str] = field(default_factory=list)
    schema_validation_result: Optional[SchemaValidationResult] = None
    schema_hash: Optional[str] = None

    @property
    def ok(self) -> bool:
        return not (
            self.missing_status
            or self.missing_metadata
            or self.missing_parquet
            or (self.state and self.state != "completed")
            or self.error
            or any(self.warnings)
        )

    @property
    def schema_ok(self) -> bool:
        """Check if schema validation passed."""
        if self.schema_validation_result is None:
            return True  # No validation performed
        return self.schema_validation_result.is_valid


def load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse {path}: {exc}") from exc


def find_metadata(chunk_dir: Path) -> Optional[Path]:
    preferred = chunk_dir / "metadata.json"
    if preferred.exists():
        return preferred
    candidates = sorted(chunk_dir.glob("*metadata.json"))
    return candidates[0] if candidates else None


def collect_chunk_status(chunk_dir: Path, validate_schema: bool = False, schema_validator: Optional[SchemaValidator] = None) -> ChunkStatus:
    chunk_id = chunk_dir.name
    status = ChunkStatus(chunk_id=chunk_id)

    status_path = chunk_dir / "status.json"
    status_payload = load_json(status_path)
    if not status_payload:
        status.missing_status = True
    else:
        status.state = status_payload.get("state")
        status.error = status_payload.get("error")

    metadata_path = find_metadata(chunk_dir)
    metadata_payload = load_json(metadata_path) if metadata_path else None
    if not metadata_payload:
        status.missing_metadata = True
    else:
        status.rows = metadata_payload.get("rows")
        status.input_start = metadata_payload.get("input_start")
        status.input_end = metadata_payload.get("input_end")
        status.output_start = metadata_payload.get("output_start")
        status.output_end = metadata_payload.get("output_end")
        status.schema_hash = metadata_payload.get("feature_schema_hash")
        if not isinstance(status.rows, int) or status.rows <= 0:
            status.warnings.append("rows_missing_or_zero")
        columns = metadata_payload.get("columns") or []
        if not columns:
            status.warnings.append("columns_missing")

    parquet_files = list(chunk_dir.glob("*.parquet"))
    if not parquet_files:
        status.missing_parquet = True

    # Perform schema validation if requested
    if validate_schema and schema_validator and parquet_files:
        try:
            parquet_file = chunk_dir / "ml_dataset.parquet"
            if parquet_file.exists():
                validation_result = schema_validator.validate_parquet(parquet_file)
                status.schema_validation_result = validation_result
                status.schema_hash = validation_result.schema_hash

                if not validation_result.is_valid:
                    status.warnings.append("schema_mismatch")
        except Exception as e:
            status.warnings.append(f"schema_validation_error: {e}")

    if status.state and status.state != "completed":
        status.warnings.append(f"state={status.state}")
    if status.error:
        status.warnings.append("error_present")

    return status


def summarize(chunks: Iterable[ChunkStatus], show_schema: bool = False) -> None:
    if show_schema:
        print(f"{'Chunk':<15} {'Rows':>10} {'State':<12} {'Schema':^18} Issues")
        print("-" * 85)
    else:
        print(f"{'Chunk':<15} {'Rows':>10} {'State':<12} Issues")
        print("-" * 60)

    for chunk in chunks:
        issues: list[str] = []
        if chunk.missing_status:
            issues.append("missing_status")
        if chunk.missing_metadata:
            issues.append("missing_metadata")
        if chunk.missing_parquet:
            issues.append("missing_parquet")
        issues.extend(chunk.warnings)
        if chunk.error:
            issues.append("error")
        state_display = chunk.state or "unknown"
        rows_display = chunk.rows if chunk.rows is not None else "-"

        if show_schema:
            schema_display = chunk.schema_hash[:8] if chunk.schema_hash else "N/A"
            schema_status = "✓" if chunk.schema_ok else "✗"
            print(f"{chunk.chunk_id:<15} {rows_display:>10} {state_display:<12} {schema_status} {schema_display:16s} {'; '.join(issues) if issues else ''}")
        else:
            print(f"{chunk.chunk_id:<15} {rows_display:>10} {state_display:<12} {'; '.join(issues) if issues else ''}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect dataset chunk health.")
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=Path("/workspace/gogooku3/gogooku5/data/output/chunks"),
        help="Path to the chunk output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Return exit code 1 if any chunk has warnings or missing artifacts.",
    )
    parser.add_argument(
        "--fail-on-schema",
        action="store_true",
        default=True,
        help="Fail if schema validation fails (default: enabled)",
    )
    parser.add_argument(
        "--no-fail-on-schema",
        action="store_false",
        dest="fail_on_schema",
        help="Disable fail-on-schema check",
    )
    parser.add_argument(
        "--validate-schema",
        action="store_true",
        default=False,
        help="Perform schema validation against manifest",
    )
    parser.add_argument(
        "--schema-manifest",
        type=Path,
        help="Path to schema manifest (default: auto-detect)",
    )
    args = parser.parse_args()

    if not args.chunks_dir.exists():
        print(f"[ERROR] Chunks directory not found: {args.chunks_dir}", file=sys.stderr)
        return 2

    chunk_dirs = sorted(p for p in args.chunks_dir.iterdir() if p.is_dir())
    if not chunk_dirs:
        print(f"[WARN] No chunk directories under {args.chunks_dir}")
        return 0

    # Initialize schema validator if requested
    schema_validator = None
    if args.validate_schema:
        if not SCHEMA_VALIDATION_AVAILABLE:
            print(f"[ERROR] Schema validation not available (import failed)", file=sys.stderr)
            return 2

        try:
            schema_validator = SchemaValidator(args.schema_manifest)
            print(f"[INFO] Using schema manifest: {schema_validator.manifest_path}")
            print(f"[INFO] Expected schema hash: {schema_validator.manifest_hash}")
        except Exception as e:
            print(f"[ERROR] Failed to load schema manifest: {e}", file=sys.stderr)
            return 2

    statuses = [
        collect_chunk_status(chunk_dir, validate_schema=args.validate_schema, schema_validator=schema_validator)
        for chunk_dir in chunk_dirs
    ]

    summarize(statuses, show_schema=args.validate_schema)

    # Check for failures
    has_warnings = any(not chunk.ok for chunk in statuses)
    has_schema_failures = any(not chunk.schema_ok for chunk in statuses)

    if args.fail_on_warning and has_warnings:
        print(f"\n[FAIL] {sum(1 for c in statuses if not c.ok)} chunks have warnings", file=sys.stderr)
        return 1

    if args.fail_on_schema and has_schema_failures:
        print(f"\n[FAIL] {sum(1 for c in statuses if not c.schema_ok)} chunks have schema mismatches", file=sys.stderr)
        # Print detailed mismatch info
        for chunk in statuses:
            if not chunk.schema_ok and chunk.schema_validation_result:
                print(f"\n{chunk.chunk_id}:", file=sys.stderr)
                print(f"  {chunk.schema_validation_result}", file=sys.stderr)
        return 1

    print(f"\n[OK] All {len(statuses)} chunks validated successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
