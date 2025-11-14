"""
Schema validation utilities for dataset chunks.

Provides manifest-driven schema validation so that chunk outputs remain
compatible before merge.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl


@dataclass
class SchemaValidationResult:
    """Result of schema validation."""

    is_valid: bool
    schema_hash: str
    manifest_hash: str
    missing_columns: List[str]
    extra_columns: List[str]
    dtype_mismatches: Dict[str, tuple[str, str]]  # column -> (expected, actual)
    column_count: int
    manifest_column_count: int
    column_order_mismatch: bool = False

    def __str__(self) -> str:
        if self.is_valid:
            prefix = f"✅ Schema valid (hash: {self.schema_hash})"
            if self.column_order_mismatch:
                return f"{prefix} – column order differs from manifest"
            return prefix

        lines = [f"❌ Schema validation failed (hash: {self.schema_hash} != {self.manifest_hash})"]
        if self.missing_columns:
            lines.append(f"   Missing columns ({len(self.missing_columns)}): {self.missing_columns[:5]}")
        if self.extra_columns:
            lines.append(f"   Extra columns ({len(self.extra_columns)}): {self.extra_columns[:5]}")
        if self.dtype_mismatches:
            lines.append(f"   Type mismatches ({len(self.dtype_mismatches)}):")
            for col, (expected, actual) in list(self.dtype_mismatches.items())[:5]:
                lines.append(f"      {col}: expected {expected}, got {actual}")
        if self.column_order_mismatch:
            lines.append("   Column order differs from manifest (informational)")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""

        return {
            "is_valid": self.is_valid,
            "schema_hash": self.schema_hash,
            "manifest_hash": self.manifest_hash,
            "missing_columns": self.missing_columns,
            "extra_columns": self.extra_columns,
            "dtype_mismatches": {
                name: {"expected": exp, "actual": act} for name, (exp, act) in self.dtype_mismatches.items()
            },
            "column_count": self.column_count,
            "manifest_column_count": self.manifest_column_count,
            "column_order_mismatch": self.column_order_mismatch,
        }


class SchemaValidator:
    """Validates dataset schemas against a reference manifest."""

    def __init__(self, manifest_path: Optional[Path] = None):
        if manifest_path is None:
            manifest_path = Path(__file__).parents[3] / "schema" / "feature_schema_manifest.json"

        self.manifest_path = manifest_path
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Schema manifest not found: {self.manifest_path}")

        with self.manifest_path.open(encoding="utf-8") as fh:
            self.manifest = json.load(fh)

        self.manifest_version = self.manifest.get("version", "unknown")
        self.enforce_order = bool(self.manifest.get("column_order_enforced", False))
        self.hash_strategy = self.manifest.get("hash_strategy", "sha256_orderless_v1")
        self.column_order = [col["name"] for col in self.manifest.get("columns", [])]
        self.manifest_hash = self.manifest["schema_hash"]
        self.expected_columns = {col["name"]: col["dtype"] for col in self.manifest.get("columns", [])}

    @staticmethod
    def _compute_hash(
        columns: Dict[str, str],
        *,
        enforce_order: bool,
        column_order: Optional[List[str]] = None,
    ) -> str:
        if enforce_order and column_order:
            ordered = [f"{name}:{columns.get(name, '')}" for name in column_order]
        else:
            ordered = [f"{name}:{dtype}" for name, dtype in sorted(columns.items())]

        return hashlib.sha256(";".join(ordered).encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def compute_hash_for_columns(
        columns: Dict[str, str],
        *,
        enforce_order: bool = False,
        column_order: Optional[List[str]] = None,
    ) -> str:
        """Public helper for computing schema hashes."""

        return SchemaValidator._compute_hash(
            columns,
            enforce_order=enforce_order,
            column_order=column_order,
        )

    def validate_dataframe(self, df: pl.DataFrame) -> SchemaValidationResult:
        """Validate a Polars DataFrame against the manifest schema."""

        actual_columns = {name: str(dtype) for name, dtype in df.schema.items()}
        actual_hash = self._compute_hash(
            actual_columns,
            enforce_order=self.enforce_order,
            column_order=self.column_order,
        )

        missing = [col for col in self.expected_columns if col not in actual_columns]
        extra = [col for col in actual_columns if col not in self.expected_columns]
        dtype_mismatches: Dict[str, tuple[str, str]] = {}
        for col in set(self.expected_columns) & set(actual_columns):
            expected_dtype = self.expected_columns[col]
            actual_dtype = actual_columns[col]
            if expected_dtype != actual_dtype:
                dtype_mismatches[col] = (expected_dtype, actual_dtype)

        column_order_mismatch = False
        if not missing and not extra and not dtype_mismatches and self.column_order:
            column_order_mismatch = list(df.columns) != self.column_order

        is_valid = (
            actual_hash == self.manifest_hash and not missing and not extra and not dtype_mismatches
        )

        return SchemaValidationResult(
            is_valid=is_valid,
            schema_hash=actual_hash,
            manifest_hash=self.manifest_hash,
            missing_columns=missing,
            extra_columns=extra,
            dtype_mismatches=dtype_mismatches,
            column_count=len(actual_columns),
            manifest_column_count=len(self.expected_columns),
            column_order_mismatch=column_order_mismatch,
        )

    def validate_parquet(self, parquet_path: Path) -> SchemaValidationResult:
        """Validate a Parquet file against the manifest schema."""

        df = pl.read_parquet(parquet_path, n_rows=0)
        return self.validate_dataframe(df)

    def validate_chunk(self, chunk_dir: Path) -> tuple[SchemaValidationResult, dict]:
        """
        Validate a chunk directory (ml_dataset.parquet + metadata.json).

        Returns a tuple of (SchemaValidationResult, updated_metadata_dict).
        """

        parquet_file = chunk_dir / "ml_dataset.parquet"
        metadata_file = chunk_dir / "metadata.json"

        if not parquet_file.exists():
            raise FileNotFoundError(f"Chunk parquet not found: {parquet_file}")

        result = self.validate_parquet(parquet_file)
        metadata: dict = {}
        if metadata_file.exists():
            with metadata_file.open(encoding="utf-8") as fh:
                metadata = json.load(fh)

        metadata["feature_schema_version"] = self.manifest_version
        metadata["feature_schema_hash"] = result.schema_hash
        metadata["schema_validation"] = result.to_dict()

        return result, metadata


def validate_chunks_directory(
    chunks_dir: Path,
    manifest_path: Optional[Path] = None,
    fail_fast: bool = False,
) -> dict[str, SchemaValidationResult]:
    """Validate all chunks in a directory."""

    validator = SchemaValidator(manifest_path)
    results: dict[str, SchemaValidationResult] = {}

    for chunk_path in sorted(chunks_dir.iterdir()):
        if not chunk_path.is_dir():
            continue

        parquet_file = chunk_path / "ml_dataset.parquet"
        if not parquet_file.exists():
            continue

        chunk_id = chunk_path.name
        try:
            result = validator.validate_parquet(parquet_file)
            results[chunk_id] = result
            if fail_fast and not result.is_valid:
                raise ValueError(f"Chunk {chunk_id} schema validation failed:\n{result}")
        except Exception as exc:
            if fail_fast:
                raise
            print(f"⚠️  Error validating {chunk_id}: {exc}")

    return results
