#!/usr/bin/env python3
"""Regenerate the feature schema manifest from a reference chunk."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "gogooku5" / "data" / "src"))

from builder.utils.schema_validator import SchemaValidator  # noqa: E402


def build_manifest(parquet_path: Path, version: str, column_order_enforced: bool) -> dict:
    df = pl.read_parquet(parquet_path, n_rows=0)
    columns = [
        {
            "name": name,
            "dtype": str(dtype),
            "nullable": True,
        }
        for name, dtype in df.schema.items()
    ]
    column_map = {col["name"]: col["dtype"] for col in columns}
    schema_hash = SchemaValidator.compute_hash_for_columns(
        column_map,
        enforce_order=column_order_enforced,
        column_order=[col["name"] for col in columns] if column_order_enforced else None,
    )
    return {
        "version": version,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
        "reference_file": str(parquet_path),
        "hash_strategy": "sha256_orderless_v1" if not column_order_enforced else "sha256_ordered_v1",
        "schema_hash": schema_hash,
        "column_order_enforced": column_order_enforced,
        "total_columns": len(columns),
        "columns": columns,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate feature_schema_manifest.json")
    default_ref = Path(__file__).resolve().parents[1] / "output" / "chunks" / "2020Q1" / "ml_dataset.parquet"
    parser.add_argument(
        "--reference-chunk",
        type=Path,
        default=default_ref,
        help="Path to the reference chunk Parquet file",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="1.1.0",
        help="Schema manifest version (semantic versioning)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "gogooku5" / "data" / "schema" / "feature_schema_manifest.json",
        help="Destination manifest path",
    )
    parser.add_argument(
        "--enforce-order",
        action="store_true",
        help="Keep column order as part of the hash (default: order-agnostic)",
    )
    args = parser.parse_args()

    if not args.reference_chunk.exists():
        raise FileNotFoundError(f"Reference chunk not found: {args.reference_chunk}")

    manifest = build_manifest(args.reference_chunk, version=args.version, column_order_enforced=args.enforce_order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"Wrote schema manifest v{args.version} ({manifest['total_columns']} columns, "
        f"hash={manifest['schema_hash']}) -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
