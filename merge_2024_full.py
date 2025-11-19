#!/usr/bin/env python3
"""
Merge multiple gogooku5 chunk parquet files using Polars Lazy pipelines.

The script standardises schemas, concatenates the chunks lazily, appends forward
return targets, and stores the result via streaming `sink_parquet`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import polars as pl

from gogooku5.data.tools.lazy_dataset_utils import (
    add_future_return_columns,
    dataset_summary,
    drop_existing_target_columns,
    format_coverage,
    scan_parquet_lazy,
)

DEFAULT_CHUNK_ROOT = Path("output/chunks")
DEFAULT_CHUNK_IDS = ("2024Q1", "2024Q2", "2024Q3", "2024Q4")
DEFAULT_OUTPUT = Path("output/ml_dataset_2024_full.parquet")
DEFAULT_SYMLINK = Path("output/ml_dataset_2024_latest.parquet")
DEFAULT_HORIZONS = (1, 5, 10, 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge gogooku5 chunk parquet files lazily.")
    parser.add_argument(
        "--chunk-root",
        type=Path,
        default=DEFAULT_CHUNK_ROOT,
        help="Directory that contains <chunk_id>/ml_dataset.parquet files.",
    )
    parser.add_argument(
        "--chunk-ids",
        nargs="+",
        default=list(DEFAULT_CHUNK_IDS),
        help="Chunk IDs (e.g., 2024Q1) merged in the provided order.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to the merged parquet.",
    )
    parser.add_argument(
        "--symlink",
        type=Path,
        default=DEFAULT_SYMLINK,
        help="Optional symlink updated after a successful merge.",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=list(DEFAULT_HORIZONS),
        help="Forward-return horizons (in trading days).",
    )
    parser.add_argument("--compression", default="zstd", help="Parquet compression codec.")
    parser.add_argument("--compression-level", type=int, default=3, help="Parquet compression level.")
    return parser.parse_args()


def inspect_chunk(scan: pl.LazyFrame) -> dict:
    """Return lightweight stats for logging."""

    schema = scan.collect_schema()
    stats = (
        scan.clone()
            .select(
                [
                    pl.len().alias("rows"),
                    pl.col("Date").min().alias("date_start"),
                    pl.col("Date").max().alias("date_end"),
                ]
            )
            .collect(streaming=True)
            .to_dicts()[0]
    )
    stats["columns"] = len(schema)
    return stats, schema


def standardise_chunk(
    scan: pl.LazyFrame,
    schema,
    columns: Sequence[str],
    reference_types: dict[str, object],
) -> pl.LazyFrame:
    """Select the intersection of columns using a shared dtype map."""

    exprs = []
    for name in columns:
        expr = pl.col(name)
        ref_type = reference_types[name]
        if schema[name] != ref_type:
            expr = expr.cast(ref_type)
        exprs.append(expr.alias(name))
    return scan.clone().select(exprs)


def update_symlink(target: Path, link_path: Path | None) -> None:
    if not link_path:
        return
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    # Use relative path for symlink to work across different directories
    import os
    if link_path.parent == target.parent:
        link_path.symlink_to(target.name)
    else:
        link_path.symlink_to(os.path.relpath(target, start=link_path.parent))


def merge_chunks(
    chunk_root: Path,
    chunk_ids: Sequence[str],
    output_path: Path,
    symlink_path: Path | None,
    horizons: Sequence[int],
    *,
    compression: str = "zstd",
    compression_level: int = 3,
) -> None:
    if not chunk_ids:
        raise ValueError("No chunk IDs supplied.")

    chunk_scans: list[pl.LazyFrame] = []
    chunk_schemas: list = []
    chunk_stats: list[dict] = []
    chunk_paths: list[Path] = []

    print("📦 Inspecting chunks:")
    for chunk_id in chunk_ids:
        path = chunk_root / chunk_id / "ml_dataset.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing chunk: {path}")
        scan = scan_parquet_lazy(path)
        stats, schema = inspect_chunk(scan)
        print(
            f"  • {chunk_id}: {stats['rows']:,} rows × {stats['columns']} cols "
            f"({stats['date_start']} → {stats['date_end']})"
        )
        chunk_paths.append(path)
        chunk_scans.append(scan)
        chunk_schemas.append(schema)
        chunk_stats.append(stats)

    # Determine common schema intersection.
    column_sets = [set(schema.names()) for schema in chunk_schemas]
    common_cols = sorted(set.intersection(*column_sets))
    if not common_cols:
        raise RuntimeError("No common columns across chunks.")

    reference_types = {col: chunk_schemas[0][col] for col in common_cols}
    standardized_frames = [
        standardise_chunk(scan, schema, common_cols, reference_types)
        for scan, schema in zip(chunk_scans, chunk_schemas)
    ]

    merged = pl.concat(standardized_frames, how="vertical", rechunk=True)
    merged, removed_targets = drop_existing_target_columns(merged)
    if removed_targets:
        print(f"⚠️  Dropping pre-existing targets: {removed_targets}")
    merged = add_future_return_columns(merged, horizons)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n💾 Writing merged parquet → {output_path}")
    merged.sink_parquet(
        str(output_path),
        compression=compression,
        compression_level=compression_level,
        maintain_order=True,
    )
    print("✅ Merge complete.")

    summary = dataset_summary(output_path, horizons)
    print(
        f"\n📊 Final dataset: {summary['rows']:,} rows × {summary['columns']} cols, "
        f"{summary['date_start']} → {summary['date_end']}, {summary['codes']:,} codes"
    )
    print("📈 Target coverage:")
    for line in format_coverage(summary, horizons):
        print(line)

    update_symlink(output_path, symlink_path)
    if symlink_path:
        print(f"\n🔗 Updated symlink: {symlink_path} → {output_path.name}")


def main() -> None:
    args = parse_args()
    horizons: Sequence[int] = tuple(dict.fromkeys(args.horizons))
    merge_chunks(
        chunk_root=args.chunk_root,
        chunk_ids=args.chunk_ids,
        output_path=args.output,
        symlink_path=args.symlink,
        horizons=horizons,
        compression=args.compression,
        compression_level=args.compression_level,
    )


if __name__ == "__main__":
    main()
