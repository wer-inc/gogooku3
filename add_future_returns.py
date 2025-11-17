#!/usr/bin/env python3
"""
Add forward-return targets to an existing merged dataset using Polars Lazy pipelines.

This script avoids loading the entire parquet into memory by relying on `scan_parquet`,
column projection, and streaming `sink_parquet` writes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from gogooku5.data.tools.lazy_dataset_utils import (
    add_future_return_columns,
    dataset_summary,
    drop_existing_target_columns,
    format_coverage,
    scan_parquet_lazy,
)

DEFAULT_INPUT = "output/ml_dataset_2024H1_20251110_224138_full.parquet"
DEFAULT_OUTPUT = "output/ml_dataset_2024H1_20251110_224138_full_with_targets.parquet"
DEFAULT_SYMLINK = "output/ml_dataset_2024H1_latest_full.parquet"
DEFAULT_HORIZONS = (1, 5, 10, 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append target_* columns using lazy Polars transforms.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Source parquet produced by gogooku5.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Destination parquet with targets.")
    parser.add_argument("--symlink", default=DEFAULT_SYMLINK, help="Symlink updated to point at the new parquet.")
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=list(DEFAULT_HORIZONS),
        help="Forward horizons (in trading days) to materialize.",
    )
    parser.add_argument("--compression", default="zstd", help="Parquet compression codec.")
    parser.add_argument("--compression-level", type=int, default=3, help="Parquet compression level.")
    return parser.parse_args()


def update_symlink(target: Path, link_path: Path) -> None:
    if not link_path:
        return
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    # Use relative path for symlink to work across different directories
    import os
    if link_path.parent == target.parent:
        link_path.symlink_to(target.name)
    else:
        link_path.symlink_to(os.path.relpath(target, start=link_path.parent))


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    symlink_path = Path(args.symlink).resolve() if args.symlink else None
    horizons: Sequence[int] = tuple(dict.fromkeys(args.horizons))  # preserve order, drop duplicates

    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    print(f"📥 Loading (lazy) from: {input_path}")
    print(f"🎯 Horizons: {horizons}")

    lf = scan_parquet_lazy(input_path)
    lf, removed = drop_existing_target_columns(lf)
    if removed:
        print(f"⚠️  Removing existing target columns: {removed}")

    enriched = add_future_return_columns(lf, horizons)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"💾 Writing (streaming) to: {output_path}")
    enriched.sink_parquet(
        str(output_path),
        compression=args.compression,
        compression_level=args.compression_level,
        maintain_order=True,
    )
    print("✅ Target columns materialised.")

    summary = dataset_summary(output_path, horizons)
    print(
        f"\n📊 Dataset summary: {summary['rows']:,} rows × {summary['columns']} cols, "
        f"{summary['date_start']} → {summary['date_end']}, {summary['codes']:,} codes"
    )
    print("📈 Coverage:")
    for line in format_coverage(summary, horizons):
        print(line)

    if symlink_path:
        update_symlink(output_path, symlink_path)
        print(f"\n🔗 Updated symlink: {symlink_path} → {output_path.name}")


if __name__ == "__main__":
    main()
