#!/usr/bin/env python3
"""Remove stale cache files that contain zero rows.

Scans the cache directory (default: data/output/cache) for Parquet/IPC files
and deletes entries that contain no data. Use --dry-run to see what would be
removed without deleting anything.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl


def detect_empty(file_path: Path) -> bool:
    try:
        if file_path.suffix == ".parquet":
            df = pl.read_parquet(file_path, n_rows=1)
        elif file_path.suffix == ".arrow":
            df = pl.read_ipc(file_path, n_rows=1)
        else:
            return False
        return df.height == 0
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cache_dir",
        nargs="?",
        default="gogooku5/data/output/cache",
        help="Directory to scan (default: gogooku5/data/output/cache)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report empty files without deleting",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        print(f"Cache directory {cache_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    empty_files: list[Path] = []
    for path in cache_dir.rglob("*.parquet"):
        if detect_empty(path):
            empty_files.append(path)
    for path in cache_dir.rglob("*.arrow"):
        if detect_empty(path):
            empty_files.append(path)

    if not empty_files:
        print("No empty cache files found.")
        return

    for path in sorted(empty_files):
        if args.dry_run:
            print(f"DRY RUN: would remove {path}")
        else:
            path.unlink(missing_ok=True)
            print(f"Removed {path}")
            sibling = path.with_suffix(".arrow" if path.suffix == ".parquet" else ".parquet")
            if sibling.exists() and detect_empty(sibling):
                sibling.unlink(missing_ok=True)
                print(f"Removed sibling {sibling}")


if __name__ == "__main__":
    main()
