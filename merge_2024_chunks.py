#!/usr/bin/env python3
"""Convenience wrapper that merges 2024Q1+2024Q2 using the lazy pipeline."""

from __future__ import annotations

from pathlib import Path

from merge_2024_full import (
    DEFAULT_CHUNK_ROOT,
    DEFAULT_HORIZONS,
    merge_chunks,
)

DEFAULT_CHUNK_IDS = ("2024Q1", "2024Q2")
DEFAULT_OUTPUT = Path("output/ml_dataset_2024H1_merged_final.parquet")
DEFAULT_SYMLINK = Path("output/ml_dataset_2024H1_latest_full.parquet")


def main() -> None:
    merge_chunks(
        chunk_root=DEFAULT_CHUNK_ROOT,
        chunk_ids=DEFAULT_CHUNK_IDS,
        output_path=DEFAULT_OUTPUT,
        symlink_path=DEFAULT_SYMLINK,
        horizons=DEFAULT_HORIZONS,
    )


if __name__ == "__main__":
    main()
