"""Optimized dataset pipeline placeholder."""
from __future__ import annotations

from .dataset_builder import DatasetBuilder


def run_optimized_pipeline(*, start: str, end: str, cache_only: bool = False) -> None:
    builder = DatasetBuilder()
    if cache_only:
        builder.cache.invalidate()
    builder.build(start=start, end=end)
