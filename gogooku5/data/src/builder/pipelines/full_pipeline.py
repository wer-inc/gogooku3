"""Standard dataset build pipeline."""

from __future__ import annotations

from .dataset_builder import DatasetBuilder


def run_full_pipeline(*, start: str, end: str, refresh_listed: bool = False) -> None:
    builder = DatasetBuilder()
    builder.build(start=start, end=end, refresh_listed=refresh_listed)
