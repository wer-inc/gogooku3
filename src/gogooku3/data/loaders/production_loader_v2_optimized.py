from __future__ import annotations

"""
Compatibility shim for ProductionDatasetOptimized.

This is a minimal placeholder to satisfy imports; the modern pipeline reads
parquet directly via Polars and does not rely on this class.
"""

from collections.abc import Iterable
from pathlib import Path

import polars as pl


class ProductionDatasetOptimized:
    def __init__(self):
        pass

    def load(self, files: Iterable[Path]) -> pl.DataFrame:
        frames = [pl.read_parquet(str(p)) for p in files]
        return pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame()

