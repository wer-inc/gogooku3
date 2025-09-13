from __future__ import annotations

"""
Public Builder facade for dataset enrichment.

This module re-exports MLDatasetBuilder from the legacy location under
scripts/data to provide a stable import path for pipelines and library code.
Refactors can evolve the implementation behind this facade without changing
call sites.
"""

from scripts.data.ml_dataset_builder import (  # type: ignore F401
    MLDatasetBuilder,
    create_sample_data,
)

__all__ = ["MLDatasetBuilder", "create_sample_data"]

