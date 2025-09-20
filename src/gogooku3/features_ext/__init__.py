"""Extended feature engineering utilities (non-breaking additions).

Modules here must:
- Avoid deleting existing columns; only add or transform in-place when explicitly configured.
- Be leak-safe when used via the training datamodule (fit on train fold, transform on OOS).

All functions are typed and Polars-first.
"""

from __future__ import annotations

__all__ = [
    "sector_loo",
    "scale_unify",
    "outliers",
    "interactions",
    "cs_standardize",
    "cache_utils",
]

