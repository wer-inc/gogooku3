"""Compatibility shim for legacy imports.

Phase 2 as-of utilities now live under :mod:`builder.utils.asof`. This module
re-exports the public helpers so existing imports continue to work.
"""
from builder.utils.asof import (
    add_asof_timestamp,
    forward_fill_after_publication,
    interval_join_pl,
    prepare_snapshot_pl,
)

__all__ = [
    "add_asof_timestamp",
    "prepare_snapshot_pl",
    "interval_join_pl",
    "forward_fill_after_publication",
]
