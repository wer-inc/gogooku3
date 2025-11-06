"""Feature engineering utilities."""

from .adv import apply_adv_filter, compute_adv60_from_raw, get_raw_quotes_paths
from .asof_join import add_asof_timestamp, interval_join_pl, prepare_snapshot_pl
from .lazy_io import lazy_load, save_with_cache
from .rolling import (
    ewm_mean_safe,
    roll_mean_safe,
    roll_quantile_safe,
    roll_std_safe,
    roll_sum_safe,
    roll_var_safe,
)
from .schema import (
    canonicalize_ohlc,
    enforce_unique_columns,
    ensure_sector_dimensions,
    safe_rename,
    validate_unique_columns,
)

__all__ = [
    # Rolling statistics (Patch C)
    "roll_mean_safe",
    "roll_std_safe",
    "roll_var_safe",
    "roll_sum_safe",
    "roll_quantile_safe",
    "ewm_mean_safe",
    # As-of joins (Patch D)
    "add_asof_timestamp",
    "prepare_snapshot_pl",
    "interval_join_pl",
    # Lazy I/O (Quick Wins Task 1)
    "lazy_load",
    "save_with_cache",
    # ADV filter (Patch E)
    "compute_adv60_from_raw",
    "apply_adv_filter",
    "get_raw_quotes_paths",
    # Schema normalization (BATCH-2B)
    "canonicalize_ohlc",
    "enforce_unique_columns",
    "ensure_sector_dimensions",
    "safe_rename",
    "validate_unique_columns",
]
