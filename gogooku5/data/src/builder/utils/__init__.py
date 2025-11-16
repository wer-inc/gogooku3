"""Shared utilities (logging, caching, environment helpers)."""

from .artifacts import DatasetArtifact, DatasetArtifactWriter, resolve_latest_dataset
from .asyncio import gather_limited, run_sync
from .asof import (
    add_asof_timestamp,
    forward_fill_after_publication,
    interval_join_pl,
    prepare_snapshot_pl,
)
from .cache import CacheManager, ensure_cache_dir
from .datetime import business_date_range, date_range, shift_trading_days
from .env import ensure_env_loaded, load_local_env, require_env_var
from .git_metadata import get_git_metadata
from .logger import configure_logger, get_logger
from .raw_snapshot import save_raw_snapshot
from .raw_store import RawDataStore
from .mlflow_tracker import MLflowTracker
from .storage import StorageClient

# GPU-ETL utilities (optional, may not be present in clean checkout)
try:
    from .gpu_utils import (
        GPU_AVAILABLE,
        apply_gpu_transform,
        cudf_to_pl,
        gpu_rolling_mean,
        gpu_rolling_std,
        init_rmm,
        pl_to_cudf,
    )

    _GPU_AVAILABLE = True
except ImportError:
    # Fallback for clean checkout where gpu_utils is not present
    GPU_AVAILABLE = False

    def apply_gpu_transform(*args, **kwargs):
        raise RuntimeError("GPU utilities not available (gpu_utils module not found)")

    def cudf_to_pl(*args, **kwargs):
        raise RuntimeError("GPU utilities not available (gpu_utils module not found)")

    def gpu_rolling_mean(*args, **kwargs):
        raise RuntimeError("GPU utilities not available (gpu_utils module not found)")

    def gpu_rolling_std(*args, **kwargs):
        raise RuntimeError("GPU utilities not available (gpu_utils module not found)")

    def init_rmm(*args, **kwargs):
        raise RuntimeError("GPU utilities not available (gpu_utils module not found)")

    def pl_to_cudf(*args, **kwargs):
        raise RuntimeError("GPU utilities not available (gpu_utils module not found)")

    _GPU_AVAILABLE = False

# Quote shard utilities (optional, may not be present in clean checkout)
try:
    from .quotes_l0 import (
        QuoteShard,
        QuoteShardIndex,
        QuoteShardStore,
        month_key,
        month_range,
    )

    _QUOTES_L0_AVAILABLE = True
except ImportError:
    # Fallback for clean checkout where quotes_l0 is not present
    # Define minimal stubs to avoid breaking imports
    class QuoteShard:
        """Placeholder for QuoteShard when quotes_l0 module is not available."""

        pass

    class QuoteShardIndex:
        """Placeholder for QuoteShardIndex when quotes_l0 module is not available."""

        pass

    class QuoteShardStore:
        """Placeholder for QuoteShardStore when quotes_l0 module is not available."""

        pass

    def month_key(*args, **kwargs):
        raise RuntimeError("Quote shard utilities not available (quotes_l0 module not found)")

    def month_range(*args, **kwargs):
        raise RuntimeError("Quote shard utilities not available (quotes_l0 module not found)")

    _QUOTES_L0_AVAILABLE = False

__all__ = [
    "CacheManager",
    "DatasetArtifact",
    "DatasetArtifactWriter",
    "StorageClient",
    "configure_logger",
    "business_date_range",
    "date_range",
    "shift_trading_days",
    "add_asof_timestamp",
    "prepare_snapshot_pl",
    "interval_join_pl",
    "forward_fill_after_publication",
    "get_git_metadata",
    "save_raw_snapshot",
    "ensure_cache_dir",
    "ensure_env_loaded",
    "gather_limited",
    "get_logger",
    "load_local_env",
    "require_env_var",
    "resolve_latest_dataset",
    "run_sync",
    "QuoteShard",
    "QuoteShardIndex",
    "QuoteShardStore",
    "month_key",
    "month_range",
    # GPU-ETL utilities (may be stubs if module not available)
    "GPU_AVAILABLE",
    "init_rmm",
    "pl_to_cudf",
    "cudf_to_pl",
    "apply_gpu_transform",
    "gpu_rolling_mean",
    "gpu_rolling_std",
    "MLflowTracker",
]
