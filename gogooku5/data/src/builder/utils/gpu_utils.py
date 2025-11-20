"""GPU-ETL utilities for gogooku5 dataset builder.

Thin wrapper around gogooku3's gpu_etl module to enable GPU-accelerated
feature engineering with cuDF/RAPIDS.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    import cudf

# Add gogooku3/src to path to import gpu_etl
_GOGOOKU3_SRC = Path(__file__).parents[6] / "src"
if str(_GOGOOKU3_SRC) not in sys.path:
    sys.path.insert(0, str(_GOGOOKU3_SRC))

try:
    from utils.gpu_etl import _has_cuda
    from utils.gpu_etl import init_rmm_legacy as init_rmm
    from utils.gpu_etl import to_cudf as pl_to_cudf
    from utils.gpu_etl import to_polars as cudf_to_pl

    GPU_AVAILABLE = _has_cuda()
except ImportError:
    GPU_AVAILABLE = False

    def _has_cuda() -> bool:  # type: ignore[misc]
        return False

    def init_rmm(pool_size: str | None = None) -> bool:  # type: ignore[misc]
        return False

    def pl_to_cudf(df: pl.DataFrame) -> cudf.DataFrame | None:  # type: ignore[misc]
        return None

    def cudf_to_pl(gdf: cudf.DataFrame) -> pl.DataFrame | None:  # type: ignore[misc]
        return None


LOGGER = logging.getLogger(__name__)


def apply_gpu_transform(
    df: pl.DataFrame,
    transform_fn: callable,  # type: ignore[valid-type]
    fallback_fn: callable,  # type: ignore[valid-type]
    operation_name: str = "GPU transform",
) -> pl.DataFrame:
    """Apply GPU-accelerated transformation with CPU fallback.

    Args:
        df: Input Polars DataFrame
        transform_fn: Function that accepts cuDF DataFrame and returns cuDF DataFrame
        fallback_fn: CPU fallback function that accepts/returns Polars DataFrame
        operation_name: Operation name for logging

    Returns:
        Transformed Polars DataFrame
    """
    if not GPU_AVAILABLE:
        LOGGER.debug("%s: GPU not available, using CPU fallback", operation_name)
        return fallback_fn(df)

    try:
        # Convert to cuDF
        gdf = pl_to_cudf(df)
        if gdf is None:
            LOGGER.debug("%s: Polars→cuDF conversion failed, using CPU fallback", operation_name)
            return fallback_fn(df)

        # Apply GPU transform
        result_gdf = transform_fn(gdf)

        # Convert back to Polars
        result_pl = cudf_to_pl(result_gdf)
        if result_pl is None:
            LOGGER.warning("%s: cuDF→Polars conversion failed, using CPU fallback", operation_name)
            return fallback_fn(df)

        LOGGER.info("✅ %s: GPU acceleration successful (%d rows)", operation_name, len(result_pl))
        return result_pl

    except Exception as e:
        LOGGER.warning("%s: GPU processing failed (%s), using CPU fallback", operation_name, e)
        return fallback_fn(df)


def gpu_rolling_mean(
    df: pl.DataFrame,
    value_col: str,
    window: int,
    group_col: str | None = None,
) -> pl.DataFrame:
    """Compute rolling mean on GPU with CPU fallback.

    Args:
        df: Input DataFrame
        value_col: Column to compute rolling mean on
        window: Rolling window size
        group_col: Optional grouping column (e.g., 'code')

    Returns:
        DataFrame with added rolling mean column
    """

    def gpu_fn(gdf):  # type: ignore[no-untyped-def]
        if group_col:
            result = gdf.groupby(group_col)[value_col].rolling(window).mean()
        else:
            result = gdf[value_col].rolling(window).mean()

        gdf[f"{value_col}_roll_mean_{window}d"] = result
        return gdf

    def cpu_fn(df: pl.DataFrame) -> pl.DataFrame:
        if group_col:
            return df.with_columns(
                pl.col(value_col)
                .rolling_mean(window_size=window)
                .over(group_col)
                .alias(f"{value_col}_roll_mean_{window}d")
            )
        else:
            return df.with_columns(
                pl.col(value_col).rolling_mean(window_size=window).alias(f"{value_col}_roll_mean_{window}d")
            )

    return apply_gpu_transform(df, gpu_fn, cpu_fn, f"rolling_mean({value_col}, {window})")


def gpu_rolling_std(
    df: pl.DataFrame,
    value_col: str,
    window: int,
    group_col: str | None = None,
) -> pl.DataFrame:
    """Compute rolling std on GPU with CPU fallback.

    Args:
        df: Input DataFrame
        value_col: Column to compute rolling std on
        window: Rolling window size
        group_col: Optional grouping column (e.g., 'code')

    Returns:
        DataFrame with added rolling std column
    """

    def gpu_fn(gdf):  # type: ignore[no-untyped-def]
        if group_col:
            result = gdf.groupby(group_col)[value_col].rolling(window).std()
        else:
            result = gdf[value_col].rolling(window).std()

        gdf[f"{value_col}_roll_std_{window}d"] = result
        return gdf

    def cpu_fn(df: pl.DataFrame) -> pl.DataFrame:
        if group_col:
            return df.with_columns(
                pl.col(value_col)
                .rolling_std(window_size=window)
                .over(group_col)
                .alias(f"{value_col}_roll_std_{window}d")
            )
        else:
            return df.with_columns(
                pl.col(value_col).rolling_std(window_size=window).alias(f"{value_col}_roll_std_{window}d")
            )

    return apply_gpu_transform(df, gpu_fn, cpu_fn, f"rolling_std({value_col}, {window})")


__all__ = [
    "GPU_AVAILABLE",
    "init_rmm",
    "pl_to_cudf",
    "cudf_to_pl",
    "apply_gpu_transform",
    "gpu_rolling_mean",
    "gpu_rolling_std",
]
