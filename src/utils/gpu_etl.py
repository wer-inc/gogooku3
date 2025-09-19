from __future__ import annotations

"""
GPU-ETL utilities (optional, safe fallback to CPU).

Provides thin wrappers to:
- Initialize RMM pooled allocator to make the most of GPU memory
- Convert between Polars <-> cuDF
- Compute cross-sectional rank/z-score on GPU (by given group keys)

If RAPIDS/cuDF or a CUDA device is not available, every function falls back to
CPU/Polars and preserves the original outputs/column names to avoid regressions.
"""

from typing import Iterable, List, Tuple
import logging
import os

import polars as pl


def _has_cuda() -> bool:
    try:
        import numba.cuda as _cuda  # type: ignore

        return _cuda.is_available()
    except Exception:
        try:
            import cupy as _cp  # type: ignore

            return _cp.cuda.runtime.getDeviceCount() > 0  # type: ignore
        except Exception:
            return False


def _attach_cupy_allocator(cp, rmm, logger: logging.Logger) -> bool:  # type: ignore[override]
    """Attach the RMM allocator to CuPy, supporting both legacy and new APIs."""
    allocator = getattr(rmm, "rmm_cupy_allocator", None)
    if allocator is not None:
        try:
            cp.cuda.set_allocator(allocator)
            return True
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug(f"Failed to attach legacy CuPy allocator: {exc}")

    try:
        from rmm.allocators.cupy import rmm_cupy_allocator  # type: ignore

        cp.cuda.set_allocator(rmm_cupy_allocator)
        return True
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.debug(f"Failed to attach new CuPy allocator: {exc}")
        return False


def init_rmm(initial_pool_size: str | None = None) -> bool:
    """Initialize RMM memory pool (best-effort).

    Args:
        initial_pool_size: e.g., "70GB"; if None, read env RMM_POOL_SIZE

    Returns:
        True if RMM initialized, else False (safe to ignore)
    """
    logger = logging.getLogger(__name__)

    try:
        import rmm  # type: ignore
        import cupy as cp  # type: ignore

        size_str = initial_pool_size or os.getenv("RMM_POOL_SIZE", "")
        kwargs: dict = {"pool_allocator": True}

        if size_str:
            # Convert human-readable size to bytes
            size_bytes = None
            size_str = size_str.upper().strip()

            if size_str.endswith("GB"):
                size_bytes = int(float(size_str[:-2]) * 1024 ** 3)
            elif size_str.endswith("G"):
                size_bytes = int(float(size_str[:-1]) * 1024 ** 3)
            elif size_str.endswith("MB"):
                size_bytes = int(float(size_str[:-2]) * 1024 ** 2)
            elif size_str.endswith("M"):
                size_bytes = int(float(size_str[:-1]) * 1024 ** 2)
            else:
                # Assume it's already in bytes or a plain number
                try:
                    size_bytes = int(size_str)
                except ValueError:
                    pass

            if size_bytes:
                kwargs["initial_pool_size"] = size_bytes

        # Managed memory can be problematic on some systems, make it optional
        if os.getenv("RMM_MANAGED_MEMORY", "0") == "1":
            kwargs["managed_memory"] = True

        rmm.reinitialize(**kwargs)
        attached = _attach_cupy_allocator(cp, rmm, logger)
        if not attached:
            logger.warning("RMM initialized but CuPy allocator attachment failed; GPU ETL will use default CuPy allocator")
        return True
    except Exception as e:
        logger.warning(f"RMM init failed: {e}")
        # Try simpler initialization without managed memory
        try:
            import rmm
            import cupy as cp
            rmm.reinitialize(pool_allocator=False)
            attached = _attach_cupy_allocator(cp, rmm, logger)
            if attached:
                logger.info("RMM initialized without pool allocator")
            else:
                logger.info("RMM initialized without pool allocator (CuPy allocator unavailable)")
            return True
        except Exception as e2:
            logger.debug(f"Fallback RMM init also failed: {e2}")
        return False


def to_cudf(df: pl.DataFrame):  # type: ignore[override]
    """Convert Polars -> cuDF (best-effort). Falls back by returning None."""
    try:
        import cudf  # type: ignore

        tbl = df.to_arrow()
        return cudf.DataFrame.from_arrow(tbl)
    except Exception:
        return None


def to_polars(gdf):  # type: ignore[override]
    """Convert cuDF -> Polars (best-effort). Returns None on failure."""
    try:
        import cudf  # type: ignore

        if not isinstance(gdf, cudf.DataFrame):
            return None
        tbl = gdf.to_arrow()
        return pl.from_arrow(tbl)
    except Exception:
        return None


def cs_rank_and_z(
    df: pl.DataFrame,
    *,
    rank_col: str,
    z_col: str,
    group_keys: Iterable[str] = ("Date",),
    out_rank_name: str = "rank_ret_1d",
    out_z_name: str = "volume_cs_z",
) -> pl.DataFrame:
    """Compute cross-sectional rank and z-score on GPU when possible.

    - Rank is computed with method='average' within each group
    - Z-score is (x - mean) / std within each group (std==0 → 0)

    Falls back to Polars implementation if GPU/cuDF is unavailable.
    Preserves input row order.
    """

    # Fast path: try GPU/cuDF
    if _has_cuda():
        try:
            init_rmm(None)
            gdf = to_cudf(
                df.with_row_count("__rid__").with_columns(
                    [
                        pl.col(rank_col).cast(pl.Float64),
                        pl.col(z_col).cast(pl.Float64),
                    ]
                )
            )
            if gdf is not None:
                import cudf  # type: ignore

                keys = list(group_keys)
                # Rank within groups (average ties), NA kept
                gdf[out_rank_name] = (
                    gdf.groupby(keys)[rank_col]
                    .rank(method="average", ascending=True)
                    .astype("float64")
                )
                # Normalize rank to [0,1] similar to CPU実装
                # We need group sizes; compute via groupby size then merge
                sizes = gdf.groupby(keys).size().rename(columns={"size": "__gsize__"})
                gdf = gdf.merge(sizes, on=keys, how="left")
                gdf[out_rank_name] = (gdf[out_rank_name] - 1.0) / (gdf["__gsize__"] - 1.0)
                gdf[out_rank_name] = gdf[out_rank_name].fillna(0.5)

                # Z-score per group
                grp = gdf.groupby(keys)[z_col]
                gmean = grp.mean().rename(columns={z_col: "__gmean__"})
                gstd = grp.std().rename(columns={z_col: "__gstd__"})
                gdf = gdf.merge(gmean, on=keys, how="left")
                gdf = gdf.merge(gstd, on=keys, how="left")
                # (x - mean) / std with epsilon guard
                eps = 1e-12
                gdf[out_z_name] = (gdf[z_col] - gdf["__gmean__"]) / (gdf["__gstd__"] + eps)
                gdf[out_z_name] = gdf[out_z_name].fillna(0.0)

                out = to_polars(
                    gdf[["__rid__", out_rank_name, out_z_name]]
                )
                if out is not None:
                    out = out.sort("__rid__").drop("__rid__")
                    return df.hstack([out])
        except Exception:
            # Fall through to CPU
            pass

    # CPU/Polars fallback — mirror the existing implementation semantics
    out = df
    if rank_col in out.columns:
        cnt = pl.count().over(list(group_keys))
        rk = pl.col(rank_col).rank(method="average").over(list(group_keys))
        out = out.with_columns(
            [pl.when(cnt > 1).then((rk - 1.0) / (cnt - 1.0)).otherwise(0.5).alias(out_rank_name)]
        )
    if z_col in out.columns:
        out = out.with_columns(
            [
                (
                    (pl.col(z_col) - pl.col(z_col).mean().over(list(group_keys)))
                    / (pl.col(z_col).std().over(list(group_keys)) + 1e-12)
                ).alias(out_z_name)
            ]
        )
    return out


__all__ = [
    "init_rmm",
    "to_cudf",
    "to_polars",
    "cs_rank_and_z",
]


def cs_z(
    df: pl.DataFrame,
    *,
    value_col: str,
    group_keys: Iterable[str] = ("Date",),
    out_name: str = "_cs_z",
) -> pl.DataFrame:
    """Compute cross-sectional z-score for a single column on GPU if available.

    Falls back to Polars when GPU/cuDF is unavailable. Preserves row order.
    """
    # Try GPU path
    if _has_cuda():
        try:
            init_rmm(None)
            gdf = to_cudf(df.with_row_count("__rid__").with_columns(pl.col(value_col).cast(pl.Float64)))
            if gdf is not None:
                keys = list(group_keys)
                grp = gdf.groupby(keys)[value_col]
                gmean = grp.mean().rename(columns={value_col: "__gmean__"})
                gstd = grp.std().rename(columns={value_col: "__gstd__"})
                gdf = gdf.merge(gmean, on=keys, how="left").merge(gstd, on=keys, how="left")
                eps = 1e-12
                gdf[out_name] = (gdf[value_col] - gdf["__gmean__"]) / (gdf["__gstd__"] + eps)
                gdf[out_name] = gdf[out_name].fillna(0.0)
                out = to_polars(gdf[["__rid__", out_name]])
                if out is not None:
                    out = out.sort("__rid__").drop("__rid__")
                    return df.hstack([out])
        except Exception:
            pass
    # CPU fallback
    return df.with_columns(
        [
            (
                (pl.col(value_col) - pl.col(value_col).mean().over(list(group_keys)))
                / (pl.col(value_col).std().over(list(group_keys)) + 1e-12)
            ).alias(out_name)
        ]
    )
