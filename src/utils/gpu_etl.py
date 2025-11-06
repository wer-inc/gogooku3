from __future__ import annotations

import logging
import os
from collections.abc import Iterable

import polars as pl

"""
GPU-ETL utilities (optional, safe fallback to CPU).

Provides thin wrappers to:
- Initialize RMM pooled allocator to make the most of GPU memory
- Convert between Polars <-> cuDF
- Compute cross-sectional rank/z-score on GPU (by given group keys)

If RAPIDS/cuDF or a CUDA device is not available, every function falls back to
CPU/Polars and preserves the original outputs/column names to avoid regressions.
"""

# Workaround for cuDF 24.12 + numba-cuda 0.0.17+ compatibility issue
# Must be done before any cuDF imports
try:
    import pynvjitlink.patch

    def _noop_patch(*_args, **_kwargs):  # type: ignore  # noqa: ARG001
        pass

    pynvjitlink.patch.patch_numba_linker = _noop_patch
except Exception:
    pass


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


def init_rmm_legacy(initial_pool_size: str | None = None) -> bool:
    """Initialize RMM memory pool (best-effort).

    Args:
        initial_pool_size: e.g., "70GB"; if None, read env RMM_POOL_SIZE

    Returns:
        True if RMM initialized, else False (safe to ignore)
    """
    logger = logging.getLogger(__name__)

    try:
        import cupy as cp  # type: ignore
        import rmm  # type: ignore

        # Check RMM_ALLOCATOR environment variable (cuda_async vs pool)
        allocator_mode = os.getenv("RMM_ALLOCATOR", "pool").lower()
        size_str = initial_pool_size or os.getenv("RMM_POOL_SIZE", "")

        # Determine whether to use pool allocator
        # Use dynamic allocation (no pool) if:
        # 1. RMM_ALLOCATOR=cuda_async explicitly set
        # 2. RMM_POOL_SIZE=0 explicitly set
        use_pool = allocator_mode != "cuda_async" and size_str != "0"

        kwargs: dict = {}

        if use_pool:
            kwargs["pool_allocator"] = True

            if size_str:
                # Convert human-readable size to bytes
                size_bytes = None
                size_str_upper = size_str.upper().strip()

                if size_str_upper.endswith("GB"):
                    size_bytes = int(float(size_str_upper[:-2]) * 1024**3)
                elif size_str_upper.endswith("G"):
                    size_bytes = int(float(size_str_upper[:-1]) * 1024**3)
                elif size_str_upper.endswith("MB"):
                    size_bytes = int(float(size_str_upper[:-2]) * 1024**2)
                elif size_str_upper.endswith("M"):
                    size_bytes = int(float(size_str_upper[:-1]) * 1024**2)
                else:
                    # Assume it's already in bytes or a plain number
                    try:
                        size_bytes = int(size_str)
                    except ValueError:
                        pass

                if size_bytes and size_bytes > 0:
                    kwargs["initial_pool_size"] = size_bytes
                    logger.info(f"RMM initialized with pool allocator, initial size={size_str}")
        else:
            # Dynamic allocation mode (cuda_async or pool_size=0)
            kwargs["pool_allocator"] = False
            logger.info(f"RMM initialized with dynamic allocation (allocator={allocator_mode}, no pool)")

        # Managed memory can be problematic on some systems, make it optional
        if os.getenv("RMM_MANAGED_MEMORY", "0") == "1":
            kwargs["managed_memory"] = True

        rmm.reinitialize(**kwargs)
        attached = _attach_cupy_allocator(cp, rmm, logger)
        if not attached:
            logger.warning(
                "RMM initialized but CuPy allocator attachment failed; GPU ETL will use default CuPy allocator"
            )
        return True
    except Exception as e:
        logger.warning(f"RMM init failed: {e}")
        # Try simpler initialization without managed memory
        try:
            import cupy as cp
            import rmm

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
                keys = list(group_keys)
                # Rank within groups (average ties), NA kept
                gdf[out_rank_name] = (
                    gdf.groupby(keys)[rank_col].rank(method="average", ascending=True).astype("float64")
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

                out = to_polars(gdf[["__rid__", out_rank_name, out_z_name]])
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
        out = out.with_columns([pl.when(cnt > 1).then((rk - 1.0) / (cnt - 1.0)).otherwise(0.5).alias(out_rank_name)])
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


def init_rmm(initial_pool_size: str | None = None) -> bool:
    """Modern RMM initialization that explicitly supports cuda_async.

    Behavior:
    - If RMM_ALLOCATOR=cuda_async → use CudaAsyncMemoryResource. When RMM_POOL_SIZE>0,
      wrap with PoolMemoryResource using the requested initial pool size.
    - Otherwise falls back to legacy reinitialize semantics (via init_rmm_legacy).
    """
    logger = logging.getLogger(__name__)

    def _parse_bytes(txt: str | None) -> int | None:
        if not txt:
            return None
        s = txt.strip().upper()
        try:
            if s.endswith("GB"):
                return int(float(s[:-2]) * (1024**3))
            if s.endswith("G"):
                return int(float(s[:-1]) * (1024**3))
            if s.endswith("MB"):
                return int(float(s[:-2]) * (1024**2))
            if s.endswith("M"):
                return int(float(s[:-1]) * (1024**2))
            if s.endswith("KB"):
                return int(float(s[:-2]) * 1024)
            if s.endswith("K"):
                return int(float(s[:-1]) * 1024)
            return int(float(s))
        except Exception:
            return None

    try:
        import cupy as cp  # type: ignore
        import rmm  # type: ignore

        allocator_mode = os.getenv("RMM_ALLOCATOR", "pool").lower().strip()
        size_str = (initial_pool_size if initial_pool_size is not None else os.getenv("RMM_POOL_SIZE", "")).strip()
        size_bytes = _parse_bytes(size_str)

        if allocator_mode == "cuda_async":
            if size_bytes and size_bytes > 0:
                try:
                    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()  # type: ignore[attr-defined]
                    safety_margin = int(2 * 1024**3)  # 2GB safety margin
                    max_allowed = max(0, free_bytes - safety_margin)
                    if size_bytes >= max_allowed and max_allowed > 0:
                        logger.warning(
                            "Requested RMM pool (%s) exceeds available GPU memory (%.1f GB free). "
                            "Clamping to %.1f GB.",
                            size_str or f"{size_bytes}B",
                            free_bytes / 1e9,
                            max_allowed / 1e9,
                        )
                        size_bytes = max_allowed
                except Exception as exc:  # pragma: no cover - defensive guard
                    logger.debug(f"Failed to query CUDA free memory: {exc}")
            try:
                import rmm.mr as mr  # type: ignore

                base = mr.CudaAsyncMemoryResource()
                use_pool = bool(size_bytes) and size_bytes > 0
                resource = mr.PoolMemoryResource(base, initial_pool_size=size_bytes) if use_pool else base
                mr.set_current_device_resource(resource)

                attached = _attach_cupy_allocator(cp, rmm, logger)
                if attached:
                    logger.info(
                        f"RMM cuda_async configured (pool={'on' if use_pool else 'off'}, size={size_str or '0'})"
                    )
                else:
                    logger.warning(
                        "RMM cuda_async set, but CuPy allocator attachment failed; CuPy will use default allocator"
                    )
                return True
            except Exception as exc:
                logger.warning(f"cuda_async MR init failed, using legacy path: {exc}")

        # Non-cuda_async or error path → legacy behavior
        return init_rmm_legacy(initial_pool_size)
    except Exception as e:
        logger.warning(f"RMM modern init failed: {e}")
        return init_rmm_legacy(initial_pool_size)


def is_gpu_available() -> bool:
    """Check if GPU/CUDA is available for ETL operations.

    Public wrapper around _has_cuda() for use in feature generators.
    """
    return _has_cuda()


__all__ = [
    "init_rmm",
    "to_cudf",
    "to_polars",
    "cs_rank_and_z",
    "is_gpu_available",
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
