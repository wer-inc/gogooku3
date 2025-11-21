"""GPU availability check for gogooku5 data package.

Simple utility to check if GPU/CUDA is available for accelerated operations.
"""

from __future__ import annotations


def is_gpu_available() -> bool:
    """Check if GPU/CUDA is available for ETL operations.

    Returns:
        True if CUDA-capable GPU is detected, False otherwise.

    Notes:
        - Tries numba.cuda first (fastest check)
        - Falls back to cupy if numba not available
        - Returns False if neither is available or imports fail
    """
    try:
        import numba.cuda as _cuda  # type: ignore

        return _cuda.is_available()
    except Exception:
        try:
            import cupy as _cp  # type: ignore

            return _cp.cuda.runtime.getDeviceCount() > 0  # type: ignore
        except Exception:
            return False


__all__ = ["is_gpu_available"]
