from __future__ import annotations

import os
import importlib
from pathlib import Path
import re
from typing import Any

import pandas as pd
from pandas import MultiIndex

__all__ = [
    "flatten_yfinance_columns",
    "get_yfinance_module",
    "ensure_yfinance_available",
    "is_yfinance_available",
    "resolve_cached_parquet",
]

_YFINANCE_CACHE: Any | None = None
_CACHE_CONFIGURED = False


def _configure_cache(module: Any) -> None:
    """
    Configure yfinance cache location to reside within the workspace.

    This avoids the default sqlite cache path which can be read-only in CI
    environments (triggering ``OperationalError: attempt to write a readonly
    database``).
    """

    global _CACHE_CONFIGURED
    if _CACHE_CONFIGURED:
        return

    disable = os.getenv("YFINANCE_DISABLE_CACHE")
    cache_dir_env = os.getenv("YFINANCE_CACHE_DIR")
    # Interpret "0"/"false" as False so users can explicitly turn it back on.
    disable_cache = False
    if disable is not None:
        disable_cache = disable.strip().lower() in {"1", "true", "yes", "on"}

    cache_dir = None
    if not disable_cache:
        default_dir = Path("output/cache/yfinance")
        cache_dir = Path(cache_dir_env) if cache_dir_env else default_dir
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            cache_dir = None

    set_cache = getattr(module, "set_cache_location", None)
    set_tz_cache = getattr(module, "set_tz_cache_location", None)

    try:
        if disable_cache or cache_dir is None:
            if callable(set_cache):
                set_cache(None)
        else:
            if callable(set_cache):
                set_cache(str(cache_dir))
            if callable(set_tz_cache):
                tz_dir = cache_dir / "tz"
                tz_dir.mkdir(parents=True, exist_ok=True)
                set_tz_cache(str(tz_dir))
    except Exception:
        # On failure disable caching to avoid hard errors.
        if callable(set_cache):
            try:
                set_cache(None)
            except Exception:
                pass

    _CACHE_CONFIGURED = True


def _import_yfinance() -> Any:
    """Import yfinance with caching; raise RuntimeError on failure."""
    global _YFINANCE_CACHE
    if _YFINANCE_CACHE is not None:
        return _YFINANCE_CACHE

    try:
        module = importlib.import_module("yfinance")
    except Exception as exc:  # pragma: no cover - defensive import guard
        raise RuntimeError("yfinance is not installed or failed to import") from exc

    if not hasattr(module, "download"):
        raise RuntimeError(
            "Imported yfinance module is missing required 'download' attribute."
        )

    _configure_cache(module)
    _YFINANCE_CACHE = module
    return module


def get_yfinance_module(*, raise_on_missing: bool = True) -> Any | None:
    """
    Retrieve the cached yfinance module if available.

    Args:
        raise_on_missing: When True, raise RuntimeError if the module cannot be imported.

    Returns:
        The imported yfinance module or None when unavailable and raise_on_missing is False.
    """

    try:
        return _import_yfinance()
    except RuntimeError:
        if raise_on_missing:
            raise
        return None


def ensure_yfinance_available() -> None:
    """Raise a RuntimeError if yfinance cannot be imported."""
    _import_yfinance()


def is_yfinance_available() -> bool:
    """Return True when yfinance can be imported."""
    try:
        _import_yfinance()
        return True
    except RuntimeError:
        return False


def flatten_yfinance_columns(
    frame: pd.DataFrame, *, ticker: str | None = None
) -> pd.DataFrame:
    """
    Flatten yfinance download columns into a single level.

    Args:
        frame: Pandas DataFrame returned by ``yfinance.download``.
        ticker: Expected ticker symbol. When provided, redundant levels that
            only contain this ticker (or empty strings) are dropped.

    Returns:
        DataFrame copy with single-level columns and a normalized ``Date`` name.
    """

    if frame.empty:
        return frame.copy()

    df = frame.copy()

    columns = df.columns
    if isinstance(columns, MultiIndex) and columns.nlevels > 1:
        level_one_values = set(columns.get_level_values(1).tolist())
        if ticker is not None:
            level_one_values -= {ticker}
        level_one_values -= {""}
        if not level_one_values:
            df.columns = columns.get_level_values(0)
        else:
            df.columns = [
                "_".join(str(part) for part in col if part not in (None, "", ticker))
                for col in columns.to_flat_index()
            ]
    elif isinstance(columns, MultiIndex):
        df.columns = [
            "_".join(str(part) for part in col if part)
            for col in columns.to_flat_index()
        ]

    if "Date" not in df.columns:
        for col in df.columns:
            if isinstance(col, str) and col.lower() == "date":
                df = df.rename(columns={col: "Date"})
                break

    return df


def resolve_cached_parquet(
    parquet_path: Path | None,
    *,
    prefix: str,
    start: str,
    end: str,
    default_dir: Path | None = None,
) -> Path | None:
    """
    Given a desired cache path, locate the best matching cached parquet.

    If the exact path does not exist, the function searches for files named
    ``{prefix}_history_YYYYMMDD_YYYYMMDD.parquet`` and returns the one that
    best covers the requested [start, end] range.
    """

    if parquet_path and parquet_path.exists():
        return parquet_path

    search_dir = (
        parquet_path.parent
        if parquet_path is not None
        else (default_dir or Path("output/macro"))
    )
    if not search_dir.exists():
        return parquet_path

    pattern = re.compile(rf"{re.escape(prefix)}_history_(\d{{8}})_(\d{{8}})\.parquet$")
    best_path = None
    best_score = -1

    try:
        start_int = int(start.replace("-", ""))
        end_int = int(end.replace("-", ""))
    except ValueError:
        start_int = end_int = 0

    for candidate in search_dir.glob(f"{prefix}_history_*.parquet"):
        match = pattern.match(candidate.name)
        if not match:
            continue
        c_start = int(match.group(1))
        c_end = int(match.group(2))
        coverage = c_end - c_start
        encloses = c_start <= start_int and c_end >= end_int
        score = coverage + (1_000_000 if encloses else 0)
        if score > best_score:
            best_score = score
            best_path = candidate

    if best_path and best_path.exists():
        return best_path

    return parquet_path
