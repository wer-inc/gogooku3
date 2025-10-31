"""Utility helpers for retrieving market data via yfinance."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def get_yfinance_module(*, raise_on_missing: bool = True) -> Any | None:
    try:
        import yfinance as yf  # type: ignore

        return yf
    except Exception:
        if raise_on_missing:
            raise
        return None


def flatten_yfinance_columns(frame, *, ticker: str | None = None):  # type: ignore[no-untyped-def]
    """Flatten MultiIndex columns produced by yfinance downloads."""

    if hasattr(frame, "columns") and getattr(frame.columns, "nlevels", 1) > 1:
        frame.columns = ["_".join([c for c in col if c]).strip("_") for col in frame.columns.to_flat_index()]  # type: ignore[attr-defined]
    if ticker and hasattr(frame, "columns"):
        prefix = f"{ticker.replace('^', '')}_"
        frame.columns = [col.replace(prefix, "") if isinstance(col, str) else col for col in frame.columns]
    return frame


def resolve_cached_parquet(parquet_path: Path | None, *, prefix: str, start: str, end: str) -> Path | None:
    if parquet_path is not None:
        return parquet_path
    cache_dir = Path("output") / "macro"
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = f"{prefix}_history_{start.replace('-', '')}_{end.replace('-', '')}.parquet"
    return cache_dir / name
