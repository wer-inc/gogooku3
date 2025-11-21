"""Helpers for saving raw API payload snapshots for reproducibility."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl

from .logger import get_logger

LOGGER = get_logger("builder.raw_snapshot")


def save_raw_snapshot(
    *,
    root: Path,
    source: str,
    df: pl.DataFrame,
    start: Optional[str] = None,
    end: Optional[str] = None,
    filename: Optional[str] = None,
    overwrite: bool = True,
) -> Path | None:
    """
    Persist a raw API payload for a given source and date window.

    Args:
        root: Base raw data directory (e.g., settings.raw_data_dir).
        source: Logical source name (e.g., \"prices\", \"macro_vix\").
        df: Raw DataFrame to persist (no-op if empty).
        start: Optional ISO start date used in naming (YYYY-MM-DD).
        end: Optional ISO end date used in naming (YYYY-MM-DD).
        filename: Optional explicit filename override.
        overwrite: If False, skips writing when the target already exists.

    Returns:
        Path to the written Parquet file, or None if skipped/df empty.
    """
    if df.is_empty():
        return None

    source_dir = root / source
    source_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        if start and end:
            filename = f"{source}_{start}_{end}_{ts}.parquet"
        elif start:
            filename = f"{source}_{start}_{ts}.parquet"
        else:
            filename = f"{source}_{ts}.parquet"

    path = source_dir / filename
    if path.exists() and not overwrite:
        LOGGER.debug("[RAW] Snapshot already exists, skipping: %s", path)
        return path

    try:
        df.write_parquet(path, compression="zstd")
        LOGGER.info(
            "[RAW] Saved snapshot for %s (%s→%s) to %s (%d rows, %d cols)",
            source,
            start,
            end,
            path,
            df.height,
            df.width,
        )
        return path
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("[RAW] Failed to save snapshot for %s to %s: %s", source, path, exc)
        return None
