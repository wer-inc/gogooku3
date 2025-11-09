"""
Phase 2 Patch E: ADV (Average Daily Volume) filter from raw quotes.

Computes 60-day trailing ADV (in JPY) excluding current day to prevent look-ahead bias.
Filters ML dataset to include only stocks with sufficient liquidity.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import polars as pl

from .rolling import roll_mean_safe

LOGGER = logging.getLogger(__name__)


def compute_adv60_from_raw(
    raw_paths: Sequence[str | Path],
    min_periods: int = 20,
) -> pl.DataFrame:
    """
    Compute 60-day trailing ADV from raw quote cache files.

    CRITICAL: Uses ONLY raw data, never ML-processed columns (no rank-transformed volumes).

    Args:
        raw_paths: Paths to raw quote parquet files (e.g., cache/quotes_*.parquet)
        min_periods: Minimum observations for rolling mean (default: 20)

    Returns:
        DataFrame with columns: [code, date, adv60_yen]
        - code: Stock code (4-digit zero-padded string)
        - date: Trading date
        - adv60_yen: 60-day trailing average daily turnover (JPY), current day excluded

    Raises:
        ValueError: If required columns not found in raw data
        RuntimeError: If no data loaded from raw_paths

    Example:
        >>> raw_files = ["output/cache/quotes_*.parquet"]
        >>> adv_df = compute_adv60_from_raw(raw_files)
        >>> print(adv_df.head())
        ┌──────┬────────────┬────────────┐
        │ code │ date       │ adv60_yen  │
        ├──────┼────────────┼────────────┤
        │ 1301 │ 2024-09-02 │ 5.2e8      │
        └──────┴────────────┴────────────┘
    """
    # 1) Load raw data (lazy for efficiency, prefer IPC cache for 3-5x faster reads)
    raw_path_list = [Path(p) for p in raw_paths]
    LOGGER.info("Loading raw quotes from %d file(s): %s", len(raw_path_list), raw_path_list)

    # Build list of LazyFrames, preferring IPC cache when available
    scans = []
    for path in raw_path_list:
        path = Path(path)
        ipc_path = path.with_suffix(".arrow")
        if ipc_path.exists():
            try:
                scans.append(pl.scan_ipc(str(ipc_path)))
                LOGGER.debug("Using IPC cache: %s", ipc_path)
            except Exception as exc:
                LOGGER.warning("Failed to scan IPC cache %s, falling back to Parquet: %s", ipc_path, exc)
                scans.append(pl.scan_parquet(str(path)))
        else:
            scans.append(pl.scan_parquet(str(path)))

    try:
        if len(scans) == 1:
            q = scans[0]
        else:
            # Concatenate multiple LazyFrames
            q = pl.concat(scans, how="vertical")
    except Exception as e:
        raise RuntimeError(f"Failed to load raw quotes from {raw_path_list}: {e}") from e

    # 2) Validate required columns
    required = {"code", "date", "turnovervalue"}
    missing = required - set(q.columns)
    if missing:
        raise ValueError(
            f"Raw quotes missing required columns: {missing}. "
            f"Available: {q.columns}"
        )

    # 3) Normalize code (zero-pad to 4 digits) and ensure date is Date type
    q = q.select([
        pl.col("code").cast(pl.Utf8).str.zfill(4).alias("code"),
        pl.col("date").cast(pl.Date).alias("date"),
        pl.col("turnovervalue").cast(pl.Float64).alias("turnover_yen"),
    ])

    # 4) Sort by (code, date) for rolling operations
    q = q.sort(["code", "date"])

    # 5) Compute 60-day trailing ADV (Phase 2 Patch C: exclude current day)
    # Use roll_mean_safe which applies shift(1) before rolling_mean
    q = q.with_columns([
        roll_mean_safe(
            pl.col("turnover_yen"),
            window=60,
            min_periods=min_periods,
            by="code"
        ).alias("adv60_yen")
    ])

    # 6) Collect and return
    result = q.select(["code", "date", "adv60_yen"]).collect()

    LOGGER.info(
        "Computed ADV60 for %d rows, %d unique codes, date range: %s to %s",
        result.height,
        result["code"].n_unique(),
        result["date"].min(),
        result["date"].max(),
    )

    return result


def apply_adv_filter(
    ml_df: pl.DataFrame,
    adv_df: pl.DataFrame,
    min_adv_yen: int = 50_000_000,
    on_code: str = "code",
    on_date: str = "date",
) -> pl.DataFrame:
    """
    Filter ML dataset to include only stocks with sufficient ADV.

    Args:
        ml_df: ML features dataframe
        adv_df: ADV dataframe from compute_adv60_from_raw()
        min_adv_yen: Minimum ADV in JPY (default: 50M yen)
        on_code: Stock code column name (default: "code")
        on_date: Date column name (default: "date")

    Returns:
        Filtered ML dataframe (only rows with adv60 >= min_adv_yen)

    Raises:
        RuntimeError: If filter removes all rows (check raw data or threshold)

    Example:
        >>> adv_df = compute_adv60_from_raw(["cache/quotes_*.parquet"])
        >>> ml_filtered = apply_adv_filter(ml_df, adv_df, min_adv_yen=50_000_000)
        >>> print(f"Filtered: {len(ml_df)} → {len(ml_filtered)} rows")
    """
    if ml_df.is_empty():
        LOGGER.warning("ML dataframe is empty, returning as-is")
        return ml_df

    if adv_df.is_empty():
        raise ValueError("ADV dataframe is empty, cannot apply filter")

    # 1) Filter ADV dataframe to qualified stocks only
    adv_qualified = (
        adv_df.filter(pl.col("adv60_yen") >= min_adv_yen)
        .select([on_code, on_date])
        .with_columns([pl.lit(True).alias("_adv_ok")])
    )

    LOGGER.info(
        "ADV filter: %d/%d (code, date) pairs qualify (≥%.1fM yen)",
        adv_qualified.height,
        adv_df.height,
        min_adv_yen / 1_000_000,
    )

    # 2) Left join ML dataframe with qualified ADV flags
    result = (
        ml_df.join(adv_qualified, on=[on_code, on_date], how="left")
        .filter(pl.col("_adv_ok").fill_null(False))
        .drop("_adv_ok")
    )

    # 3) Verify filter didn't remove all rows
    if result.is_empty():
        raise RuntimeError(
            f"ADV filter removed all rows. Check raw data or reduce min_adv_yen (current: {min_adv_yen:,} yen)"
        )

    removed_count = ml_df.height - result.height
    removed_pct = 100.0 * removed_count / ml_df.height if ml_df.height > 0 else 0.0

    LOGGER.info(
        "ADV filter applied: %d → %d rows (removed %d, %.1f%%)",
        ml_df.height,
        result.height,
        removed_count,
        removed_pct,
    )

    return result


def get_raw_quotes_paths(
    cache_dir: str | Path = "output/cache",
    pattern: str = "quotes_*.parquet",
) -> list[Path]:
    """
    Find raw quotes cache files matching pattern.

    Args:
        cache_dir: Cache directory path (default: "output/cache")
        pattern: File glob pattern (default: "quotes_*.parquet")

    Returns:
        List of matching parquet file paths

    Raises:
        FileNotFoundError: If no matching files found

    Example:
        >>> paths = get_raw_quotes_paths()
        >>> print(paths)
        [PosixPath('output/cache/quotes_2024-09-02_2025-01-07_abc123.parquet')]
    """
    cache_path = Path(cache_dir)
    matches = sorted(cache_path.glob(pattern))

    if not matches:
        raise FileNotFoundError(
            f"No raw quotes files found: {cache_path}/{pattern}"
        )

    LOGGER.debug("Found %d raw quotes file(s): %s", len(matches), [p.name for p in matches])

    return matches
