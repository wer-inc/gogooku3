"""Schema normalization utilities for duplicate column handling."""
from __future__ import annotations

from collections import Counter

import polars as pl


def canonicalize_ohlc(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize OHLC columns using priority-based coalesce.

    Handles case variations (Close/close/EndPrice) by coalescing
    them into a single canonical column and dropping duplicates.

    Args:
        df: DataFrame with potentially duplicate OHLC columns

    Returns:
        DataFrame with single canonical OHLC columns
    """
    # Column candidates in priority order (first has highest priority)
    candidates = {
        "Close": ["Close", "close", "EndPrice", "AdjClose", "Close_adj"],
        "Open": ["Open", "open", "StartPrice", "OpenPrice"],
        "High": ["High", "high", "HighPrice"],
        "Low": ["Low", "low", "LowPrice"],
        "Volume": ["Volume", "volume", "TradingVolume"],
    }

    exprs = []
    drops = []

    for target, alts in candidates.items():
        # Find which candidates actually exist
        present = [c for c in alts if c in df.columns]
        if present:
            # Coalesce: take first non-null value in priority order
            exprs.append(pl.coalesce([pl.col(c) for c in present]).alias(target))
            # Drop all variants except the target itself
            drops += [c for c in present if c != target]

    if exprs:
        df = df.with_columns(exprs)

    if drops:
        # Remove all OHLC variants, keeping only canonical names
        df = df.drop([c for c in drops if c in df.columns])

    return df


def enforce_unique_columns(df: pl.DataFrame, prefer: list[str] | None = None) -> pl.DataFrame:
    """Safe handling of duplicate column names.

    Keeps first occurrence of each column name and drops subsequent duplicates.

    Args:
        df: DataFrame with potentially duplicate columns
        prefer: Optional list of column names to prioritize

    Returns:
        DataFrame with unique column names
    """
    prefer = prefer or ["Close", "Open", "High", "Low", "Volume"]

    cnt = Counter(df.columns)
    dups = [name for name, k in cnt.items() if k > 1]

    if not dups:
        return df

    # Build select expression keeping only first occurrence of each column
    seen = {}
    new_cols = []

    for c in df.columns:
        if c not in seen:
            seen[c] = 1
            new_cols.append(pl.col(c))
        else:
            # Skip subsequent duplicates (already logged first occurrence)
            continue

    return df.select(new_cols)


def safe_rename(df: pl.DataFrame, mapping: dict[str, str]) -> pl.DataFrame:
    """Rename columns with collision sanitization.

    If rename target already exists, drops the source column first
    to avoid DuplicateError.

    Args:
        df: DataFrame to rename
        mapping: Dict mapping source -> target column names

    Returns:
        DataFrame with renamed columns (collision-free)
    """
    # Clean mapping: remove entries where source doesn't exist
    mapping = {k: v for k, v in mapping.items() if k in df.columns}

    drops = []
    for src, dst in list(mapping.items()):
        if dst in df.columns and src != dst:
            # Target already exists and is different from source
            # Drop source to avoid collision (prioritize existing target)
            drops.append(src)
            mapping.pop(src, None)

    if drops:
        df = df.drop([c for c in drops if c in df.columns])

    if mapping:
        df = df.rename(mapping)

    return df


def ensure_sector_dimensions(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure sector_code and sector columns exist for BATCH-2B compatibility.

    Handles name variations (SectorCode, 17SectorCode, SectorName) and creates
    standardized sector_code (Int32) and sector (Utf8) columns if missing.

    Args:
        df: DataFrame with potentially varying sector column names

    Returns:
        DataFrame with standardized sector_code and sector columns
    """
    cols = set(df.columns)
    out = df

    # sector_code (int)
    if "sector_code" not in cols:
        if "SectorCode" in cols:
            out = out.with_columns(pl.col("SectorCode").cast(pl.Int32).alias("sector_code"))
        elif "17SectorCode" in cols:
            out = out.with_columns(pl.col("17SectorCode").cast(pl.Int32).alias("sector_code"))

    # sector (name)
    if "sector" not in out.columns:
        if "SectorName" in cols:
            out = out.with_columns(pl.col("SectorName").cast(pl.Utf8).alias("sector"))

    return out


def validate_unique_columns(df: pl.DataFrame) -> None:
    """Validate that DataFrame has no duplicate column names.

    Args:
        df: DataFrame to validate

    Raises:
        AssertionError: If duplicate columns are found
    """
    cnt = Counter(df.columns)
    dups = [k for k, v in cnt.items() if v > 1]
    assert not dups, f"Duplicate columns in final output: {dups[:5]} (total {len(dups)})"
