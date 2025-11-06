"""Schema normalization utilities for duplicate column handling."""
from __future__ import annotations

from collections import Counter
from typing import Any, MutableMapping

import polars as pl

CANONICAL_FAMILY = (
    "adjustmentclose",
    "adjustmentopen",
    "adjustmenthigh",
    "adjustmentlow",
    "adjustmentvolume",
)

_ALIAS_SANITIZE = {
    "adjclose": "adjclose_yf",
    "adj_close": "adjclose_yf",
    "adjcloseyf": "adjclose_yf",
    "adjcloseytm": "adjclose_yf",
    "adjcloseyf": "adjclose_yf",
    "adjcloseyfraw": "adjclose_yf",
    "adj close": "adjclose_yf",
    "close": "close_raw",
    "closeprice": "close_raw",
    "endprice": "close_raw",
    "open": "open_raw",
    "openprice": "open_raw",
    "startprice": "open_raw",
    "high": "high_raw",
    "highprice": "high_raw",
    "low": "low_raw",
    "lowprice": "low_raw",
    "volume": "volume_raw",
    "tradingvolume": "volume_raw",
}

_BANNED_EXACT = {
    "Close",
    "Open",
    "High",
    "Low",
    "Volume",
    "Adj Close",
    "AdjClose",
    "adjclose",
    "adj_close",
    "close",
    "open",
    "high",
    "low",
    "volume",
}

_BANNED_PREFIXES = (
    "Close_",
    "Open_",
    "High_",
    "Low_",
    "Volume_",
    "Adj Close_",
    "AdjClose_",
)


def _normalize_alias(name: str) -> str:
    return name.replace(" ", "").replace("-", "").lower()


def canonicalize_ohlc(
    df: pl.DataFrame,
    *,
    meta: MutableMapping[str, Any] | None = None,
) -> pl.DataFrame:
    """Enforce a single canonical OHLCV family (`adjustment*` columns).

    Drops raw OHLC variants (Close/Open/...) and yfinance-style Adj Close,
    records provenance into `meta`, and fail-fast if canonical columns are
    missing. The canonical family is kept in lowercase form so that downstream
    renames can normalize as needed.
    """
    alias_map: dict[str, str] = {}
    rename_plan: dict[str, str] = {}
    existing = set(df.columns)

    # 1) sanitize aliases to collision-free temporary names
    for column in list(df.columns):
        normalized = _normalize_alias(column)
        target_base = _ALIAS_SANITIZE.get(normalized)
        if target_base is None:
            continue
        alias_map[column] = target_base
        candidate = target_base
        suffix = 1
        while candidate in existing or candidate in rename_plan.values():
            suffix += 1
            candidate = f"{target_base}__{suffix}"
        rename_plan[column] = candidate
        existing.add(candidate)

    if rename_plan:
        df = df.rename(rename_plan)

    canonical_missing = [col for col in CANONICAL_FAMILY if col not in df.columns]
    if canonical_missing:
        raise RuntimeError(f"[canon] missing canonical OHLCV columns: {canonical_missing}")

    # 2) drop alias columns that were sanitized and other banned variants
    dropped: set[str] = set()

    sanitized_aliases = set(rename_plan.values())
    drop_candidates = list(sanitized_aliases)

    for column in df.columns:
        if column in CANONICAL_FAMILY:
            continue
        if column in {"turnovervalue", "adjustmentfactor"}:
            continue
        if column in sanitized_aliases:
            continue  # already queued
        if column in _BANNED_EXACT:
            drop_candidates.append(column)
            continue
        if any(column.startswith(prefix) for prefix in _BANNED_PREFIXES):
            drop_candidates.append(column)

    if drop_candidates:
        unique_candidates = [c for c in dict.fromkeys(drop_candidates) if c in df.columns]
        df = df.drop(unique_candidates)
        dropped.update(unique_candidates)

    # 3) ensure no banned columns remain
    banned_present = [col for col in _BANNED_EXACT if col in df.columns]
    if banned_present:
        raise RuntimeError(f"[canon] non-canonical OHLCV columns survived drop: {banned_present}")

    if meta is not None:
        schema_meta = meta.setdefault("schema_governance", {})
        schema_meta["alias_map"] = {k: alias_map[k] for k in sorted(alias_map)}
        schema_meta["dropped"] = sorted(dropped)
        schema_meta["canonical_family"] = list(CANONICAL_FAMILY)
        schema_meta.setdefault("policy", "canonical_adjusted_only_split_no_yf_adj")

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
    prefer = prefer or list(CANONICAL_FAMILY)

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
