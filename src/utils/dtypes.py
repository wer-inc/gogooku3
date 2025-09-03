#!/usr/bin/env python3
"""
Common dtype utilities (Date/Code normalization).
"""

from __future__ import annotations

import polars as pl


def ensure_date(df: pl.DataFrame, col: str = "Date") -> pl.DataFrame:
    """Ensure the given column is pl.Date (safe cast if possible)."""
    if col not in df.columns:
        return df
    try:
        if df[col].dtype != pl.Date:
            return df.with_columns(pl.col(col).cast(pl.Date))
    except Exception:
        pass
    return df


def ensure_code(df: pl.DataFrame, col: str = "Code") -> pl.DataFrame:
    """Ensure code column is 4-digit zero-padded string (Utf8)."""
    if col not in df.columns:
        return df
    return df.with_columns(pl.col(col).cast(pl.Utf8).str.zfill(4))

