"""Shared helpers for memory-efficient Polars workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import polars as pl

TARGET_PREFIX = "target_"
PRICE_COL = "AdjustmentClose"
CODE_COL = "Code"
DATE_COL = "Date"


def scan_parquet_lazy(path: str | Path) -> pl.LazyFrame:
    """Return a lazy scan with sensible defaults for large Parquet collections."""

    return pl.scan_parquet(str(path), low_memory=True)


def drop_existing_target_columns(
    lf: pl.LazyFrame,
    prefix: str = TARGET_PREFIX,
) -> tuple[pl.LazyFrame, list[str]]:
    """Remove already-materialised target columns to avoid duplicates."""

    schema = lf.collect_schema()
    targets = [name for name in schema.names() if name.startswith(prefix)]
    if targets:
        lf = lf.drop(targets)
    return lf, targets


def add_future_return_columns(
    lf: pl.LazyFrame,
    horizons: Sequence[int],
    *,
    price_col: str = PRICE_COL,
    code_col: str = CODE_COL,
    date_col: str = DATE_COL,
) -> pl.LazyFrame:
    """
    Append forward-return targets using lazy window expressions.

    The underlying dataset is kept lazy; callers should finish with sink_parquet().
    """

    price = pl.col(price_col)
    denom = price + 1e-12
    date_expr = pl.col(date_col)
    target_exprs = []
    for horizon in horizons:
        future_price = price.sort_by(date_expr).shift(-horizon).over(code_col)
        target_exprs.append(((future_price / denom) - 1.0).alias(f"target_{horizon}d"))

    return lf.with_columns(target_exprs)


def dataset_summary(
    path: str | Path,
    horizons: Iterable[int],
    *,
    code_col: str = CODE_COL,
    date_col: str = DATE_COL,
) -> dict:
    """Collect lightweight summary stats (row count, coverage, min/max date)."""

    scan = scan_parquet_lazy(path)
    exprs = [
        pl.len().alias("rows"),
        pl.col(code_col).n_unique().alias("codes"),
        pl.col(date_col).min().alias("date_start"),
        pl.col(date_col).max().alias("date_end"),
    ]
    for horizon in horizons:
        col = f"target_{horizon}d"
        exprs.append(pl.col(col).is_not_null().mean().alias(f"{col}_coverage"))

    summary = scan.select(exprs).collect(streaming=True).to_dicts()[0]
    summary["columns"] = len(scan.collect_schema())
    return summary


def format_coverage(summary: dict, horizons: Iterable[int]) -> list[str]:
    """Human-readable coverage lines for logging/reporting."""

    lines = []
    for horizon in horizons:
        key = f"target_{horizon}d_coverage"
        value = summary.get(key, 0.0) or 0.0
        lines.append(f"  target_{horizon}d: {value * 100:.1f}% non-null")
    return lines
