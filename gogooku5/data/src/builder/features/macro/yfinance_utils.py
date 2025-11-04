"""Utility helpers for retrieving market data via yfinance."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def get_yfinance_module(*, raise_on_missing: bool = True) -> Any | None:
    try:
        import yfinance as yf  # type: ignore

        return yf
    except Exception:
        if raise_on_missing:
            raise
        return None


def flatten_yfinance_columns(
    frame: pd.DataFrame,
    *,
    ticker: str | None = None,
    sep: str = "_",
) -> pd.DataFrame:
    """Normalize yfinance output for consistent downstream use.

    Behavior:
    1) If columns are MultiIndex and `ticker` is provided, slice that ticker via .xs()
       regardless of whether the layout is ('field','ticker') or ('ticker','field').
       Returns a single-level frame with field-only columns (Open/High/Low/Close/...).

    2) Otherwise, if columns are MultiIndex, flatten by joining levels with `sep`.

    3) If `ticker` is provided on a flat frame, remove the ticker token anywhere in the
       name (prefix or suffix), e.g. 'SPY_Close' -> 'Close', 'Close_SPY' -> 'Close'.

    Args:
        frame: Input pandas DataFrame from yfinance
        ticker: Optional ticker symbol to extract
        sep: Separator for flattening MultiIndex (default: "_")

    Returns:
        DataFrame with normalized column names
    """
    out = frame.copy()

    # 1) MultiIndex + ticker: slice the exact ticker regardless of level order
    if isinstance(out.columns, pd.MultiIndex) and ticker is not None:
        for lvl in range(out.columns.nlevels):
            if ticker in out.columns.get_level_values(lvl):
                sliced = out.xs(ticker, axis=1, level=lvl, drop_level=True)
                sliced.columns.name = None
                return sliced

    # 2) MultiIndex (no ticker provided): flatten
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            sep.join(str(x) for x in tup if x not in (None, ""))
            for tup in out.columns.to_list()
        ]

    # 3) Flat columns + ticker: remove token no matter where it appears
    if ticker is not None:

        def _strip_token(name: str) -> str:
            parts = str(name).split(sep)
            return sep.join([p for p in parts if p != ticker])

        out = out.rename(columns=_strip_token)

    return out


def resolve_cached_parquet(parquet_path: Path | None, *, prefix: str, start: str, end: str) -> Path | None:
    if parquet_path is not None:
        return parquet_path
    cache_dir = Path("output") / "macro"
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = f"{prefix}_history_{start.replace('-', '')}_{end.replace('-', '')}.parquet"
    return cache_dir / name
