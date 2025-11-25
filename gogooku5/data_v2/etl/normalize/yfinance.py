"""Normalize yfinance price history."""

from __future__ import annotations

from typing import Any

import polars as pl


def normalize_yfinance_history(ticker: str, raw: Any) -> pl.DataFrame:
    """Normalize yfinance download output into (date, ticker, ohlcv)."""

    if raw is None:
        return pl.DataFrame()

    # yfinance returns pandas DataFrame
    df = raw.copy()
    if getattr(df, "empty", True):
        return pl.DataFrame()

    # Ensure Date column exists
    if "Date" not in df.columns:
        if getattr(df.index, "name", None) is None:
            df.index.name = "Date"
        df = df.reset_index()
    else:
        # If Date already present, drop index to avoid duplicate insert
        df = df.reset_index(drop=True)

    # Some intervals return timezone-aware timestamps; drop tz
    if "Date" in df.columns:
        df["Date"] = df["Date"].apply(lambda x: x.tz_localize(None) if hasattr(x, "tz_localize") else x)

    # Standardize column casing
    col_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Adj_Close": "adj_close",
        "AdjClose": "adj_close",
        "Volume": "volume",
        # Flattened MultiIndex variants
        "Price_Open": "open",
        "Price_High": "high",
        "Price_Low": "low",
        "Price_Close": "close",
        "Price_Volume": "volume",
    }
    rename = {c: col_map[c] for c in df.columns if c in col_map}
    df = df.rename(columns=rename)

    # Build Polars frame and enforce schema
    pl_df = pl.from_pandas(df, include_index=False)
    required = ["Date", "open", "high", "low", "close", "adj_close", "volume"]
    missing = [c for c in required if c not in pl_df.columns]
    for col in missing:
        dtype = pl.Float64 if col != "volume" else pl.Int64
        pl_df = pl_df.with_columns(pl.lit(None, dtype=dtype).alias(col))

    # If adj_close is missing but close exists, fill it from close
    if "adj_close" in pl_df.columns and "close" in pl_df.columns:
        pl_df = pl_df.with_columns(
            pl.when(pl.col("adj_close").is_null())
            .then(pl.col("close"))
            .otherwise(pl.col("adj_close"))
            .alias("adj_close")
        )

    out = (
        pl_df.select(
            [
                pl.col("Date").cast(pl.Date, strict=False).alias("date"),
                pl.lit(ticker).cast(pl.Utf8).alias("ticker"),
                pl.col("open").cast(pl.Float64, strict=False),
                pl.col("high").cast(pl.Float64, strict=False),
                pl.col("low").cast(pl.Float64, strict=False),
                pl.col("close").cast(pl.Float64, strict=False),
                pl.col("adj_close").cast(pl.Float64, strict=False),
                pl.col("volume").cast(pl.Int64, strict=False),
            ]
        )
        .drop_nulls(subset=["date"])
        .unique(subset=["date", "ticker"], keep="last")
        .sort(["ticker", "date"])
    )
    return out


def normalize_yfinance_multi(ticker: str, raw: Any) -> pl.DataFrame:
    """Compatibility wrapper for MultiIndex columns when multiple tickers are fetched."""

    # If columns are MultiIndex like ('Close', 'SPY'), pick the requested ticker
    if getattr(raw, "columns", None) is not None and hasattr(raw.columns, "levels"):
        levels = raw.columns
        # level 1 is ticker in yfinance MultiIndex for download([...])
        tickers_in_frame = set([lvl for lvl in levels.get_level_values(-1)])
        if len(tickers_in_frame) > 1:
            # subset to columns for the target ticker only
            try:
                subset = raw.xs(ticker, axis=1, level=-1, drop_level=False)
            except Exception:
                # fallback: filter columns manually
                subset_cols = [c for c in raw.columns if isinstance(c, tuple) and c[-1] == ticker]
                subset = raw[subset_cols]
            df = subset.copy()
            if getattr(df.index, "name", None) is None:
                df.index.name = "Date"
            df = df.reset_index()
            # rename ('Close', 'SPY') -> Close
            new_cols = []
            for col in df.columns:
                if isinstance(col, tuple):
                    name = str(col[0]) if col and col[0] else ""
                    new_cols.append(name or "col")
                else:
                    new_cols.append(str(col))
            df.columns = new_cols
            return normalize_yfinance_history(ticker, df)
        # If only one ticker is present, fall through to single-ticker normalization
    return normalize_yfinance_history(ticker, raw)
