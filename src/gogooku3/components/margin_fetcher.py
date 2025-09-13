from __future__ import annotations

"""
MarginFetcher
 - Thin wrapper interface for J-Quants margin interest endpoints.
 - This module does not perform network I/O by itself; it expects a `client`
   object that provides `get_weekly_margin_interest(start_date, end_date)` and
   `get_daily_margin_interest(start_date, end_date)` compatible methods.

Returned DataFrames are normalized for downstream Polars processing and follow
the expected schema used by margin_weekly feature builder.
"""

from typing import Any

import polars as pl


class MarginFetcher:
    def __init__(self, client: Any, logger: Any, calendar: Any | None = None):
        self.client = client
        self.logger = logger
        self.calendar = calendar

    def fetch_weekly(self, start_date: str, end_date: str) -> pl.DataFrame:
        """
        Fetch weekly margin interest and coerce to a consistent schema.

        Returns a Polars DataFrame with columns:
          - Code (Utf8)
          - Date (Date)                     # week reference date (usually Friday)
          - PublishedDate (Date, nullable)  # optional publish date
          - LongMarginTradeVolume (Float64)
          - ShortMarginTradeVolume (Float64)
          - LongNegotiableMarginTradeVolume (Float64)
          - ShortNegotiableMarginTradeVolume (Float64)
          - LongStandardizedMarginTradeVolume (Float64)
          - ShortStandardizedMarginTradeVolume (Float64)
          - IssueType (Int8)
        """
        try:
            raw = self.client.get_weekly_margin_interest(start_date, end_date)
        except Exception as e:  # pragma: no cover - defensive
            if self.logger:
                self.logger.warning(f"weekly_margin_interest fetch failed: {e}")
            raw = None
        if raw is None:
            return pl.DataFrame(
                schema={
                    "Code": pl.Utf8,
                    "Date": pl.Date,
                    "PublishedDate": pl.Date,
                    "LongMarginTradeVolume": pl.Float64,
                    "ShortMarginTradeVolume": pl.Float64,
                    "LongNegotiableMarginTradeVolume": pl.Float64,
                    "ShortNegotiableMarginTradeVolume": pl.Float64,
                    "LongStandardizedMarginTradeVolume": pl.Float64,
                    "ShortStandardizedMarginTradeVolume": pl.Float64,
                    "IssueType": pl.Int8,
                }
            )

        df = pl.DataFrame(raw)
        cols = df.columns
        # Cast and normalize
        def _cast_if(name: str, dtype: pl.DataType) -> pl.Expr:
            return (pl.col(name).cast(dtype)) if name in cols else pl.lit(None, dtype)

        out = (
            df.with_columns(
                [
                    _cast_if("Code", pl.Utf8),
                    pl.when(pl.col("Date").dtype == pl.Utf8)
                    .then(pl.col("Date").str.strptime(pl.Date, strict=False))
                    .otherwise(pl.col("Date").cast(pl.Date))
                    .alias("Date"),
                    pl.when(pl.col("PublishedDate").is_not_null())
                    .then(
                        pl.when(pl.col("PublishedDate").dtype == pl.Utf8)
                        .then(pl.col("PublishedDate").str.strptime(pl.Date, strict=False))
                        .otherwise(pl.col("PublishedDate").cast(pl.Date))
                    )
                    .otherwise(pl.lit(None, dtype=pl.Date))
                    .alias("PublishedDate"),
                    _cast_if("LongMarginTradeVolume", pl.Float64),
                    _cast_if("ShortMarginTradeVolume", pl.Float64),
                    _cast_if("LongNegotiableMarginTradeVolume", pl.Float64),
                    _cast_if("ShortNegotiableMarginTradeVolume", pl.Float64),
                    _cast_if("LongStandardizedMarginTradeVolume", pl.Float64),
                    _cast_if("ShortStandardizedMarginTradeVolume", pl.Float64),
                    _cast_if("IssueType", pl.Int8),
                ]
            )
            .drop_nulls(subset=["Code", "Date"])  # keep essentials
            .sort(["Code", "Date"])  # stable order
        )
        if self.logger:
            self.logger.info(f"Fetched weekly_margin_interest: {out.height} rows")
        return out

    def fetch_daily(self, start_date: str, end_date: str) -> pl.DataFrame:
        """Optional helper for /markets/daily_margin_interest (may be unavailable).

        Returns an empty DataFrame when the endpoint is inaccessible. The weekly
        series remains the primary source for downstream features.
        """
        try:
            raw = self.client.get_daily_margin_interest(start_date, end_date)
        except Exception:  # pragma: no cover - defensive only
            return pl.DataFrame({"Code": pl.Series([], pl.Utf8)})
        if raw is None:
            return pl.DataFrame({"Code": pl.Series([], pl.Utf8)})
        df = pl.DataFrame(raw)
        if df.is_empty():
            return df
        # Keep minimal normalization to ease optional reconciliation if used
        return (
            df.with_columns(
                [
                    pl.col("Code").cast(pl.Utf8),
                    pl.when(pl.col("Date").dtype == pl.Utf8)
                    .then(pl.col("Date").str.strptime(pl.Date, strict=False))
                    .otherwise(pl.col("Date")).alias("Date"),
                ]
            )
            .sort(["Code", "Date"])
        )

