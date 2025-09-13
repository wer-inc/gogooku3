from __future__ import annotations

"""
Asynchronous J-Quants API fetcher (extracted from legacy pipeline).

Minimal surface used by current pipelines:
 - authenticate(session)
 - get_trades_spec(session, from_date, to_date) -> pl.DataFrame
 - get_listed_info(session, date=None) -> pl.DataFrame
 - fetch_topix_data(session, from_date, to_date) -> pl.DataFrame

Notes:
 - Optionally filters listed_info via scripts.components.market_code_filter if available.
 - No dependency on scripts/_archive; safe when _archive is removed.
"""

from typing import List, Optional
import asyncio
import os

import aiohttp
import polars as pl


class JQuantsAsyncFetcher:
    """Asynchronous J-Quants API fetcher with basic pagination handling."""

    def __init__(self, email: str, password: str, max_concurrent: int | None = None):
        self.email = email
        self.password = password
        self.base_url = "https://api.jquants.com/v1"
        self.id_token: str | None = None
        self.max_concurrent = max_concurrent or int(os.getenv("MAX_CONCURRENT_FETCH", 32))
        self.semaphore = asyncio.Semaphore(self.max_concurrent)

    async def authenticate(self, session: aiohttp.ClientSession) -> None:
        """Authenticate and store ID token."""
        # 1) Refresh token
        auth_url = f"{self.base_url}/token/auth_user"
        payload = {"mailaddress": self.email, "password": self.password}
        async with session.post(auth_url, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
            refresh_token = data["refreshToken"]

        # 2) ID token
        refresh_url = f"{self.base_url}/token/auth_refresh"
        params = {"refreshtoken": refresh_token}
        async with session.post(refresh_url, params=params) as resp:
            resp.raise_for_status()
            data = await resp.json()
            self.id_token = data["idToken"]

    async def get_listed_info(
        self, session: aiohttp.ClientSession, date: Optional[str] = None
    ) -> pl.DataFrame:
        """Fetch listed company info; optionally filter by market codes if helper is available."""
        if not self.id_token:
            raise RuntimeError("authenticate() must be called first")
        url = f"{self.base_url}/listed/info"
        headers = {"Authorization": f"Bearer {self.id_token}"}
        params = {"date": date} if date else None
        async with session.get(url, headers=headers, params=params) as resp:
            if resp.status != 200:
                return pl.DataFrame()
            data = await resp.json()
            df = pl.DataFrame(data.get("info", []))
        # Optional filter using scripts/components if available
        try:  # pragma: no cover - optional path
            from scripts.components.market_code_filter import MarketCodeFilter  # type: ignore

            if not df.is_empty():
                df = MarketCodeFilter.filter_stocks(df)
        except Exception:
            pass
        return df

    async def get_trades_spec(
        self, session: aiohttp.ClientSession, from_date: str, to_date: str
    ) -> pl.DataFrame:
        """Fetch markets/trades_spec (weekly investor flows)."""
        if not self.id_token:
            raise RuntimeError("authenticate() must be called first")
        url = f"{self.base_url}/markets/trades_spec"
        headers = {"Authorization": f"Bearer {self.id_token}"}
        params = {"from": from_date, "to": to_date}
        async with session.get(url, headers=headers, params=params) as resp:
            if resp.status != 200:
                return pl.DataFrame()
            data = await resp.json()
            items = data.get("trades_spec", [])
            return pl.DataFrame(items) if items else pl.DataFrame()

    async def fetch_topix_data(
        self, session: aiohttp.ClientSession, from_date: str, to_date: str
    ) -> pl.DataFrame:
        """Fetch TOPIX index time series; handles pagination when provided."""
        if not self.id_token:
            raise RuntimeError("authenticate() must be called first")
        url = f"{self.base_url}/indices/topix"
        headers = {"Authorization": f"Bearer {self.id_token}"}

        all_rows: list[dict] = []
        pagination_key: str | None = None
        while True:
            params = {"from": from_date, "to": to_date}
            if pagination_key:
                params["pagination_key"] = pagination_key
            async with session.get(url, headers=headers, params=params) as resp:
                if resp.status != 200:
                    break
                data = await resp.json()
                rows = data.get("topix", [])
                if rows:
                    all_rows.extend(rows)
                pagination_key = data.get("pagination_key")
                if not pagination_key:
                    break

        if not all_rows:
            return pl.DataFrame()
        df = pl.DataFrame(all_rows)
        if "Date" in df.columns:
            df = df.with_columns(pl.col("Date").str.strptime(pl.Date, strict=False))
        if "Close" in df.columns:
            df = df.with_columns(pl.col("Close").cast(pl.Float64))
        return df.sort("Date")

    async def get_weekly_margin_interest(
        self, session: aiohttp.ClientSession, from_date: str, to_date: str
    ) -> pl.DataFrame:
        """Fetch weekly margin interest series.

        Returns normalized DataFrame with columns:
          - Code (Utf8)
          - Date (Date)
          - PublishedDate (Date, nullable)
          - LongMarginTradeVolume, ShortMarginTradeVolume,
            LongNegotiableMarginTradeVolume, ShortNegotiableMarginTradeVolume,
            LongStandardizedMarginTradeVolume, ShortStandardizedMarginTradeVolume (Float64)
          - IssueType (Int8)
        """
        if not self.id_token:
            raise RuntimeError("authenticate() must be called first")
        url = f"{self.base_url}/markets/weekly_margin_interest"
        headers = {"Authorization": f"Bearer {self.id_token}"}
        import datetime as _dt

        def _parse(d: str) -> _dt.date:
            # Accept YYYY-MM-DD or YYYYMMDD
            if "-" in d:
                return _dt.datetime.strptime(d, "%Y-%m-%d").date()
            return _dt.datetime.strptime(d, "%Y%m%d").date()

        start = _parse(from_date)
        end = _parse(to_date)

        async def _fetch_for_date(date_str: str) -> list[dict]:
            rows: list[dict] = []
            pagination_key: str | None = None
            while True:
                params = {"date": date_str}
                if pagination_key:
                    params["pagination_key"] = pagination_key
                async with session.get(url, headers=headers, params=params) as resp:
                    if resp.status != 200:
                        break
                    data = await resp.json()
                if isinstance(data, dict):
                    items = data.get("weekly_margin_interest", [])
                    if items:
                        rows.extend(items)
                    pagination_key = data.get("pagination_key")
                    if not pagination_key:
                        break
                else:
                    break
            return rows

        all_rows: list[dict] = []
        cur = start
        # Query weekly snapshots across the range. Prefer Fridays (weekday=4).
        while cur <= end:
            if cur.weekday() == 4:
                items = await _fetch_for_date(cur.strftime("%Y-%m-%d"))
                if items:
                    all_rows.extend(items)
            cur += _dt.timedelta(days=1)
        if not all_rows:
            return pl.DataFrame()
        df = pl.DataFrame(all_rows)
        # Normalize dtypes
        def _dtcol(name: str) -> pl.Expr:
            col = pl.col(name)
            return (
                pl.when(col.dtype == pl.Utf8)
                .then(col.str.strptime(pl.Date, strict=False))
                .otherwise(col.cast(pl.Date))
                .alias(name)
            )
        cols = df.columns
        out = df.with_columns([
            pl.col("Code").cast(pl.Utf8) if "Code" in cols else pl.lit(None, dtype=pl.Utf8).alias("Code"),
            _dtcol("Date") if "Date" in cols else pl.lit(None, dtype=pl.Date).alias("Date"),
            pl.when(pl.col("PublishedDate").is_not_null())
            .then(_dtcol("PublishedDate"))
            .otherwise(pl.lit(None, dtype=pl.Date))
            .alias("PublishedDate"),
            pl.col("LongMarginTradeVolume").cast(pl.Float64) if "LongMarginTradeVolume" in cols else pl.lit(None, dtype=pl.Float64).alias("LongMarginTradeVolume"),
            pl.col("ShortMarginTradeVolume").cast(pl.Float64) if "ShortMarginTradeVolume" in cols else pl.lit(None, dtype=pl.Float64).alias("ShortMarginTradeVolume"),
            pl.col("LongNegotiableMarginTradeVolume").cast(pl.Float64) if "LongNegotiableMarginTradeVolume" in cols else pl.lit(None, dtype=pl.Float64).alias("LongNegotiableMarginTradeVolume"),
            pl.col("ShortNegotiableMarginTradeVolume").cast(pl.Float64) if "ShortNegotiableMarginTradeVolume" in cols else pl.lit(None, dtype=pl.Float64).alias("ShortNegotiableMarginTradeVolume"),
            pl.col("LongStandardizedMarginTradeVolume").cast(pl.Float64) if "LongStandardizedMarginTradeVolume" in cols else pl.lit(None, dtype=pl.Float64).alias("LongStandardizedMarginTradeVolume"),
            pl.col("ShortStandardizedMarginTradeVolume").cast(pl.Float64) if "ShortStandardizedMarginTradeVolume" in cols else pl.lit(None, dtype=pl.Float64).alias("ShortStandardizedMarginTradeVolume"),
            (pl.col("IssueType").cast(pl.Int8) if "IssueType" in cols else pl.lit(None, dtype=pl.Int8)).alias("IssueType"),
        ]).drop_nulls(subset=["Code", "Date"]).sort(["Code", "Date"])
        return out
