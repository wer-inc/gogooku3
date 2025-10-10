"""
Prices fetcher for J-Quants API (daily quotes, indices).

Extracted from legacy JQuantsAsyncFetcher for better separation of concerns.
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp
import polars as pl

from gogooku3.api.jquants_client import JQuantsClient

logger = logging.getLogger(__name__)


class PricesFetcher:
    """
    Fetcher for price-related data from J-Quants API.

    Responsibilities:
    - Daily quotes (prices/daily_quotes)
    - Indices OHLC (indices/*)
    - TOPIX data (indices/topix)
    """

    def __init__(self, client: JQuantsClient):
        """
        Initialize prices fetcher.

        Args:
            client: Base J-Quants client (authenticated)
        """
        self.client = client
        self.base_url = client.base_url

    async def fetch_daily_quotes_for_date(
        self,
        session: aiohttp.ClientSession,
        date: str,
    ) -> pl.DataFrame:
        """
        Fetch all daily quotes for a specific date with pagination.

        Args:
            session: aiohttp client session
            date: Date (YYYY-MM-DD or YYYYMMDD)

        Returns:
            Polars DataFrame with daily quotes
        """
        if not self.client.id_token:
            raise RuntimeError("authenticate() must be called first")

        url = f"{self.base_url}/prices/daily_quotes"
        headers = {"Authorization": f"Bearer {self.client.id_token}"}
        date_api = self.client.format_date_param(date)

        all_quotes: list[dict] = []
        pagination_key: str | None = None

        while True:
            params: dict[str, str] = {"date": date_api} if date_api else {}
            if pagination_key:
                params["pagination_key"] = pagination_key

            status, data = await self.client._request_json(
                session,
                "GET",
                url,
                label=f"daily_quotes:{date}",
                params=params,
                headers=headers,
            )

            if status != 200:
                break

            if not isinstance(data, dict):
                break

            quotes = data.get("daily_quotes", [])
            if quotes:
                all_quotes.extend(quotes)

            pagination_key = data.get("pagination_key")
            if not pagination_key:
                break

        if not all_quotes:
            return pl.DataFrame()

        return pl.DataFrame(all_quotes)

    async def fetch_daily_quotes_bulk(
        self,
        session: aiohttp.ClientSession,
        business_days: list[str],
        batch_size: int = 30,
    ) -> pl.DataFrame:
        """
        Fetch daily quotes for multiple dates in parallel batches.

        Args:
            session: aiohttp client session
            business_days: List of business days (YYYY-MM-DD)
            batch_size: Number of dates to fetch in parallel

        Returns:
            Combined Polars DataFrame
        """
        import asyncio

        all_quotes: list[pl.DataFrame] = []

        for i in range(0, len(business_days), batch_size):
            batch_days = business_days[i : i + batch_size]

            tasks = [
                self.fetch_daily_quotes_for_date(session, date)
                for date in batch_days
            ]
            results = await asyncio.gather(*tasks)

            for df in results:
                if not df.is_empty():
                    all_quotes.append(df)

        if not all_quotes:
            return pl.DataFrame()

        return pl.concat(all_quotes, how="diagonal_relaxed")

    async def fetch_topix_data(
        self,
        session: aiohttp.ClientSession,
        from_date: str,
        to_date: str,
    ) -> pl.DataFrame:
        """
        Fetch TOPIX index time series with pagination.

        Args:
            session: aiohttp client session
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)

        Returns:
            Polars DataFrame with TOPIX data
        """
        if not self.client.id_token:
            raise RuntimeError("authenticate() must be called first")

        url = f"{self.base_url}/indices/topix"
        headers = {"Authorization": f"Bearer {self.client.id_token}"}

        all_rows: list[dict] = []
        pagination_key: str | None = None

        while True:
            params = {"from": from_date, "to": to_date}
            if pagination_key:
                params["pagination_key"] = pagination_key

            status, data = await self.client._request_json(
                session,
                "GET",
                url,
                label="topix",
                params=params,
                headers=headers,
            )

            if status != 200 or not isinstance(data, dict):
                break

            rows = data.get("topix", [])
            if rows:
                all_rows.extend(rows)

            pagination_key = data.get("pagination_key")
            if not pagination_key:
                break

        if not all_rows:
            return pl.DataFrame()

        # Build DataFrame with proper types
        keys: set[str] = set()
        for r in all_rows:
            keys.update(r.keys())

        schema = {k: pl.Utf8 for k in keys}
        df = pl.DataFrame(all_rows, schema=schema, orient="row")

        # Normalize types
        if "Date" in df.columns:
            df = df.with_columns(
                pl.col("Date").str.strptime(pl.Date, strict=False)
            )
        if "Close" in df.columns:
            df = df.with_columns(
                pl.col("Close").cast(pl.Float64)
            )

        return df.sort("Date") if "Date" in df.columns else df

    async def fetch_indices_ohlc(
        self,
        session: aiohttp.ClientSession,
        from_date: str,
        to_date: str,
        codes: list[str] | None = None,
    ) -> pl.DataFrame:
        """
        Fetch multiple index OHLC series.

        Args:
            session: aiohttp client session
            from_date: Start date (YYYY-MM-DD or YYYYMMDD)
            to_date: End date (YYYY-MM-DD or YYYYMMDD)
            codes: Optional list of index codes

        Returns:
            Polars DataFrame with OHLC data
        """
        import asyncio

        if not self.client.id_token:
            raise RuntimeError("authenticate() must be called first")

        headers = {"Authorization": f"Bearer {self.client.id_token}"}
        base_url = f"{self.base_url}/indices"

        async def _fetch_for_code(code: str) -> list[dict]:
            rows: list[dict] = []
            pagination_key: str | None = None

            while True:
                params = {"code": code, "from": from_date, "to": to_date}
                if pagination_key:
                    params["pagination_key"] = pagination_key

                status, data = await self.client._request_json(
                    session,
                    "GET",
                    base_url,
                    label=f"indices:{code}",
                    params=params,
                    headers=headers,
                )

                if status != 200 or not isinstance(data, dict):
                    break

                items = data.get("indices") or data.get("data") or []
                if items:
                    for it in items:
                        it.setdefault("Code", code)
                    rows.extend(items)

                pagination_key = data.get("pagination_key")
                if not pagination_key:
                    break

            return rows

        all_rows: list[dict] = []

        if codes:
            # Fetch by code concurrently
            async def _runner(code: str) -> None:
                try:
                    rows = await _fetch_for_code(code)
                    if rows:
                        all_rows.extend(rows)
                except Exception:
                    pass

            tasks = [asyncio.create_task(_runner(c)) for c in codes]
            if tasks:
                await asyncio.gather(*tasks)

        if not all_rows:
            return pl.DataFrame()

        # Sanitize sentinel values
        for row in all_rows:
            for k, v in list(row.items()):
                if isinstance(v, str) and v.strip().lower() in {"-", "*", "", "null"}:
                    row[k] = None

        df = pl.DataFrame(all_rows)
        cols = df.columns

        # Normalize types
        def _dtcol(name: str) -> pl.Expr:
            if name not in cols:
                return pl.lit(None, dtype=pl.Date).alias(name)
            return pl.col(name).str.strptime(pl.Date, strict=False).alias(name)

        def _fcol(name: str) -> pl.Expr:
            if name not in cols:
                return pl.lit(None, dtype=pl.Float64).alias(name)
            return pl.col(name).cast(pl.Float64, strict=False).alias(name)

        out = df.with_columns(
            [
                pl.col("Code").cast(pl.Utf8)
                if "Code" in cols
                else pl.lit(None, dtype=pl.Utf8).alias("Code"),
                _dtcol("Date"),
                _fcol("Open"),
                _fcol("High"),
                _fcol("Low"),
                _fcol("Close"),
            ]
        )

        if {"Code", "Date"}.issubset(out.columns):
            return out.sort(["Code", "Date"])

        return out
