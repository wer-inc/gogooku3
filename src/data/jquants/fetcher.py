#!/usr/bin/env python3
"""
JQuants API Async Fetcher
Moved from scripts/_archive/run_pipeline.py for proper organization
"""

import os
import asyncio
import aiohttp
import polars as pl
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class JQuantsAsyncFetcher:
    """Asynchronous JQuants API fetcher with high concurrency."""

    def __init__(self, email: str, password: str, max_concurrent: Optional[int] = None):
        self.email = email
        self.password = password
        self.base_url = "https://api.jquants.com/v1"
        self.id_token = None
        # 有料プラン向け設定
        self.max_concurrent = max_concurrent or int(
            os.getenv("MAX_CONCURRENT_FETCH", "75")  # 有料プラン向け
        )
        self.semaphore = asyncio.Semaphore(self.max_concurrent)

    async def authenticate(self, session: aiohttp.ClientSession):
        """Authenticate with JQuants API."""
        # Get refresh token
        auth_url = f"{self.base_url}/token/auth_user"
        auth_payload = {"mailaddress": self.email, "password": self.password}

        async with session.post(auth_url, json=auth_payload) as response:
            if response.status != 200:
                text = await response.text()
                raise Exception(f"Auth failed: {response.status} - {text}")
            data = await response.json()
            refresh_token = data["refreshToken"]

        # Get ID token
        refresh_url = f"{self.base_url}/token/auth_refresh"
        params = {"refreshtoken": refresh_token}

        async with session.post(refresh_url, params=params) as response:
            if response.status != 200:
                raise Exception(f"Failed to get ID token: {response.status}")
            data = await response.json()
            self.id_token = data["idToken"]

        logger.info("✅ JQuants authentication successful")

    async def get_listed_info(self, session: aiohttp.ClientSession, date: Optional[str] = None) -> pl.DataFrame:
        """Get listed company information with Market Code filtering."""
        url = f"{self.base_url}/listed/info"
        headers = {"Authorization": f"Bearer {self.id_token}"}
        params = {}
        if date:
            params["date"] = date

        async with session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                df = pl.DataFrame(data.get("info", []))
                # Market Codeフィルタリング（8市場のみ）
                if not df.is_empty():
                    try:
                        from components.market_code_filter import MarketCodeFilter
                        df = MarketCodeFilter.filter_stocks(df)
                    except ImportError:
                        pass
                return df
            return pl.DataFrame()

    async def fetch_price_batch(
        self, session: aiohttp.ClientSession, code: str, from_date: str, to_date: str
    ) -> Optional[pl.DataFrame]:
        """Fetch price data for a single stock."""
        async with self.semaphore:
            url = f"{self.base_url}/prices/daily_quotes"
            headers = {"Authorization": f"Bearer {self.id_token}"}
            params = {"code": code, "from": from_date, "to": to_date}

            try:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        quotes = data.get("daily_quotes", [])
                        if quotes:
                            df = pl.DataFrame(quotes)
                            df = df.with_columns(pl.lit(code).alias("Code"))
                            return df
            except Exception as e:
                logger.warning(f"Failed to fetch {code}: {e}")

            return None

    async def fetch_all_prices(
        self,
        session: aiohttp.ClientSession,
        codes: List[str],
        from_date: str,
        to_date: str,
    ) -> pl.DataFrame:
        """Fetch price data for all stocks concurrently."""
        logger.info(f"Fetching price data for {len(codes)} stocks...")

        tasks = [
            self.fetch_price_batch(session, code, from_date, to_date) for code in codes
        ]

        results = await asyncio.gather(*tasks)

        # Filter out None results and concatenate
        valid_results = [df for df in results if df is not None and not df.is_empty()]

        if valid_results:
            combined = pl.concat(valid_results, how="vertical")
            logger.info(
                f"✅ Fetched {len(combined)} price records for {combined['Code'].n_unique()} stocks"
            )
            return combined

        return pl.DataFrame()

    async def get_trades_spec(
        self, session: aiohttp.ClientSession, from_date: str, to_date: str
    ) -> pl.DataFrame:
        """Get trades specification data."""
        url = f"{self.base_url}/markets/trades_spec"
        headers = {"Authorization": f"Bearer {self.id_token}"}
        params = {"from": from_date, "to": to_date}

        async with session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                trades = data.get("trades_spec", [])
                if trades:
                    return pl.DataFrame(trades)
            return pl.DataFrame()

    async def fetch_topix_data(
        self, session: aiohttp.ClientSession, from_date: str, to_date: str
    ) -> pl.DataFrame:
        """Fetch TOPIX index data."""
        url = f"{self.base_url}/indices/topix"
        headers = {"Authorization": f"Bearer {self.id_token}"}

        all_data = []
        pagination_key = None

        while True:
            params = {"from": from_date, "to": to_date}
            if pagination_key:
                params["pagination_key"] = pagination_key

            try:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to fetch TOPIX: {response.status}")
                        break

                    data = await response.json()
                    topix_data = data.get("topix", [])

                    if topix_data:
                        all_data.extend(topix_data)

                    # Check for pagination
                    pagination_key = data.get("pagination_key")
                    if not pagination_key:
                        break

            except Exception as e:
                logger.error(f"Error fetching TOPIX: {e}")
                break

        if all_data:
            df = pl.DataFrame(all_data)
            # Ensure proper column names and types
            if "Date" in df.columns:
                df = df.with_columns(
                    pl.col("Date").str.strptime(
                        pl.Date, format="%Y-%m-%d", strict=False
                    )
                )
            if "Close" in df.columns:
                df = df.with_columns(pl.col("Close").cast(pl.Float64))

            logger.info(f"✅ Fetched {len(df)} TOPIX records")
            return df.sort("Date")

        return pl.DataFrame()
