from __future__ import annotations

"""
Asynchronous J-Quants API fetcher (extracted from legacy pipeline).

Minimal surface used by current pipelines:
 - authenticate(session)
 - get_trades_spec(session, from_date, to_date) -> pl.DataFrame
 - get_listed_info(session, date=None) -> pl.DataFrame
 - fetch_topix_data(session, from_date, to_date) -> pl.DataFrame
 - get_futures_daily(session, from_date, to_date) -> pl.DataFrame

Notes:
 - Optionally filters listed_info via scripts.components.market_code_filter if available.
 - No dependency on scripts/_archive; safe when _archive is removed.
"""

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
        self, session: aiohttp.ClientSession, date: str | None = None
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
            from scripts.components.market_code_filter import (
                MarketCodeFilter,  # type: ignore
            )

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

    async def get_futures_daily(
        self, session: aiohttp.ClientSession, from_date: str, to_date: str
    ) -> pl.DataFrame:
        """Fetch index futures daily OHLC (derivatives/futures) within a date range.

        Attempts range fetch with pagination; if the API requires per-date queries,
        falls back to iterating by date. Returns normalized DataFrame with the raw
        schema provided by J-Quants (columns used downstream include:
          Date, DerivativesProductCategory, CentralContractMonthFlag,
          EmergencyMarginTriggerDivision, SpecialQuotationDay,
          NightSessionOpen/High/Low/Close, DaySessionOpen/High/Low/Close,
          WholeDayOpen/High/Low/Close, Volume, OpenInterest, SettlementPrice).
        """
        if not self.id_token:
            raise RuntimeError("authenticate() must be called first")

        headers = {"Authorization": f"Bearer {self.id_token}"}
        base_url = f"{self.base_url}/derivatives/futures"

        async def _fetch_range() -> pl.DataFrame:
            rows: list[dict] = []
            pagination_key: str | None = None
            while True:
                params = {"from": from_date, "to": to_date}
                if pagination_key:
                    params["pagination_key"] = pagination_key
                async with session.get(base_url, headers=headers, params=params) as resp:
                    if resp.status != 200:
                        break
                    data = await resp.json()
                # Try common keys
                items = (
                    data.get("futures")
                    or data.get("derivatives")
                    or data.get("data")
                    or []
                )
                if items:
                    rows.extend(items)
                pagination_key = data.get("pagination_key")
                if not pagination_key:
                    break
            return pl.DataFrame(rows) if rows else pl.DataFrame()

        df = await _fetch_range()
        # Fallback: per-day queries (if range not supported)
        if df.is_empty():
            import datetime as _dt

            def _parse(d: str) -> _dt.date:
                return _dt.datetime.strptime(d, "%Y-%m-%d").date() if "-" in d else _dt.datetime.strptime(d, "%Y%m%d").date()

            cur = _parse(from_date)
            end = _parse(to_date)
            rows: list[dict] = []
            while cur <= end:
                date_str = cur.strftime("%Y-%m-%d")
                pagination_key: str | None = None
                while True:
                    params = {"date": date_str}
                    if pagination_key:
                        params["pagination_key"] = pagination_key
                    async with session.get(base_url, headers=headers, params=params) as resp:
                        if resp.status != 200:
                            break
                        data = await resp.json()
                    items = data.get("futures") or data.get("derivatives") or data.get("data") or []
                    if items:
                        rows.extend(items)
                    pagination_key = data.get("pagination_key")
                    if not pagination_key:
                        break
                cur += _dt.timedelta(days=1)
            df = pl.DataFrame(rows) if rows else pl.DataFrame()

        if df.is_empty():
            return df

        # Normalize date columns
        def _dtcol(name: str) -> pl.Expr:
            if name not in df.columns:
                return pl.lit(None, dtype=pl.Date).alias(name)
            col = pl.col(name)
            return (
                pl.when(col.dtype == pl.Utf8)
                .then(col.str.strptime(pl.Date, strict=False))
                .otherwise(col.cast(pl.Date))
                .alias(name)
            )

        out = df.with_columns(
            [
                _dtcol("Date"),
                _dtcol("SpecialQuotationDay"),
            ]
        )
        return out.sort(["Date"]) if "Date" in out.columns else out

    async def get_daily_margin_interest(
        self, session: aiohttp.ClientSession, from_date: str, to_date: str
    ) -> pl.DataFrame:
        """Fetch daily margin interest series with correction handling.

        Returns normalized DataFrame with columns:
          - Code (Utf8)
          - PublishedDate (Date)
          - ApplicationDate (Date)
          - PublishReason (Struct with Restricted/DailyPublication/etc flags)
          - ShortMarginOutstanding, LongMarginOutstanding (Float64)
          - DailyChangeShortMarginOutstanding, DailyChangeLongMarginOutstanding (Float64)
          - ShortMarginOutstandingListedShareRatio, LongMarginOutstandingListedShareRatio (Float64)
          - ShortLongRatio (Float64)
          - ShortNegotiableMarginOutstanding, ShortStandardizedMarginOutstanding (Float64)
          - LongNegotiableMarginOutstanding, LongStandardizedMarginOutstanding (Float64)
          - DailyChange* versions of above (Float64)
          - TSEMarginBorrowingAndLendingRegulationClassification (Utf8)

        Handles corrections by keeping only the latest PublishedDate for each (Code, ApplicationDate).
        """
        if not self.id_token:
            raise RuntimeError("authenticate() must be called first")

        url = f"{self.base_url}/markets/daily_margin_interest"
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
            """Fetch all data for a given date with pagination handling."""
            rows: list[dict] = []
            pagination_key: str | None = None
            while True:
                params = {"date": date_str}
                if pagination_key:
                    params["pagination_key"] = pagination_key

                async with self.semaphore:
                    async with session.get(url, headers=headers, params=params) as resp:
                        if resp.status != 200:
                            break
                        data = await resp.json()

                if isinstance(data, dict):
                    items = data.get("daily_margin_interest", [])
                    if items:
                        rows.extend(items)
                    pagination_key = data.get("pagination_key")
                    if not pagination_key:
                        break
                else:
                    break
            return rows

        # Fetch data for each business day in range
        all_rows: list[dict] = []
        cur = start
        while cur <= end:
            # Only fetch for business days (weekdays)
            if cur.weekday() < 5:
                date_str = cur.strftime("%Y-%m-%d")
                items = await _fetch_for_date(date_str)
                if items:
                    all_rows.extend(items)
            cur += _dt.timedelta(days=1)

        if not all_rows:
            return pl.DataFrame()

        df = pl.DataFrame(all_rows)

        # Normalize dtypes
        cols = df.columns

        # Handle date columns
        def _dtcol(name: str) -> pl.Expr:
            col = pl.col(name)
            return (
                pl.when(col.dtype == pl.Utf8)
                .then(col.str.strptime(pl.Date, strict=False))
                .otherwise(col.cast(pl.Date))
                .alias(name)
            )

        # Handle numeric columns that may contain "-" for missing values
        def _float_col(name: str) -> pl.Expr:
            if name not in cols:
                return pl.lit(None, dtype=pl.Float64).alias(name)
            return (
                pl.when(pl.col(name) == "-")
                .then(None)
                .when(pl.col(name) == "*")
                .then(None)
                .otherwise(pl.col(name))
                .cast(pl.Float64)
                .alias(name)
            )

        out = df.with_columns([
            pl.col("Code").cast(pl.Utf8) if "Code" in cols else pl.lit(None, dtype=pl.Utf8).alias("Code"),
            _dtcol("PublishedDate") if "PublishedDate" in cols else pl.lit(None, dtype=pl.Date).alias("PublishedDate"),
            _dtcol("ApplicationDate") if "ApplicationDate" in cols else pl.lit(None, dtype=pl.Date).alias("ApplicationDate"),
            # Keep PublishReason as struct for now - will be processed in feature engineering
            pl.col("PublishReason") if "PublishReason" in cols else pl.lit(None).alias("PublishReason"),
            # Core margin balances
            _float_col("ShortMarginOutstanding"),
            _float_col("LongMarginOutstanding"),
            _float_col("DailyChangeShortMarginOutstanding"),
            _float_col("DailyChangeLongMarginOutstanding"),
            _float_col("ShortMarginOutstandingListedShareRatio"),
            _float_col("LongMarginOutstandingListedShareRatio"),
            _float_col("ShortLongRatio"),
            # Breakdown by negotiable/standardized
            _float_col("ShortNegotiableMarginOutstanding"),
            _float_col("ShortStandardizedMarginOutstanding"),
            _float_col("LongNegotiableMarginOutstanding"),
            _float_col("LongStandardizedMarginOutstanding"),
            _float_col("DailyChangeShortNegotiableMarginOutstanding"),
            _float_col("DailyChangeShortStandardizedMarginOutstanding"),
            _float_col("DailyChangeLongNegotiableMarginOutstanding"),
            _float_col("DailyChangeLongStandardizedMarginOutstanding"),
            # Regulation classification
            pl.col("TSEMarginBorrowingAndLendingRegulationClassification").cast(pl.Utf8) if "TSEMarginBorrowingAndLendingRegulationClassification" in cols else pl.lit(None, dtype=pl.Utf8).alias("TSEMarginBorrowingAndLendingRegulationClassification"),
        ])

        # Handle corrections: for each (Code, ApplicationDate), keep only the latest PublishedDate
        out = (
            out.filter(pl.col("Code").is_not_null() & pl.col("ApplicationDate").is_not_null())
            .sort(["Code", "ApplicationDate", "PublishedDate"])
            .group_by(["Code", "ApplicationDate"])
            .agg(pl.all().last())  # Keep the latest PublishedDate for each (Code, ApplicationDate)
            .sort(["Code", "ApplicationDate"])
        )

        return out

    async def get_futures_daily(
        self, session: aiohttp.ClientSession, from_date: str, to_date: str
    ) -> pl.DataFrame:
        """
        Fetch derivatives futures daily data from J-Quants API.

        Args:
            session: aiohttp session
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)

        Returns:
            pl.DataFrame with futures daily data
        """
        # Define priority categories for futures (TOPIX, Nikkei225, JPX400, REIT)
        priority_categories = [
            "TOPIXF", "TOPIXMF",           # TOPIX futures
            "NK225F", "NK225MF", "NK225MCF",  # Nikkei225 futures
            "JN400F",                      # JPX400 futures
            "REITF",                       # REIT futures
            "MOTF",                        # Mothers futures (if available)
        ]

        all_futures = []

        for category in priority_categories:
            try:
                print(f"Fetching futures data for category: {category}")

                # Fetch derivatives futures for specific category
                # Using pagination similar to other endpoints
                page = 1
                while True:
                    async with self.semaphore:
                        url = f"{self.base_url}/derivatives/futures"
                        params = {
                            "from": from_date,
                            "to": to_date,
                            "DerivativesProductCategory": category
                        }

                        headers = {"Authorization": f"Bearer {self.id_token}"}
                        async with session.get(url, params=params, headers=headers) as resp:
                            if resp.status == 404:
                                print(f"No data found for category {category}")
                                break
                            elif resp.status != 200:
                                print(f"API error for {category}: {resp.status}")
                                break

                            data = await resp.json()

                            if not data or "derivatives_futures" not in data:
                                break

                            batch = data["derivatives_futures"]
                            if not batch:
                                break

                            # Convert to DataFrame and add category info
                            df_batch = pl.DataFrame(batch)
                            if not df_batch.is_empty():
                                df_batch = df_batch.with_columns([
                                    pl.lit(category).alias("ProductCategory")
                                ])
                                all_futures.append(df_batch)

                            # Check if we have more pages (simple check)
                            if len(batch) < 1000:  # Assume full page is 1000 records
                                break
                            page += 1

                    # Add delay to respect rate limits
                    await asyncio.sleep(0.1)

            except Exception as e:
                print(f"Error fetching futures category {category}: {e}")
                continue

        if not all_futures:
            print("No futures data retrieved")
            return pl.DataFrame()

        # Combine all categories
        df = pl.concat(all_futures, how="vertical")
        print(f"Retrieved {len(df)} futures records")

        # Normalize the data structure
        df = self._normalize_futures_data(df)

        return df

    def _normalize_futures_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Normalize futures data structure and types."""
        if df.is_empty():
            return df

        cols = df.columns

        # Helper for date columns
        def _date_col(name: str) -> pl.Expr:
            if name not in cols:
                return pl.lit(None, dtype=pl.Date).alias(name)
            col = pl.col(name)
            return (
                pl.when(col.dtype == pl.Utf8)
                .then(col.str.strptime(pl.Date, strict=False))
                .otherwise(col.cast(pl.Date))
                .alias(name)
            )

        # Helper for numeric columns that may contain "-" or "*"
        def _float_col(name: str) -> pl.Expr:
            if name not in cols:
                return pl.lit(None, dtype=pl.Float64).alias(name)
            return (
                pl.when(pl.col(name) == "-")
                .then(None)
                .when(pl.col(name) == "*")
                .then(None)
                .when(pl.col(name).is_null())
                .then(None)
                .otherwise(pl.col(name))
                .cast(pl.Float64)
                .alias(name)
            )

        # Helper for integer columns
        def _int_col(name: str) -> pl.Expr:
            if name not in cols:
                return pl.lit(None, dtype=pl.Int64).alias(name)
            return (
                pl.when(pl.col(name) == "-")
                .then(None)
                .when(pl.col(name) == "*")
                .then(None)
                .when(pl.col(name).is_null())
                .then(None)
                .otherwise(pl.col(name))
                .cast(pl.Int64)
                .alias(name)
            )

        # Normalize columns
        normalized = df.with_columns([
            # Basic identification
            pl.col("Code").cast(pl.Utf8) if "Code" in cols else pl.lit(None, dtype=pl.Utf8).alias("Code"),
            _date_col("Date"),
            pl.col("ProductCategory").cast(pl.Utf8) if "ProductCategory" in cols else pl.lit(None, dtype=pl.Utf8).alias("ProductCategory"),

            # Contract details
            pl.col("ContractMonth").cast(pl.Utf8) if "ContractMonth" in cols else pl.lit(None, dtype=pl.Utf8).alias("ContractMonth"),
            pl.col("CentralContractMonthFlag").cast(pl.Utf8) if "CentralContractMonthFlag" in cols else pl.lit(None, dtype=pl.Utf8).alias("CentralContractMonthFlag"),

            # OHLC prices
            _float_col("Open"),
            _float_col("High"),
            _float_col("Low"),
            _float_col("Close"),

            # Trading data
            _int_col("Volume"),
            _int_col("OpenInterest"),
            _float_col("TurnoverValue"),

            # Emergency margin data
            pl.col("EmergencyMarginTriggerDivision").cast(pl.Utf8) if "EmergencyMarginTriggerDivision" in cols else pl.lit(None, dtype=pl.Utf8).alias("EmergencyMarginTriggerDivision"),
            _float_col("EmergencyMarginValue"),

            # Session data (night/day)
            _float_col("NightSessionOpen"),
            _float_col("NightSessionHigh"),
            _float_col("NightSessionLow"),
            _float_col("NightSessionClose"),
            _int_col("NightSessionVolume"),
            _float_col("NightSessionTurnoverValue"),
        ])

        # Handle emergency margin duplicates: prefer 002 (clearing) over 001 (trigger)
        # Create emergency margin flag before deduplication
        normalized = normalized.with_columns([
            (pl.col("EmergencyMarginTriggerDivision") == "001").cast(pl.Int8).alias("emergency_margin_triggered")
        ])

        # Deduplicate by keeping 002 (clearing) records when both 001 and 002 exist
        deduped = (
            normalized
            .sort(["Code", "Date", "ContractMonth", "EmergencyMarginTriggerDivision"])
            .group_by(["Code", "Date", "ContractMonth"])
            .agg([
                # Keep 002 if exists, otherwise 001 or null
                pl.col("EmergencyMarginTriggerDivision").filter(pl.col("EmergencyMarginTriggerDivision") == "002").first().alias("EmergencyMarginTriggerDivision_tmp"),
                pl.col("emergency_margin_triggered").max().alias("emergency_margin_triggered"), # 1 if any 001 exists
                # For other columns, take last non-null value
                pl.all().exclude(["EmergencyMarginTriggerDivision", "emergency_margin_triggered"]).last()
            ])
            .with_columns([
                # Use 002 if available, otherwise use original
                pl.when(pl.col("EmergencyMarginTriggerDivision_tmp").is_not_null())
                .then(pl.col("EmergencyMarginTriggerDivision_tmp"))
                .otherwise(pl.col("EmergencyMarginTriggerDivision"))
                .alias("EmergencyMarginTriggerDivision")
            ])
            .drop("EmergencyMarginTriggerDivision_tmp")
            .sort(["Code", "Date", "ContractMonth"])
        )

        return deduped

    async def get_index_option(
        self, session: aiohttp.ClientSession, from_date: str, to_date: str
    ) -> pl.DataFrame:
        """Fetch Nikkei225 index option daily data (/option/index_option) by date.

        Endpoint requires a date parameter; this iterates dates and handles pagination.
        Returns a normalized DataFrame with consistent types including Date, Code, price fields,
        session fields, IV, theoretical, OI/Volume, ContractMonth, StrikePrice, EmergencyMarginTriggerDivision, etc.
        """
        if not self.id_token:
            raise RuntimeError("authenticate() must be called first")

        headers = {"Authorization": f"Bearer {self.id_token}"}
        base_url = f"{self.base_url}/option/index_option"

        import datetime as _dt

        def _parse(d: str) -> _dt.date:
            return _dt.datetime.strptime(d, "%Y-%m-%d").date() if "-" in d else _dt.datetime.strptime(d, "%Y%m%d").date()

        start = _parse(from_date)
        end = _parse(to_date)
        rows: list[dict] = []

        cur = start
        while cur <= end:
            date_str = cur.strftime("%Y-%m-%d")
            pagination_key: str | None = None
            while True:
                params = {"date": date_str}
                if pagination_key:
                    params["pagination_key"] = pagination_key
                async with session.get(base_url, headers=headers, params=params) as resp:
                    if resp.status == 404:
                        break
                    if resp.status != 200:
                        break
                    data = await resp.json()
                items = data.get("index_option") or data.get("data") or []
                if items:
                    rows.extend(items)
                pagination_key = data.get("pagination_key")
                if not pagination_key:
                    break
            cur += _dt.timedelta(days=1)

        if not rows:
            return pl.DataFrame()

        df = pl.DataFrame(rows)
        return self._normalize_index_option_data(df)

    def _normalize_index_option_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Normalize index option data structure and types for downstream processing."""
        if df.is_empty():
            return df

        # Rename volume(only auction)
        if "Volume(OnlyAuction)" in df.columns and "VolumeOnlyAuction" not in df.columns:
            df = df.rename({"Volume(OnlyAuction)": "VolumeOnlyAuction"})

        def _dtcol(name: str) -> pl.Expr:
            if name not in df.columns:
                return pl.lit(None, dtype=pl.Date).alias(name)
            c = pl.col(name)
            return (
                pl.when(c.dtype == pl.Utf8)
                .then(c.str.strptime(pl.Date, strict=False))
                .otherwise(c.cast(pl.Date))
                .alias(name)
            )

        out = df.with_columns(
            [
                _dtcol("Date"),
                _dtcol("LastTradingDay"),
                _dtcol("SpecialQuotationDay"),
                pl.col("Code").cast(pl.Utf8) if "Code" in df.columns else pl.lit(None, dtype=pl.Utf8).alias("Code"),
                (pl.col("ContractMonth").cast(pl.Utf8) if "ContractMonth" in df.columns else pl.lit(None, dtype=pl.Utf8)).alias("ContractMonth"),
                (pl.col("EmergencyMarginTriggerDivision").cast(pl.Utf8) if "EmergencyMarginTriggerDivision" in df.columns else pl.lit(None, dtype=pl.Utf8)).alias("EmergencyMarginTriggerDivision"),
                (pl.col("PutCallDivision").cast(pl.Utf8) if "PutCallDivision" in df.columns else pl.lit(None, dtype=pl.Utf8)).alias("PutCallDivision"),
            ]
        )

        def _num(col: str) -> pl.Expr:
            return pl.when(pl.col(col) == "").then(None).otherwise(pl.col(col)).cast(pl.Float64)

        for c in [
            "WholeDayOpen",
            "WholeDayHigh",
            "WholeDayLow",
            "WholeDayClose",
            "NightSessionOpen",
            "NightSessionHigh",
            "NightSessionLow",
            "NightSessionClose",
            "DaySessionOpen",
            "DaySessionHigh",
            "DaySessionLow",
            "DaySessionClose",
            "Volume",
            "OpenInterest",
            "TurnoverValue",
            "SettlementPrice",
            "TheoreticalPrice",
            "BaseVolatility",
            "ImpliedVolatility",
            "UnderlyingPrice",
            "InterestRate",
            "StrikePrice",
            "VolumeOnlyAuction",
        ]:
            if c in out.columns:
                out = out.with_columns(_num(c).alias(c))

        # Sort for determinism
        out = out.sort(["Date", "Code", "EmergencyMarginTriggerDivision"]) if "Date" in out.columns else out
        return out

    async def get_short_selling(
        self, session: aiohttp.ClientSession, from_date: str, to_date: str
    ) -> pl.DataFrame:
        """
        Fetch short selling data from J-Quants API.

        Args:
            session: aiohttp session
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)

        Returns:
            pl.DataFrame with short selling ratio data
        """
        all_data = []

        async with self.semaphore:
            url = f"{self.base_url}/markets/short_selling"
            params = {"from": from_date, "to": to_date}
            headers = {"Authorization": f"Bearer {self.id_token}"}

            try:
                async with session.get(url, params=params, headers=headers) as resp:
                    if resp.status == 404:
                        print("No short selling data found")
                        return pl.DataFrame()
                    elif resp.status != 200:
                        print(f"API error for short selling: {resp.status}")
                        return pl.DataFrame()

                    data = await resp.json()

                    if not data or "short_selling" not in data:
                        return pl.DataFrame()

                    batch = data["short_selling"]
                    if batch:
                        all_data.extend(batch)

            except Exception as e:
                print(f"Error fetching short selling data: {e}")
                return pl.DataFrame()

        if not all_data:
            return pl.DataFrame()

        df = pl.DataFrame(all_data)
        df = self._normalize_short_selling_data(df)

        print(f"Retrieved {len(df)} short selling records")
        return df

    async def get_short_selling_positions(
        self, session: aiohttp.ClientSession, from_date: str, to_date: str
    ) -> pl.DataFrame:
        """
        Fetch short selling positions data from J-Quants API.

        Args:
            session: aiohttp session
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)

        Returns:
            pl.DataFrame with short selling positions data
        """
        all_data = []

        async with self.semaphore:
            url = f"{self.base_url}/markets/short_selling_positions"
            params = {"from": from_date, "to": to_date}
            headers = {"Authorization": f"Bearer {self.id_token}"}

            try:
                async with session.get(url, params=params, headers=headers) as resp:
                    if resp.status == 404:
                        print("No short selling positions data found")
                        return pl.DataFrame()
                    elif resp.status != 200:
                        print(f"API error for short selling positions: {resp.status}")
                        return pl.DataFrame()

                    data = await resp.json()

                    if not data or "short_selling_positions" not in data:
                        return pl.DataFrame()

                    batch = data["short_selling_positions"]
                    if batch:
                        all_data.extend(batch)

            except Exception as e:
                print(f"Error fetching short selling positions data: {e}")
                return pl.DataFrame()

        if not all_data:
            return pl.DataFrame()

        df = pl.DataFrame(all_data)
        df = self._normalize_short_selling_positions_data(df)

        print(f"Retrieved {len(df)} short selling positions records")
        return df

    async def get_earnings_announcements(
        self, session: aiohttp.ClientSession, from_date: str, to_date: str
    ) -> pl.DataFrame:
        """
        Fetch earnings announcement schedule data from J-Quants API.

        Args:
            session: aiohttp session
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)

        Returns:
            pl.DataFrame with earnings announcement schedule data
        """
        all_data = []

        async with self.semaphore:
            url = f"{self.base_url}/fins/announcement"
            params = {"from": from_date, "to": to_date}
            headers = {"Authorization": f"Bearer {self.id_token}"}

            try:
                async with session.get(url, params=params, headers=headers) as resp:
                    if resp.status == 404:
                        print(f"Earnings announcements endpoint not found: {url}")
                        return pl.DataFrame()

                    if resp.status == 429:
                        retry_after = int(resp.headers.get("Retry-After", 30))
                        print(f"Rate limited. Waiting {retry_after} seconds...")
                        await asyncio.sleep(retry_after)
                        return await self.get_earnings_announcements(session, from_date, to_date)

                    if resp.status != 200:
                        print(f"Error fetching earnings announcements: {resp.status}")
                        return pl.DataFrame()

                    data = await resp.json()
                    if "announcement" in data:
                        all_data.extend(data["announcement"])
                    elif isinstance(data, list):
                        all_data.extend(data)

            except asyncio.TimeoutError:
                print(f"Timeout fetching earnings announcements for {from_date} to {to_date}")
                return pl.DataFrame()
            except Exception as e:
                print(f"Error fetching earnings announcements: {e}")
                return pl.DataFrame()

        if not all_data:
            return pl.DataFrame()

        df = pl.DataFrame(all_data)
        df = self._normalize_earnings_announcement_data(df)

        print(f"Retrieved {len(df)} earnings announcement records")
        return df

    def _normalize_earnings_announcement_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Normalize earnings announcement data with consistent column names and types.

        Args:
            df: Raw earnings announcement DataFrame from API

        Returns:
            Normalized DataFrame with standard column names
        """
        if df.is_empty():
            return df

        # Standardize column names
        column_mapping = {
            "LocalCode": "Code",
            "Code": "Code",
            "Date": "Date",
            "AnnouncementDate": "AnnouncementDate",
            "CompanyName": "CompanyName",
            "FiscalYear": "FiscalYear",
            "FiscalQuarter": "FiscalQuarter",
            "AnnouncementTime": "AnnouncementTime",
        }

        # Rename columns if they exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and old_name != new_name:
                df = df.rename({old_name: new_name})

        # Ensure essential columns exist with proper types
        if "Code" not in df.columns:
            print("Warning: No Code column in earnings announcement data")
            return pl.DataFrame()

        # Convert date columns to proper format
        date_columns = ["Date", "AnnouncementDate"]
        for col in date_columns:
            if col in df.columns:
                df = df.with_columns([
                    pl.col(col).str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias(col)
                ])

        # Ensure Code is string
        df = df.with_columns([
            pl.col("Code").cast(pl.Utf8)
        ])

        # Sort by announcement date and code
        df = df.sort(["AnnouncementDate", "Code"])

        return df

    async def get_sector_short_selling(
        self, session: aiohttp.ClientSession, from_date: str, to_date: str
    ) -> pl.DataFrame:
        """
        Fetch sector-wise short selling data from J-Quants API (/markets/short_selling).

        Args:
            session: aiohttp session
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)

        Returns:
            pl.DataFrame with sector short selling data
        """
        all_data = []

        async with self.semaphore:
            url = f"{self.base_url}/markets/short_selling"
            params = {"from": from_date, "to": to_date}
            headers = {"Authorization": f"Bearer {self.id_token}"}

            try:
                async with session.get(url, params=params, headers=headers) as resp:
                    if resp.status == 404:
                        print(f"Sector short selling endpoint not found: {url}")
                        return pl.DataFrame()

                    if resp.status == 429:
                        retry_after = int(resp.headers.get("Retry-After", 30))
                        print(f"Rate limited. Waiting {retry_after} seconds...")
                        await asyncio.sleep(retry_after)
                        return await self.get_sector_short_selling(session, from_date, to_date)

                    if resp.status != 200:
                        print(f"Error fetching sector short selling: {resp.status}")
                        return pl.DataFrame()

                    data = await resp.json()
                    if "short_selling" in data:
                        all_data.extend(data["short_selling"])
                    elif isinstance(data, list):
                        all_data.extend(data)

            except asyncio.TimeoutError:
                print(f"Timeout fetching sector short selling for {from_date} to {to_date}")
                return pl.DataFrame()
            except Exception as e:
                print(f"Error fetching sector short selling: {e}")
                return pl.DataFrame()

        if not all_data:
            return pl.DataFrame()

        df = pl.DataFrame(all_data)
        df = self._normalize_sector_short_selling_data(df)

        print(f"Retrieved {len(df)} sector short selling records")
        return df

    def _normalize_sector_short_selling_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Normalize sector short selling data with consistent column names and types.

        Args:
            df: Raw sector short selling DataFrame from API

        Returns:
            Normalized DataFrame with standard column names
        """
        if df.is_empty():
            return df

        # Standardize column names
        column_mapping = {
            "Date": "Date",
            "Sector33Code": "Sector33Code",
            "SellingExcludingShortSellingTurnoverValue": "SellingExcludingShortSellingTurnoverValue",
            "ShortSellingWithRestrictionsTurnoverValue": "ShortSellingWithRestrictionsTurnoverValue",
            "ShortSellingWithoutRestrictionsTurnoverValue": "ShortSellingWithoutRestrictionsTurnoverValue",
        }

        # Rename columns if they exist with different names
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and old_name != new_name:
                df = df.rename({old_name: new_name})

        # Ensure essential columns exist
        required_cols = ["Date", "Sector33Code", "SellingExcludingShortSellingTurnoverValue",
                        "ShortSellingWithRestrictionsTurnoverValue", "ShortSellingWithoutRestrictionsTurnoverValue"]

        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns in sector short selling data: {missing_cols}")
            return pl.DataFrame()

        # Convert date column to proper format
        df = df.with_columns([
            pl.col("Date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("Date")
        ])

        # Ensure proper types
        df = df.with_columns([
            pl.col("Sector33Code").cast(pl.Utf8),
            pl.col("SellingExcludingShortSellingTurnoverValue").cast(pl.Float64),
            pl.col("ShortSellingWithRestrictionsTurnoverValue").cast(pl.Float64),
            pl.col("ShortSellingWithoutRestrictionsTurnoverValue").cast(pl.Float64),
        ])

        # Sort by date and sector
        df = df.sort(["Date", "Sector33Code"])

        return df

    def _normalize_short_selling_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Normalize short selling data structure and types."""
        if df.is_empty():
            return df

        cols = df.columns

        # Helper for date columns
        def _date_col(name: str) -> pl.Expr:
            if name not in cols:
                return pl.lit(None, dtype=pl.Date).alias(name)
            col = pl.col(name)
            return (
                pl.when(col.dtype == pl.Utf8)
                .then(col.str.strptime(pl.Date, strict=False))
                .otherwise(col.cast(pl.Date))
                .alias(name)
            )

        # Helper for float columns that may contain "-" or null
        def _float_col(name: str) -> pl.Expr:
            if name not in cols:
                return pl.lit(None, dtype=pl.Float64).alias(name)
            return (
                pl.when(pl.col(name) == "-")
                .then(None)
                .when(pl.col(name).is_null())
                .then(None)
                .otherwise(pl.col(name))
                .cast(pl.Float64)
                .alias(name)
            )

        # Helper for integer columns
        def _int_col(name: str) -> pl.Expr:
            if name not in cols:
                return pl.lit(None, dtype=pl.Int64).alias(name)
            return (
                pl.when(pl.col(name) == "-")
                .then(None)
                .when(pl.col(name).is_null())
                .then(None)
                .otherwise(pl.col(name))
                .cast(pl.Int64)
                .alias(name)
            )

        # Normalize columns
        normalized = df.with_columns([
            # Basic identification
            pl.col("Code").cast(pl.Utf8) if "Code" in cols else pl.lit(None, dtype=pl.Utf8).alias("Code"),
            _date_col("Date"),
            _date_col("PublishedDate"),

            # Short selling ratio and volume
            _float_col("ShortSellingRatio"),
            _int_col("ShortSellingVolume"),
            _int_col("TotalVolume"),

            # Section information
            pl.col("Section").cast(pl.Utf8) if "Section" in cols else pl.lit(None, dtype=pl.Utf8).alias("Section"),
        ])

        # Remove duplicates by (Code, Date) keeping latest PublishedDate
        deduped = (
            normalized
            .filter(pl.col("Code").is_not_null() & pl.col("Date").is_not_null())
            .sort(["Code", "Date", "PublishedDate"])
            .group_by(["Code", "Date"])
            .agg(pl.all().last())
            .sort(["Code", "Date"])
        )

        return deduped

    def _normalize_short_selling_positions_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Normalize short selling positions data structure and types."""
        if df.is_empty():
            return df

        cols = df.columns

        # Helper for date columns
        def _date_col(name: str) -> pl.Expr:
            if name not in cols:
                return pl.lit(None, dtype=pl.Date).alias(name)
            col = pl.col(name)
            return (
                pl.when(col.dtype == pl.Utf8)
                .then(col.str.strptime(pl.Date, strict=False))
                .otherwise(col.cast(pl.Date))
                .alias(name)
            )

        # Helper for float columns
        def _float_col(name: str) -> pl.Expr:
            if name not in cols:
                return pl.lit(None, dtype=pl.Float64).alias(name)
            return (
                pl.when(pl.col(name) == "-")
                .then(None)
                .when(pl.col(name).is_null())
                .then(None)
                .otherwise(pl.col(name))
                .cast(pl.Float64)
                .alias(name)
            )

        # Helper for integer columns
        def _int_col(name: str) -> pl.Expr:
            if name not in cols:
                return pl.lit(None, dtype=pl.Int64).alias(name)
            return (
                pl.when(pl.col(name) == "-")
                .then(None)
                .when(pl.col(name).is_null())
                .then(None)
                .otherwise(pl.col(name))
                .cast(pl.Int64)
                .alias(name)
            )

        # Normalize columns
        normalized = df.with_columns([
            # Basic identification
            pl.col("Code").cast(pl.Utf8) if "Code" in cols else pl.lit(None, dtype=pl.Utf8).alias("Code"),
            _date_col("Date"),
            _date_col("PublishedDate"),

            # Short selling positions
            _int_col("ShortSellingBalance"),
            _int_col("ShortSellingBalanceChange"),
            _float_col("ShortSellingBalanceRatio"),

            # Additional position data
            _int_col("LendingBalance"),
            _float_col("LendingBalanceRatio"),

            # Section information
            pl.col("Section").cast(pl.Utf8) if "Section" in cols else pl.lit(None, dtype=pl.Utf8).alias("Section"),
        ])

        # Remove duplicates by (Code, Date) keeping latest PublishedDate
        deduped = (
            normalized
            .filter(pl.col("Code").is_not_null() & pl.col("Date").is_not_null())
            .sort(["Code", "Date", "PublishedDate"])
            .group_by(["Code", "Date"])
            .agg(pl.all().last())
            .sort(["Code", "Date"])
        )

        return deduped
