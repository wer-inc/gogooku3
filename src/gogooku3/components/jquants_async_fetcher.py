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
import datetime as _dt
import logging
import math
import os
import time

from contextlib import nullcontext
import aiohttp
import polars as pl
from typing import Any, Iterable, Tuple


def enforce_code_column_types(df: pl.DataFrame) -> pl.DataFrame:
    """
    Systematically enforce Code column type consistency for all dataframes.

    This fixes the issue: "is_in cannot check for String values in Float64 data"
    by ensuring all Code columns are cast to Utf8 type.

    Args:
        df: Input DataFrame that may have Code column

    Returns:
        DataFrame with Code column cast to pl.Utf8 if Code column exists
    """
    if df.is_empty():
        return df

    cols = df.columns
    if "Code" in cols:
        df = df.with_columns(pl.col("Code").cast(pl.Utf8))

    # Also handle LocalCode if present (used in some data sources)
    if "LocalCode" in cols:
        df = df.with_columns(pl.col("LocalCode").cast(pl.Utf8))

    return df


def clean_join_conflicts(df: pl.DataFrame) -> pl.DataFrame:
    """
    Clean up common column conflicts that occur during join operations.

    This function removes duplicate columns that commonly occur during join_asof
    and join operations, such as Code_right, Date_right, etc.

    Args:
        df: DataFrame that may contain join conflict columns

    Returns:
        DataFrame with conflict columns removed
    """
    if df.is_empty():
        return df

    # Common conflict suffixes from Polars joins
    conflict_suffixes = ["_right", "_left", "_y"]

    # Common base column names that cause conflicts
    base_columns = ["Code", "Date", "Section", "effective_date", "effective_start", "effective_end"]

    columns_to_drop = []

    for base_col in base_columns:
        for suffix in conflict_suffixes:
            conflict_col = f"{base_col}{suffix}"
            if conflict_col in df.columns:
                columns_to_drop.append(conflict_col)

    if columns_to_drop:
        df = df.drop(columns_to_drop)

    return df


def read_parquet_with_consistent_code_types(file_path: str | os.PathLike) -> pl.DataFrame:
    """
    Read parquet file and ensure Code column has consistent Utf8 type.

    This wrapper function prevents type inconsistency issues when loading
    previously saved parquet files that might have Code column as Float64.

    Args:
        file_path: Path to the parquet file

    Returns:
        DataFrame with Code column cast to pl.Utf8 if Code column exists

    Usage:
        # Instead of: df = pl.read_parquet(path)
        df = read_parquet_with_consistent_code_types(path)
    """
    df = pl.read_parquet(file_path)
    return enforce_code_column_types(df)


class JQuantsAsyncFetcher:
    """Asynchronous J-Quants API fetcher with basic pagination handling."""

    def __init__(self, email: str, password: str, max_concurrent: int | None = None):
        self.email = email
        self.password = password
        self.base_url = "https://api.jquants.com/v1"
        self.id_token: str | None = None
        self.max_concurrent = max_concurrent or int(os.getenv("MAX_CONCURRENT_FETCH", 32))
        self.semaphore = asyncio.Semaphore(self.max_concurrent)

        # Adaptive throttling configuration/state
        self._current_concurrency = self.max_concurrent
        self._max_concurrency_ceiling = self.max_concurrent
        self._min_concurrency = max(1, int(os.getenv("JQUANTS_MIN_CONCURRENCY", "4")))
        self._min_concurrency = min(self._min_concurrency, self._max_concurrency_ceiling)
        self._throttle_backoff = float(os.getenv("JQUANTS_THROTTLE_BACKOFF", "0.6"))
        self._throttle_sleep_seconds = float(os.getenv("JQUANTS_THROTTLE_SLEEP", "30"))
        self._recovery_step = int(os.getenv("JQUANTS_THROTTLE_STEP", "2"))
        self._success_threshold = int(os.getenv("JQUANTS_THROTTLE_RECOVERY_SUCCESS", "180"))
        self._retry_statuses: Tuple[int, ...] = (429, 503)
        self._success_streak = 0
        self._throttle_hits = 0
        self._throttle_recoveries = 0
        self._throttle_history: list[dict[str, Any]] = []
        self._logger = logging.getLogger(__name__)

    @staticmethod
    def _format_date_param(date_str: str | None) -> str | None:
        """Normalize date tokens to the YYYYMMDD format required by J-Quants."""

        if date_str is None:
            return None

        value = date_str.strip()
        if not value:
            return value

        if len(value) == 8 and value.isdigit():
            return value

        if len(value) == 10 and value.count("-") == 2:
            parsed = _dt.datetime.strptime(value, "%Y-%m-%d")
            return parsed.strftime("%Y%m%d")

        raise ValueError(f"Unsupported date format for J-Quants request: '{date_str}'")

    async def _ensure_session_health(self, session: aiohttp.ClientSession) -> bool:
        """
        Check if session is healthy and can be used for API calls.

        Returns:
            True if session is healthy, False if session is closed/unusable
        """
        try:
            return not session.closed
        except Exception:
            return False

    def _apply_concurrency(self, new_limit: int, *, reason: str) -> None:
        new_limit = max(self._min_concurrency, min(self._max_concurrency_ceiling, new_limit))
        if new_limit == self._current_concurrency:
            return
        self._current_concurrency = new_limit
        self.semaphore = asyncio.Semaphore(new_limit)
        self._logger.info("Adjusted JQuants concurrency → %s (reason=%s)", new_limit, reason)

    def _record_success(self) -> None:
        self._success_streak += 1
        if (
            self._success_streak >= self._success_threshold
            and self._current_concurrency < self._max_concurrency_ceiling
        ):
            new_limit = min(
                self._max_concurrency_ceiling,
                self._current_concurrency + max(1, self._recovery_step),
            )
            if new_limit > self._current_concurrency:
                self._throttle_recoveries += 1
                self._apply_concurrency(new_limit, reason="recovery")
                self._success_streak = 0

    async def _handle_throttle(
        self,
        label: str,
        resp: aiohttp.ClientResponse,
    ) -> None:
        status = resp.status
        self._throttle_hits += 1
        self._success_streak = 0

        retry_after = resp.headers.get("Retry-After")
        delay = self._throttle_sleep_seconds
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except (TypeError, ValueError):
                pass

        scaled = max(1, math.floor(self._current_concurrency * self._throttle_backoff))
        if scaled < self._current_concurrency:
            self._apply_concurrency(scaled, reason=f"throttle:{label}:{status}")

        self._throttle_history.append(
            {
                "label": label,
                "status": status,
                "delay": delay,
                "concurrency": self._current_concurrency,
                "timestamp": time.time(),
            }
        )
        if len(self._throttle_history) > 64:
            self._throttle_history.pop(0)

        self._logger.warning(
            "JQuants throttle detected (%s, status=%s). Sleeping %.1fs.",
            label,
            status,
            delay,
        )
        await asyncio.sleep(delay)

    async def _request_json(
        self,
        session: aiohttp.ClientSession,
        method: str,
        url: str,
        *,
        label: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        json_payload: Any = None,
        data: Any = None,
        expected_statuses: Iterable[int] = (200,),
        decode_json: bool = True,
        use_semaphore: bool = True,
    ) -> Tuple[int, Any]:
        while True:
            cm = self.semaphore if use_semaphore else nullcontext()
            async with cm:
                try:
                    async with session.request(
                        method,
                        url,
                        params=params,
                        headers=headers,
                        json=json_payload,
                        data=data,
                    ) as resp:
                        status = resp.status
                        if status in self._retry_statuses:
                            await self._handle_throttle(label, resp)
                            continue

                        payload: Any = None
                        if decode_json:
                            try:
                                payload = await resp.json()
                            except Exception:
                                payload = None

                        self._record_success()
                        return status, payload
                except aiohttp.ClientError:
                    self._success_streak = 0
                    raise

    def throttle_metrics(self) -> dict[str, Any]:
        return {
            "hits": self._throttle_hits,
            "recoveries": self._throttle_recoveries,
            "current_concurrency": self._current_concurrency,
            "history": list(self._throttle_history),
        }

    async def _safe_api_call(self, session: aiohttp.ClientSession, api_func, *args, **kwargs):
        """
        Safely execute API call with session health checking.

        If session is closed, raises a clear exception that the caller can handle.

        Args:
            session: aiohttp ClientSession
            api_func: The API function to call
            *args, **kwargs: Arguments to pass to api_func

        Returns:
            Result of api_func call

        Raises:
            RuntimeError: If session is closed and cannot be used
        """
        if not await self._ensure_session_health(session):
            raise RuntimeError(
                "Session is closed. The caller should create a new session and re-authenticate."
            )

        return await api_func(session, *args, **kwargs)

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
        """Fetch listed company info with pagination support and type normalization."""
        if not self.id_token:
            raise RuntimeError("authenticate() must be called first")

        url = f"{self.base_url}/listed/info"
        headers = {"Authorization": f"Bearer {self.id_token}"}

        rows: list[dict] = []
        pagination_key: str | None = None
        while True:
            params: dict[str, str] = {}
            if date:
                params["date"] = date
            if pagination_key:
                params["pagination_key"] = pagination_key

            status, data = await self._request_json(
                session,
                "GET",
                url,
                label="listed_info",
                params=params or None,
                headers=headers,
            )

            if status == 404:
                return pl.DataFrame()
            if status != 200 or not isinstance(data, dict):
                self._logger.error(
                    "J-Quants trades_spec failed", extra={"status": status, "body": data}
                )
                raise RuntimeError(f"J-Quants trades_spec request failed with status {status}")

            items = data.get("info") or []
            if items:
                rows.extend(items)

            pagination_key = data.get("pagination_key")
            if not pagination_key:
                break

        if not rows:
            return pl.DataFrame()

        df = pl.DataFrame(rows)

        # Normalize core dtypes
        sentinel = {"", "-", "*", "null", "NULL", "None"}
        if "Date" in df.columns:
            df = df.with_columns(
                pl.col("Date").cast(pl.Utf8).str.strptime(pl.Date, strict=False).alias("Date")
            )
        if "Code" in df.columns:
            df = df.with_columns(pl.col("Code").cast(pl.Utf8))

        str_columns = [
            c
            for c in df.columns
            if c not in {"Date"} and ("Code" in c or c.endswith("Name") or c.endswith("Division") or c.endswith("Category"))
        ]
        if str_columns:
            df = df.with_columns([pl.col(c).cast(pl.Utf8) for c in str_columns])

        numeric_columns = [
            c
            for c in df.columns
            if c not in {"Date", "Code", "MarketCode"}
            and c not in str_columns
        ]
        if numeric_columns:
            df = df.with_columns(
                [
                    pl.when(pl.col(c).cast(pl.Utf8).str.strip_chars().is_in(sentinel))
                    .then(None)
                    .otherwise(pl.col(c))
                    .alias(c)
                    for c in numeric_columns
                ]
            ).with_columns([
                pl.col(c).cast(pl.Float64, strict=False).alias(c) for c in numeric_columns
            ])

        # Optional filter using scripts/components if available
        try:  # pragma: no cover - optional path
            from scripts.components.market_code_filter import (
                MarketCodeFilter,  # type: ignore
            )

            if not df.is_empty():
                df = MarketCodeFilter.filter_stocks(df)
        except Exception:
            pass

        return enforce_code_column_types(df)

    async def get_trades_spec(
        self, session: aiohttp.ClientSession, from_date: str, to_date: str
    ) -> pl.DataFrame:
        """Fetch markets/trades_spec (weekly investor flows)."""
        if not self.id_token:
            raise RuntimeError("authenticate() must be called first")
        url = f"{self.base_url}/markets/trades_spec"
        headers = {"Authorization": f"Bearer {self.id_token}"}
        sentinel = {"", "-", "*", "null", "NULL", "None"}

        rows: list[dict] = []
        pagination_key: str | None = None
        formatted_from = self._format_date_param(from_date)
        formatted_to = self._format_date_param(to_date)
        while True:
            params = {"from": formatted_from, "to": formatted_to}
            if pagination_key:
                params["pagination_key"] = pagination_key

            status, data = await self._request_json(
                session,
                "GET",
                url,
                label="trades_spec",
                params=params,
                headers=headers,
            )

            if status == 404:
                return pl.DataFrame()
            if status != 200 or not isinstance(data, dict):
                body_preview = str(data)[:256]
                self._logger.error(
                    "J-Quants trades_spec failed (status=%s, body=%s)", status, body_preview
                )
                raise RuntimeError(f"J-Quants trades_spec request failed with status {status}")

            items = data.get("trades_spec") or []
            if items:
                rows.extend(items)

            pagination_key = data.get("pagination_key")
            if not pagination_key:
                break

        if not rows:
            return pl.DataFrame()

        df = pl.DataFrame(rows)

        date_cols = [c for c in ("PublishedDate", "StartDate", "EndDate") if c in df.columns]
        if date_cols:
            df = df.with_columns([
                pl.col(c).cast(pl.Utf8).str.strptime(pl.Date, strict=False).alias(c) for c in date_cols
            ])

        if "Section" in df.columns:
            df = df.with_columns(pl.col("Section").cast(pl.Utf8))

        numeric_cols = [
            c
            for c in df.columns
            if c not in date_cols and c != "Section"
        ]
        if numeric_cols:
            df = df.with_columns(
                [
                    pl.when(pl.col(c).cast(pl.Utf8).str.strip_chars().is_in(sentinel))
                    .then(None)
                    .otherwise(pl.col(c))
                    .alias(c)
                    for c in numeric_cols
                ]
            ).with_columns([
                pl.col(c).cast(pl.Float64, strict=False).alias(c) for c in numeric_cols
            ])

        if date_cols:
            df = df.sort(date_cols)
        return df

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
            status, data = await self._request_json(
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
        # Build as Utf8 first to avoid builder dtype conflicts; cast later via _float_col
        keys: set[str] = set()
        for r in all_rows:
            keys.update(r.keys())
        schema = {k: pl.Utf8 for k in keys}
        df = pl.DataFrame(all_rows, schema=schema, orient="row")
        if "Date" in df.columns:
            df = df.with_columns(pl.col("Date").str.strptime(pl.Date, strict=False))
        if "Close" in df.columns:
            df = df.with_columns(pl.col("Close").cast(pl.Float64))
        return enforce_code_column_types(df.sort("Date"))

    async def fetch_indices_ohlc(
        self,
        session: aiohttp.ClientSession,
        from_date: str,
        to_date: str,
        codes: list[str] | None = None,
    ) -> pl.DataFrame:
        """Fetch multiple index OHLC series via /indices endpoint.

        Args:
            session: aiohttp client session (authenticated)
            from_date: inclusive start (YYYY-MM-DD or YYYYMMDD)
            to_date: inclusive end (YYYY-MM-DD or YYYYMMDD)
            codes: optional list of index codes to fetch. If None, attempts
                   to fetch all available within date range (API dependent).

        Returns:
            Polars DataFrame with columns: Code, Date, Open, High, Low, Close
            and any additional fields returned by API. Types are normalized.
        """
        if not self.id_token:
            raise RuntimeError("authenticate() must be called first")

        headers = {"Authorization": f"Bearer {self.id_token}"}
        base_url = f"{self.base_url}/indices"

        async def _fetch_for_code(code: str) -> list[dict]:
            rows: list[dict] = []
            pagination_key: str | None = None
            while True:
                params = {"code": code, "from": from_date, "to": to_date}
                if pagination_key:
                    params["pagination_key"] = pagination_key
                status, data = await self._request_json(
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
                    # Ensure Code is set if API omits it in each row
                    for it in items:
                        it.setdefault("Code", code)
                    rows.extend(items)
                pagination_key = data.get("pagination_key") if isinstance(data, dict) else None
                if not pagination_key:
                    break
            return rows

        async def _fetch_all() -> list[dict]:
            """Attempt range fetch for all indices if codes not provided."""
            rows: list[dict] = []
            pagination_key: str | None = None
            while True:
                params = {"from": from_date, "to": to_date}
                if pagination_key:
                    params["pagination_key"] = pagination_key
                status, data = await self._request_json(
                    session,
                    "GET",
                    base_url,
                    label="indices:all",
                    params=params,
                    headers=headers,
                )
                if status != 200 or not isinstance(data, dict):
                    break
                items = data.get("indices") or data.get("data") or []
                if items:
                    rows.extend(items)
                pagination_key = data.get("pagination_key") if isinstance(data, dict) else None
                if not pagination_key:
                    break
            return rows

        all_rows: list[dict] = []
        if codes:
            # Concurrency-limited fan-out by code
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
        else:
            try:
                all_rows = await _fetch_all()
            except Exception:
                all_rows = []

        if not all_rows:
            return pl.DataFrame()

        # Pre-sanitize raw payload to avoid schema conflicts during DataFrame construction
        # Replace known sentinels ("-", "*", "", "null") with None so that Polars can infer dtypes safely
        for row in all_rows:
            for k, v in list(row.items()):
                if isinstance(v, str) and v.strip().lower() in {"-", "*", "", "null"}:
                    row[k] = None

        df = pl.DataFrame(all_rows)
        cols = df.columns

        def _dtcol(name: str) -> pl.Expr:
            if name not in cols:
                return pl.lit(None, dtype=pl.Date).alias(name)
            return pl.col(name).str.strptime(pl.Date, strict=False).alias(name)

        def _fcol(name: str) -> pl.Expr:
            if name not in cols:
                return pl.lit(None, dtype=pl.Float64).alias(name)
            return pl.col(name).cast(pl.Float64, strict=False).alias(name)

        out = df.with_columns([
            pl.col("Code").cast(pl.Utf8) if "Code" in cols else pl.lit(None, dtype=pl.Utf8).alias("Code"),
            _dtcol("Date"),
            _fcol("Open"),
            _fcol("High"),
            _fcol("Low"),
            _fcol("Close"),
        ])

        return enforce_code_column_types(out.sort(["Code", "Date"])) if {"Code", "Date"}.issubset(out.columns) else enforce_code_column_types(out)

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
                status, data = await self._request_json(
                    session,
                    "GET",
                    url,
                    label=f"weekly_margin:{date_str}",
                    params=params,
                    headers=headers,
                )
                if status != 200 or not isinstance(data, dict):
                    break
                items = data.get("weekly_margin_interest", [])
                if items:
                    rows.extend(items)
                pagination_key = data.get("pagination_key")
                if not pagination_key:
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
            return pl.col(name).str.strptime(pl.Date, strict=False).alias(name)
        cols = df.columns
        out = df.with_columns([
            pl.col("Code").cast(pl.Utf8) if "Code" in cols else pl.lit(None, dtype=pl.Utf8).alias("Code"),
            _dtcol("Date") if "Date" in cols else pl.lit(None, dtype=pl.Date).alias("Date"),
            _dtcol("PublishedDate") if "PublishedDate" in cols else pl.lit(None, dtype=pl.Date).alias("PublishedDate"),
            pl.col("LongMarginTradeVolume").cast(pl.Float64) if "LongMarginTradeVolume" in cols else pl.lit(None, dtype=pl.Float64).alias("LongMarginTradeVolume"),
            pl.col("ShortMarginTradeVolume").cast(pl.Float64) if "ShortMarginTradeVolume" in cols else pl.lit(None, dtype=pl.Float64).alias("ShortMarginTradeVolume"),
            pl.col("LongNegotiableMarginTradeVolume").cast(pl.Float64) if "LongNegotiableMarginTradeVolume" in cols else pl.lit(None, dtype=pl.Float64).alias("LongNegotiableMarginTradeVolume"),
            pl.col("ShortNegotiableMarginTradeVolume").cast(pl.Float64) if "ShortNegotiableMarginTradeVolume" in cols else pl.lit(None, dtype=pl.Float64).alias("ShortNegotiableMarginTradeVolume"),
            pl.col("LongStandardizedMarginTradeVolume").cast(pl.Float64) if "LongStandardizedMarginTradeVolume" in cols else pl.lit(None, dtype=pl.Float64).alias("LongStandardizedMarginTradeVolume"),
            pl.col("ShortStandardizedMarginTradeVolume").cast(pl.Float64) if "ShortStandardizedMarginTradeVolume" in cols else pl.lit(None, dtype=pl.Float64).alias("ShortStandardizedMarginTradeVolume"),
            (pl.col("IssueType").cast(pl.Int8) if "IssueType" in cols else pl.lit(None, dtype=pl.Int8)).alias("IssueType"),
        ]).drop_nulls(subset=["Code", "Date"]).sort(["Code", "Date"])
        return enforce_code_column_types(out)

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
                status, data = await self._request_json(
                    session,
                    "GET",
                    base_url,
                    label="futures_range",
                    params=params,
                    headers=headers,
                )
                if status != 200 or not isinstance(data, dict):
                    break
                items = data.get("futures") or data.get("derivatives") or data.get("data") or []
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
                    status, data = await self._request_json(
                        session,
                        "GET",
                        base_url,
                        label=f"futures_date:{date_str}",
                        params=params,
                        headers=headers,
                    )
                    if status != 200 or not isinstance(data, dict):
                        break
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
        self, session: aiohttp.ClientSession, from_date: str, to_date: str, *, business_days: list[str] | None = None
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
                status, data = await self._request_json(
                    session,
                    "GET",
                    url,
                    label=f"daily_margin:{date_str}",
                    params=params,
                    headers=headers,
                )
                if status != 200 or not isinstance(data, dict):
                    break

                items = data.get("daily_margin_interest", [])
                if items:
                    # Normalize sentinels before creating DataFrame
                    for item in items:
                        for key, value in item.items():
                            if isinstance(value, str) and value in ("-", "*", "", "null", "NULL", "None"):
                                item[key] = None
                    rows.extend(items)
                pagination_key = data.get("pagination_key")
                if not pagination_key:
                    break
            return rows

        # Fetch data for each business day in range
        all_rows: list[dict] = []
        if business_days:
            def _canon(d: str) -> str:
                return f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 and d.isdigit() else d
            for d in business_days:
                d_str = _canon(d)
                try:
                    if not (start <= _dt.datetime.strptime(d_str, "%Y-%m-%d").date() <= end):
                        continue
                except Exception:
                    continue
                items = await _fetch_for_date(d_str)
                if items:
                    all_rows.extend(items)
        else:
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

        # Keep PublishReason as dict when provided (features側でStruct/Utf8両対応)

        # Create DataFrame - let Polars infer types first
        df = pl.DataFrame(all_rows)

        # Normalize dtypes
        cols = df.columns

        # Handle date columns
        def _dtcol(name: str) -> pl.Expr:
            return pl.col(name).str.strptime(pl.Date, strict=False).alias(name)

        # Handle numeric columns that may contain "-" for missing values
        def _float_col(name: str) -> pl.Expr:
            if name not in cols:
                return pl.lit(None, dtype=pl.Float64).alias(name)
            # Check if column is already numeric
            if df[name].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                return pl.col(name).cast(pl.Float64).alias(name)
            # Otherwise handle string conversion
            return (
                pl.when(pl.col(name).cast(pl.Utf8).is_in(["-", "*", "", "null", "NULL", "None"]))
                .then(None)
                .otherwise(pl.col(name).cast(pl.Float64, strict=False))
                .alias(name)
            )

        out = df.with_columns([
            pl.col("Code").cast(pl.Utf8) if "Code" in cols else pl.lit(None, dtype=pl.Utf8).alias("Code"),
            _dtcol("PublishedDate") if "PublishedDate" in cols else pl.lit(None, dtype=pl.Date).alias("PublishedDate"),
            _dtcol("ApplicationDate") if "ApplicationDate" in cols else pl.lit(None, dtype=pl.Date).alias("ApplicationDate"),
            # PublishReason はそのまま（dict/Null）
            (pl.col("PublishReason")) if "PublishReason" in cols else pl.lit(None).alias("PublishReason"),
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

        return enforce_code_column_types(out)

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

        # First try: Range-based fetch with different parameter names
        try:
            print(f"Trying range-based futures fetch: {from_date} to {to_date}")
            url = f"{self.base_url}/derivatives/futures"
            # Try different parameter combinations for compatibility
            param_sets = [
                {"from": from_date, "to": to_date},
                {"from_date": from_date, "to_date": to_date},
                {"start_date": from_date, "end_date": to_date}
            ]

            for params in param_sets:
                headers = {"Authorization": f"Bearer {self.id_token}"}
                status, data = await self._request_json(
                    session,
                    "GET",
                    url,
                    label="futures_range_alt",
                    params=params,
                    headers=headers,
                )
                if status == 200 and isinstance(data, dict):
                    for key in ["futures", "derivatives", "data", "derivatives_futures"]:
                        if key in data and data[key]:
                            batch = data[key]
                            df_batch = pl.DataFrame(batch)
                            if not df_batch.is_empty():
                                all_futures.append(df_batch)
                                print(f"Retrieved {len(df_batch)} futures records with range query")
                                break
                    if all_futures:
                        break
                elif status == 400:
                    print(f"Bad request with params {params}, trying next...")
                    continue
        except Exception as e:
            print(f"Range-based futures fetch failed: {e}")

        # Second try: Category-based fetch (original method)
        if not all_futures:
            for category in priority_categories:
                try:
                    print(f"Fetching futures data for category: {category}")

                    # Fetch derivatives futures for specific category
                    # Using pagination similar to other endpoints
                    page = 1
                    while True:
                        url = f"{self.base_url}/derivatives/futures"
                        params = {
                            "from": from_date,
                            "to": to_date,
                            "DerivativesProductCategory": category
                        }

                        headers = {"Authorization": f"Bearer {self.id_token}"}
                        status, data = await self._request_json(
                            session,
                            "GET",
                            url,
                            label=f"futures_category:{category}",
                            params=params,
                            headers=headers,
                        )
                        if status == 404:
                            print(f"No data found for category {category}")
                            break
                        if status == 400:
                            print(f"Bad request for range, trying date-by-date for {category}")
                            break
                        if status != 200 or not isinstance(data, dict):
                            print(f"API error for {category}: {status}")
                            break

                        batch = None
                        for key in ["futures", "derivatives", "data", "derivatives_futures"]:
                            if key in data and data[key]:
                                batch = data[key]
                                break

                        if not batch:
                            break

                        df_batch = pl.DataFrame(batch)
                        if not df_batch.is_empty():
                            df_batch = df_batch.with_columns([
                                pl.lit(category).alias("ProductCategory")
                            ])
                            all_futures.append(df_batch)

                        if len(batch) < 1000:
                            break
                        page += 1

                        # Add delay to respect rate limits
                        await asyncio.sleep(0.1)

                except Exception as e:
                    print(f"Error fetching futures category {category}: {e}")
                    continue

        # Third try: Date-by-date fetch if range fetch failed
        if not all_futures:
            print("Trying date-by-date futures fetch as final fallback")
            import datetime as _dt

            def _parse(d: str) -> _dt.date:
                if "-" in d:
                    return _dt.datetime.strptime(d, "%Y-%m-%d").date()
                return _dt.datetime.strptime(d, "%Y%m%d").date()

            start = _parse(from_date)
            end = _parse(to_date)
            cur = start

            while cur <= end and len(all_futures) < 100:  # Limit to prevent excessive API calls
                if cur.weekday() < 5:  # Business days only
                    date_str = cur.strftime("%Y-%m-%d")
                    try:
                        url = f"{self.base_url}/derivatives/futures"
                        params = {"date": date_str}
                        headers = {"Authorization": f"Bearer {self.id_token}"}

                        status, data = await self._request_json(
                            session,
                            "GET",
                            url,
                            label=f"futures_single_date:{date_str}",
                            params=params,
                            headers=headers,
                        )
                        if status == 200 and isinstance(data, dict):
                            for key in ["futures", "derivatives", "data", "derivatives_futures"]:
                                if key in data and data[key]:
                                    batch = data[key]
                                    df_batch = pl.DataFrame(batch)
                                    if not df_batch.is_empty():
                                        all_futures.append(df_batch)
                                        print(f"Retrieved futures for {date_str}")
                                        break
                        await asyncio.sleep(0.1)
                    except Exception as e:
                        print(f"Failed to fetch futures for {date_str}: {e}")

                cur += _dt.timedelta(days=1)

        if not all_futures:
            print("No futures data retrieved from any method")
            return pl.DataFrame()

        # Combine all categories
        df = pl.concat(all_futures, how="vertical")
        print(f"Retrieved {len(df)} futures records total")

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
                status, data = await self._request_json(
                    session,
                    "GET",
                    base_url,
                    label=f"index_option:{date_str}",
                    params=params,
                    headers=headers,
                )
                if status == 404:
                    break
                if status != 200 or not isinstance(data, dict):
                    break
                items = data.get("index_option") or data.get("data") or []
                if items:
                    # Normalize sentinels before adding to rows
                    for item in items:
                        for key, value in item.items():
                            if isinstance(value, str) and value in ("-", "*", "", "null", "NULL", "None"):
                                item[key] = None
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
            # Check actual dtype from dataframe
            if df[name].dtype == pl.Date:
                return pl.col(name)
            elif df[name].dtype in [pl.Utf8, pl.Categorical]:
                return pl.col(name).str.strptime(pl.Date, strict=False).alias(name)
            else:
                return pl.col(name).cast(pl.Date).alias(name)

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
            # Check if column is already numeric
            if col in df.columns and df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                return pl.col(col).cast(pl.Float64)
            # Otherwise handle string conversion with multiple sentinel values
            return (
                pl.when(pl.col(col).cast(pl.Utf8).is_in(["-", "*", "", "null", "NULL", "None"]))
                .then(None)
                .otherwise(pl.col(col).cast(pl.Float64, strict=False))
            )

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
        self, session: aiohttp.ClientSession, from_date: str, to_date: str, *, business_days: list[str] | None = None
    ) -> pl.DataFrame:
        """
        Fetch short selling data from J-Quants API.

        API Requirements:
        - Must specify either 'date' OR 'sector33code'
        - Cannot use from/to without sector33code
        - We iterate through each date to get all sectors' data

        Args:
            session: aiohttp session
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)

        Returns:
            pl.DataFrame with short selling ratio data
        """
        url = f"{self.base_url}/markets/short_selling"
        headers = {"Authorization": f"Bearer {self.id_token}"}
        sentinel = {"", "-", "*", "null", "NULL", "None"}

        rows: list[dict] = []

        # Iterate through each date to get all sectors' data
        import datetime
        current_date = datetime.datetime.strptime(from_date, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(to_date, "%Y-%m-%d")

        self._logger.info(f"Fetching short selling data from {from_date} to {to_date}")

        def _iter_dates() -> list[str]:
            if business_days:
                out = []
                for d in business_days:
                    d_str = f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 and d.isdigit() else d
                    if from_date <= d_str <= to_date:
                        out.append(d_str)
                return out
            out = []
            cur = current_date
            while cur <= end_date:
                out.append(cur.strftime("%Y-%m-%d"))
                cur += datetime.timedelta(days=1)
            return out

        for date_str in _iter_dates():
            formatted_date = self._format_date_param(date_str)
            # Use date parameter for each date
            params = {"date": formatted_date if formatted_date else date_str}

            self._logger.debug(f"Fetching short selling for date: {date_str}")

            pagination_key: str | None = None
            while True:
                if pagination_key:
                    params["pagination_key"] = pagination_key

                try:
                    status, data = await self._request_json(
                        session,
                        "GET",
                        url,
                        label="short_selling",
                        params=params,
                        headers=headers,
                    )

                    if status == 200 and isinstance(data, dict):
                        batch = data.get("short_selling") or []
                        if batch:
                            rows.extend(batch)

                        pagination_key = data.get("pagination_key")
                        if not pagination_key:
                            break
                    elif status == 404:
                        self._logger.debug(f"No short selling data for {date_str}")
                        break
                    else:
                        self._logger.warning(f"Failed to fetch short selling for {date_str}, status: {status}")
                        break

                except Exception as e:
                    self._logger.warning(f"Error fetching short selling for {date_str}: {e}")
                    break

        # end for date loop

        if not rows:
            return pl.DataFrame()

        df = pl.DataFrame(rows)

        date_cols = [c for c in df.columns if c.endswith("Date") or c == "Date"]
        if date_cols:
            df = df.with_columns([
                pl.col(c).cast(pl.Utf8).str.strptime(pl.Date, strict=False).alias(c)
                for c in date_cols
            ])

        # Identify categorical columns by name heuristics
        string_markers = ("Code", "Category", "Division", "Class", "Name", "Type", "Section")
        str_cols = [
            c
            for c in df.columns
            if any(marker in c for marker in string_markers)
        ]
        if str_cols:
            df = df.with_columns([pl.col(c).cast(pl.Utf8) for c in str_cols])

        numeric_cols = [c for c in df.columns if c not in date_cols and c not in str_cols]
        if numeric_cols:
            df = df.with_columns(
                [
                    pl.when(pl.col(c).cast(pl.Utf8).str.strip_chars().is_in(sentinel))
                    .then(None)
                    .otherwise(pl.col(c))
                    .alias(c)
                    for c in numeric_cols
                ]
            ).with_columns([
                pl.col(c).cast(pl.Float64, strict=False).alias(c) for c in numeric_cols
            ])

        df = df.sort(date_cols) if date_cols else df
        print(f"Retrieved {len(df)} short selling records")
        # Normalize to ensure consistent column structure (adds PublishedDate if missing)
        df = self._normalize_short_selling_data(df)
        return df

    async def get_short_selling_positions(
        self, session: aiohttp.ClientSession, from_date: str, to_date: str, *, business_days: list[str] | None = None
    ) -> pl.DataFrame:
        """
        Fetch short selling positions data from J-Quants API.

        API Requirements:
        - Must specify either 'code', 'disclosed_date', or 'calculated_date'
        - We iterate through each disclosed_date to get all data

        Args:
            session: aiohttp session
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)

        Returns:
            pl.DataFrame with short selling positions data
        """
        url = f"{self.base_url}/markets/short_selling_positions"
        headers = {"Authorization": f"Bearer {self.id_token}"}
        sentinel = {"", "-", "*", "null", "NULL", "None"}

        rows: list[dict] = []

        # Iterate through each date using disclosed_date parameter
        import datetime
        current_date = datetime.datetime.strptime(from_date, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(to_date, "%Y-%m-%d")

        self._logger.info(f"Fetching short selling positions from {from_date} to {to_date}")

        def _iter_dates() -> list[str]:
            if business_days:
                out = []
                for d in business_days:
                    d_str = f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 and d.isdigit() else d
                    if from_date <= d_str <= to_date:
                        out.append(d_str)
                return out
            out = []
            cur = current_date
            while cur <= end_date:
                out.append(cur.strftime("%Y-%m-%d"))
                cur += datetime.timedelta(days=1)
            return out

        for date_str in _iter_dates():
            formatted_date = self._format_date_param(date_str)
            # Use disclosed_date parameter for each date
            params = {"disclosed_date": formatted_date if formatted_date else date_str}

            self._logger.debug(f"Fetching short positions for disclosed_date: {date_str}")

            try:
                status, data = await self._request_json(
                    session,
                    "GET",
                    url,
                    label="short_selling_positions",
                    params=params,
                    headers=headers,
                )

                if status == 200 and isinstance(data, dict):
                    batch = data.get("short_selling_positions") or []
                    if batch:
                        rows.extend(batch)
                elif status == 404:
                    self._logger.debug(f"No short positions data for {date_str}")
                else:
                    self._logger.warning(f"Failed to fetch short positions for {date_str}, status: {status}")

            except Exception as e:
                self._logger.warning(f"Error fetching short positions for {date_str}: {e}")

        # end for date loop

        if not rows:
            return pl.DataFrame()

        df = pl.DataFrame(rows)

        date_cols = [c for c in df.columns if c.endswith("Date") or c == "Date"]
        if date_cols:
            df = df.with_columns([
                pl.col(c).cast(pl.Utf8).str.strptime(pl.Date, strict=False).alias(c)
                for c in date_cols
            ])

        string_markers = ("Code", "Category", "Division", "Class", "Name", "Type", "Section")
        str_cols = [
            c
            for c in df.columns
            if any(marker in c for marker in string_markers)
        ]
        if str_cols:
            df = df.with_columns([pl.col(c).cast(pl.Utf8) for c in str_cols])

        numeric_cols = [c for c in df.columns if c not in date_cols and c not in str_cols]
        if numeric_cols:
            df = df.with_columns(
                [
                    pl.when(pl.col(c).cast(pl.Utf8).str.strip_chars().is_in(sentinel))
                    .then(None)
                    .otherwise(pl.col(c))
                    .alias(c)
                    for c in numeric_cols
                ]
            ).with_columns([
                pl.col(c).cast(pl.Float64, strict=False).alias(c) for c in numeric_cols
            ])

        df = df.sort(date_cols + ["Code"]) if date_cols and "Code" in df.columns else df
        print(f"Retrieved {len(df)} short selling positions records")
        # Normalize to ensure consistent column structure (adds PublishedDate if missing)
        df = self._normalize_short_selling_positions_data(df)
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

        url = f"{self.base_url}/fins/announcement"
        params = {"from": from_date, "to": to_date}
        headers = {"Authorization": f"Bearer {self.id_token}"}

        try:
            status, data = await self._request_json(
                session,
                "GET",
                url,
                label="earnings_announcements",
                params=params,
                headers=headers,
            )
            if status == 404:
                print(f"Earnings announcements endpoint not found: {url}")
                return pl.DataFrame()
            if status != 200 or data is None:
                print(f"Error fetching earnings announcements: {status}")
                return pl.DataFrame()

            if isinstance(data, dict) and "announcement" in data:
                all_data.extend(data.get("announcement") or [])
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
        # Note: API returns "Date" as the announcement date (not "AnnouncementDate")
        column_mapping = {
            "LocalCode": "Code",
            "Code": "Code",
            "Date": "Date",
            "CompanyName": "CompanyName",
            "FiscalYear": "FiscalYear",
            "FiscalQuarter": "FiscalQuarter",
            "SectorName": "SectorName",
            "Section": "Section",
        }

        # Rename columns if they exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and old_name != new_name:
                df = df.rename({old_name: new_name})

        # Ensure essential columns exist with proper types
        if "Code" not in df.columns:
            print("Warning: No Code column in earnings announcement data")
            return pl.DataFrame()

        # Convert Date column to proper format (this is the announcement date)
        if "Date" in df.columns:
            df = df.with_columns([
                pl.col("Date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("Date")
            ])

        # Ensure Code is string
        df = df.with_columns([
            pl.col("Code").cast(pl.Utf8)
        ])

        # Sort by announcement date and code
        df = df.sort(["Date", "Code"])

        return df

    async def get_sector_short_selling(
        self, session: aiohttp.ClientSession, from_date: str, to_date: str, *, business_days: list[str] | None = None
    ) -> pl.DataFrame:
        """
        Fetch sector-wise short selling data from J-Quants API (/markets/short_selling).

        API Requirements:
        - Must specify either 'date' OR 'sector33code'
        - When using date range (from/to), must also specify sector33code
        - Without sector33code, we iterate through each date individually

        Args:
            session: aiohttp session
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)

        Returns:
            pl.DataFrame with sector short selling data
        """
        all_data = []

        url = f"{self.base_url}/markets/short_selling"

        # Since we don't have sector33code, iterate through each date
        import datetime
        current_date = datetime.datetime.strptime(from_date, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(to_date, "%Y-%m-%d")

        self._logger.info(f"Fetching sector short selling data from {from_date} to {to_date}")

        def _iter_dates() -> list[str]:
            if business_days:
                out = []
                for d in business_days:
                    d_str = f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 and d.isdigit() else d
                    if from_date <= d_str <= to_date:
                        out.append(d_str)
                return out
            out: list[str] = []
            cur = current_date
            while cur <= end_date:
                out.append(cur.strftime("%Y-%m-%d"))
                cur += datetime.timedelta(days=1)
            return out

        for date_str in _iter_dates():
            params = {"date": date_str}  # Use single date parameter

            self._logger.debug(f"Fetching sector short selling for date: {date_str}")

            headers = {"Authorization": f"Bearer {self.id_token}"}

            try:
                status, data = await self._request_json(
                    session,
                    "GET",
                    url,
                    label="sector_short_selling",
                    params=params,
                    headers=headers,
                )

                if status == 200 and data is not None:
                    if isinstance(data, dict) and "short_selling" in data:
                        all_data.extend(data["short_selling"])
                    elif isinstance(data, list):
                        all_data.extend(data)
                elif status == 404:
                    self._logger.debug(f"No sector short selling data for {date_str}")
                else:
                    self._logger.warning(f"Failed to fetch sector short selling for {date_str}, status: {status}")

            except asyncio.TimeoutError:
                self._logger.warning(f"Timeout fetching sector short selling for {date_str}")
            except Exception as e:
                self._logger.warning(f"Error fetching sector short selling for {date_str}: {e}")

        # end for date loop

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

        # If PublishedDate doesn't exist in API response, use Date as PublishedDate
        # (Sector-level short selling data only has Date column)
        if "PublishedDate" not in cols and "Date" in cols:
            normalized = normalized.with_columns([
                pl.col("Date").alias("PublishedDate")
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

        # If PublishedDate doesn't exist in API response, use Date as PublishedDate
        # (Sector-level data only has Date column)
        if "PublishedDate" not in cols and "Date" in cols:
            normalized = normalized.with_columns([
                pl.col("Date").alias("PublishedDate")
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

    # ========== Safe Session Management Wrapper Methods ==========

    async def safe_get_futures_daily(
        self, session: aiohttp.ClientSession, from_date: str, to_date: str
    ) -> pl.DataFrame:
        """
        Safe wrapper for get_futures_daily with session health checking.

        This method handles session closure issues common with futures API calls.
        """
        try:
            if not await self._ensure_session_health(session):
                raise RuntimeError("Session is closed - requires new session and re-authentication")

            return await self.get_futures_daily(session, from_date, to_date)

        except Exception as e:
            # Log specific error details for futures API
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Futures API call failed: {e}")
            raise

    async def safe_get_short_selling(
        self, session: aiohttp.ClientSession, from_date: str, to_date: str
    ) -> pl.DataFrame:
        """
        Safe wrapper for get_short_selling with session health checking.

        This method handles session closure issues common with short selling API calls.
        """
        try:
            if not await self._ensure_session_health(session):
                raise RuntimeError("Session is closed - requires new session and re-authentication")

            return await self.get_short_selling(session, from_date, to_date)

        except Exception as e:
            # Log specific error details for short selling API
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Short selling API call failed: {e}")
            raise

    async def safe_get_short_selling_positions(
        self, session: aiohttp.ClientSession, from_date: str, to_date: str
    ) -> pl.DataFrame:
        """
        Safe wrapper for get_short_selling_positions with session health checking.

        This method handles session closure issues common with short selling positions API calls.
        """
        try:
            if not await self._ensure_session_health(session):
                raise RuntimeError("Session is closed - requires new session and re-authentication")

            return await self.get_short_selling_positions(session, from_date, to_date)

        except Exception as e:
            # Log specific error details for short selling positions API
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Short selling positions API call failed: {e}")
            raise

    async def safe_get_sector_short_selling(
        self, session: aiohttp.ClientSession, from_date: str, to_date: str
    ) -> pl.DataFrame:
        """
        Safe wrapper for get_sector_short_selling with session health checking.

        This method handles session closure issues common with sector short selling API calls.
        """
        try:
            if not await self._ensure_session_health(session):
                raise RuntimeError("Session is closed - requires new session and re-authentication")

            return await self.get_sector_short_selling(session, from_date, to_date)

        except Exception as e:
            # Log specific error details for sector short selling API
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Sector short selling API call failed: {e}")
            raise
