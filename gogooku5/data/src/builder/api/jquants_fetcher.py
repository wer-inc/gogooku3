"""Client for interacting with the J-Quants API."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set

from requests.exceptions import HTTPError, ReadTimeout

from .base import APIClient

AUTH_ENDPOINT = "/token/auth_user"
REFRESH_ENDPOINT = "/token/auth_refresh"
DAILY_QUOTES_ENDPOINT = "/prices/daily_quotes"
LISTED_ENDPOINT = "/listed/info"
MARGIN_DAILY_ENDPOINT = "/markets/daily_margin_interest"
MARGIN_WEEKLY_ENDPOINT = "/markets/weekly_margin_interest"
MAX_RETRIES = 8
BACKOFF_CAP_SECONDS = 60.0
MAX_PAGINATION_PAGES = 1000


@dataclass
class JQuantsFetcher(APIClient):
    """J-Quants API client with minimal pagination helpers."""

    base_url: str = "https://api.jquants.com/v1"
    _refresh_token: Optional[str] = field(default=None, init=False)
    _token: Optional[str] = field(default=None, init=False)
    _token_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    # ------------------------------------------------------------------
    # Authentication helpers
    # ------------------------------------------------------------------
    def authenticate(self) -> str:
        payload = {
            "mailaddress": self.settings.jquants_auth_email,
            "password": self.settings.jquants_auth_password,
        }
        response = self.request("POST", AUTH_ENDPOINT, json=payload)
        data = response.json()
        refresh_token = data.get("refreshToken") or data.get("refresh_token")
        if not refresh_token:
            raise RuntimeError("J-Quants authentication failed: missing refreshToken")
        return self._exchange_refresh_token(refresh_token)

    def _ensure_token(self) -> str:
        token = self._token
        if token:
            return token
        with self._token_lock:
            if self._token:
                return self._token
        return self.authenticate()

    def refresh(self) -> str:
        with self._token_lock:
            refresh_token = self._refresh_token
        if not refresh_token:
            return self.authenticate()
        return self._exchange_refresh_token(refresh_token)

    def _exchange_refresh_token(self, refresh_token: str) -> str:
        response = self.request("POST", REFRESH_ENDPOINT, params={"refreshtoken": refresh_token})
        data = response.json()
        id_token = data.get("idToken") or data.get("token")
        if not id_token:
            raise RuntimeError("J-Quants token refresh failed: missing idToken")
        new_refresh = data.get("refreshToken") or data.get("refresh_token")
        with self._token_lock:
            self._refresh_token = new_refresh or refresh_token
            self._token = id_token
        return id_token

    # ------------------------------------------------------------------
    # Data endpoints
    # ------------------------------------------------------------------
    def fetch_listed_info(self) -> Dict[str, Any]:
        payload = self._collect_paginated(LISTED_ENDPOINT, data_key="info")
        return payload

    def fetch_daily_quotes(
        self, *, code: str, from_: str, to: str, pagination_key: Optional[str] = None
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"code": code, "from": from_, "to": to}
        if pagination_key:
            params["pagination_key"] = pagination_key
        response = self._authorized_request("GET", DAILY_QUOTES_ENDPOINT, params=params)
        return response.json()

    def fetch_daily_quotes_by_date(self, *, date: str, pagination_key: Optional[str] = None) -> Dict[str, Any]:
        """Fetch daily quotes for a specific date (all stocks)."""
        params: Dict[str, Any] = {"date": date}
        if pagination_key:
            params["pagination_key"] = pagination_key
        response = self._authorized_request("GET", DAILY_QUOTES_ENDPOINT, params=params)
        return response.json()

    def fetch_margin_daily(
        self,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        code: Optional[str] = None,
        date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch daily margin interest over a window (fallback `date` kept for compat)."""

        params: Dict[str, Any] = {}
        if date and not start:
            params["date"] = date
        else:
            if not start:
                raise ValueError("start date is required when `date` is not provided for margin_daily")
            params["from"] = start
            params["to"] = end or start
        if code:
            params["code"] = code
        payload = self._collect_paginated(
            MARGIN_DAILY_ENDPOINT,
            params=params,
            data_key="daily_margin_interest",
        )
        return payload

    def fetch_margin_weekly(
        self,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        code: Optional[str] = None,
        date: Optional[str] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if date and not start:
            params["date"] = date
        else:
            if not start:
                raise ValueError("start date is required when `date` is not provided for margin_weekly")
            params["from"] = start
            params["to"] = end or start
        if code:
            params["code"] = code
        payload = self._collect_paginated(
            MARGIN_WEEKLY_ENDPOINT,
            params=params,
            data_key="weekly_margin_interest",
        )
        return payload

    # ------------------------------------------------------------------
    # Pagination helpers
    # ------------------------------------------------------------------
    def fetch_quotes_paginated(self, *, code: str, from_: str, to: str) -> List[Dict[str, Any]]:
        all_rows: List[Dict[str, Any]] = []
        pagination_key: Optional[str] = None
        seen_keys: Set[str] = set()
        for page in range(1, MAX_PAGINATION_PAGES + 1):
            payload = self.fetch_daily_quotes(code=code, from_=from_, to=to, pagination_key=pagination_key)
            rows = payload.get("daily_quotes", [])
            if rows:
                all_rows.extend(rows)
            pagination_key = payload.get("pagination_key")
            if not pagination_key:
                break
            pagination_key = self._validate_pagination_key(pagination_key, seen_keys, page)
        else:
            raise RuntimeError(
                f"Exceeded maximum pagination depth ({MAX_PAGINATION_PAGES}) for quotes fetch code={code}"
            )
        return all_rows

    def fetch_quotes_by_date_paginated(self, *, date: str) -> List[Dict[str, Any]]:
        """Fetch all quotes for a specific date with pagination."""
        all_rows: List[Dict[str, Any]] = []
        pagination_key: Optional[str] = None
        seen_keys: Set[str] = set()
        for page in range(1, MAX_PAGINATION_PAGES + 1):
            payload = self.fetch_daily_quotes_by_date(date=date, pagination_key=pagination_key)
            rows = payload.get("daily_quotes", [])
            if rows:
                all_rows.extend(rows)
            pagination_key = payload.get("pagination_key")
            if not pagination_key:
                break
            pagination_key = self._validate_pagination_key(pagination_key, seen_keys, page)
        else:
            raise RuntimeError(
                f"Exceeded maximum pagination depth ({MAX_PAGINATION_PAGES}) for by-date quotes fetch date={date}"
            )
        return all_rows

    def fetch_margin_daily_window(self, *, dates: Iterable[str]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for date in dates:
            payload = self.fetch_margin_daily(start=date, end=date)
            results.extend(payload.get("daily_margin_interest", []))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _collect_paginated(
        self,
        endpoint: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        data_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        aggregated: List[Dict[str, Any]] = []
        pagination_key: Optional[str] = None
        last_payload: Optional[Dict[str, Any]] = None
        seen_keys: Set[str] = set()
        for page in range(1, MAX_PAGINATION_PAGES + 1):
            query = dict(params or {})
            if pagination_key:
                query["pagination_key"] = pagination_key
            response = self._authorized_request("GET", endpoint, params=query or None)
            payload = response.json()
            last_payload = payload
            if data_key:
                aggregated.extend(payload.get(data_key, []))
            pagination_key = payload.get("pagination_key")
            if not pagination_key:
                break
            pagination_key = self._validate_pagination_key(pagination_key, seen_keys, page)
        else:
            raise RuntimeError(f"Exceeded maximum pagination depth ({MAX_PAGINATION_PAGES}) for endpoint {endpoint}")
        if data_key:
            base: Dict[str, Any] = dict(last_payload or {})
            base[data_key] = aggregated
            base.pop("pagination_key", None)
            return base
        return last_payload or {}

    def _validate_pagination_key(
        self,
        pagination_key: Any,
        seen_keys: Set[str],
        page: int,
    ) -> str:
        """Validate pagination key values before re-use."""

        if not isinstance(pagination_key, str) or not pagination_key.strip():
            raise RuntimeError(f"Invalid pagination key received on page {page}: {pagination_key!r}")
        if pagination_key in seen_keys:
            raise RuntimeError(f"Pagination key repeated on page {page}: {pagination_key}")
        seen_keys.add(pagination_key)
        return pagination_key

    def _authorized_request(
        self,
        method: str,
        endpoint: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
    ):
        backoff = 1.0
        for attempt in range(MAX_RETRIES):
            token = self._ensure_token()
            headers = {"Authorization": f"Bearer {token}"}
            try:
                return self.request(method, endpoint, params=params, json=json, headers=headers)
            except ReadTimeout:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(min(backoff, BACKOFF_CAP_SECONDS))
                backoff = min(backoff * 2.0, BACKOFF_CAP_SECONDS)
            except HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status == 401:
                    self.refresh()
                    continue
                if status == 429:
                    retry_after: Optional[float] = None
                    if exc.response is not None:
                        header = exc.response.headers.get("Retry-After")
                        if header is not None:
                            try:
                                retry_after = float(header)
                            except ValueError:
                                retry_after = None
                    if attempt == MAX_RETRIES - 1:
                        raise
                    delay = retry_after if retry_after is not None else backoff
                    time.sleep(min(delay, BACKOFF_CAP_SECONDS))
                    if retry_after is not None:
                        backoff = min(retry_after * 1.5, BACKOFF_CAP_SECONDS)
                    else:
                        backoff = min(backoff * 2.0, BACKOFF_CAP_SECONDS)
                    continue
                raise
        raise RuntimeError(f"Failed to call {endpoint} after {MAX_RETRIES} attempts due to repeated timeouts.")
