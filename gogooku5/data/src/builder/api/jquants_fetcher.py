"""Client for interacting with the J-Quants API."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from requests.exceptions import HTTPError, ReadTimeout

from .base import APIClient

AUTH_ENDPOINT = "/token/auth_user"
REFRESH_ENDPOINT = "/token/auth_refresh"
DAILY_QUOTES_ENDPOINT = "/prices/daily_quotes"
LISTED_ENDPOINT = "/listed/info"
MARGIN_DAILY_ENDPOINT = "/margin/daily"
MARGIN_WEEKLY_ENDPOINT = "/margin/weekly"
MAX_RETRIES = 8
BACKOFF_CAP_SECONDS = 60.0


@dataclass
class JQuantsFetcher(APIClient):
    """J-Quants API client with minimal pagination helpers."""

    base_url: str = "https://api.jquants.com/v1"
    _refresh_token: Optional[str] = field(default=None, init=False)
    _token: Optional[str] = field(default=None, init=False)

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
        self._refresh_token = refresh_token
        return self._exchange_refresh_token(refresh_token)

    def _ensure_token(self) -> str:
        if not self._token:
            return self.authenticate()
        return self._token

    def refresh(self) -> str:
        if not self._refresh_token:
            return self.authenticate()
        return self._exchange_refresh_token(self._refresh_token)

    def _exchange_refresh_token(self, refresh_token: str) -> str:
        response = self.request("POST", REFRESH_ENDPOINT, params={"refreshtoken": refresh_token})
        data = response.json()
        id_token = data.get("idToken") or data.get("token")
        if not id_token:
            raise RuntimeError("J-Quants token refresh failed: missing idToken")
        new_refresh = data.get("refreshToken") or data.get("refresh_token")
        if new_refresh:
            self._refresh_token = new_refresh
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

    def fetch_margin_daily(self, *, date: str) -> Dict[str, Any]:
        payload = self._collect_paginated(MARGIN_DAILY_ENDPOINT, params={"date": date}, data_key="margin_daily")
        return payload

    def fetch_margin_weekly(self, *, date: str) -> Dict[str, Any]:
        payload = self._collect_paginated(MARGIN_WEEKLY_ENDPOINT, params={"date": date}, data_key="margin_weekly")
        return payload

    # ------------------------------------------------------------------
    # Pagination helpers
    # ------------------------------------------------------------------
    def fetch_quotes_paginated(self, *, code: str, from_: str, to: str) -> List[Dict[str, Any]]:
        all_rows: List[Dict[str, Any]] = []
        pagination_key: Optional[str] = None
        while True:
            payload = self.fetch_daily_quotes(code=code, from_=from_, to=to, pagination_key=pagination_key)
            rows = payload.get("daily_quotes", [])
            if rows:
                all_rows.extend(rows)
            pagination_key = payload.get("pagination_key")
            if not pagination_key:
                break
        return all_rows

    def fetch_margin_daily_window(self, *, dates: Iterable[str]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for date in dates:
            payload = self.fetch_margin_daily(date=date)
            results.extend(payload.get("margin_daily", []))
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
        while True:
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
        if data_key:
            base: Dict[str, Any] = dict(last_payload or {})
            base[data_key] = aggregated
            base.pop("pagination_key", None)
            return base
        return last_payload or {}

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
