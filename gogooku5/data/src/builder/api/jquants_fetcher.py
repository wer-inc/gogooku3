"""Client for interacting with the J-Quants API."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .base import APIClient

AUTH_ENDPOINT = "/token/auth_user"
REFRESH_ENDPOINT = "/token/auth_refresh"
DAILY_QUOTES_ENDPOINT = "/prices/daily_quotes"
LISTED_ENDPOINT = "/listed/info"
MARGIN_DAILY_ENDPOINT = "/margin/daily"
MARGIN_WEEKLY_ENDPOINT = "/margin/weekly"


@dataclass
class JQuantsFetcher(APIClient):
    """J-Quants API client with minimal pagination helpers."""

    base_url: str = "https://api.jquants.com/v1"
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
        token = response.json().get("token")
        if not token:
            raise RuntimeError("J-Quants authentication failed: missing token")
        self._token = token
        return token

    def _ensure_token(self) -> str:
        if not self._token:
            return self.authenticate()
        return self._token

    def refresh(self) -> str:
        token = self._ensure_token()
        response = self.request("POST", REFRESH_ENDPOINT, json={"token": token})
        new_token = response.json().get("token")
        if not new_token:
            raise RuntimeError("J-Quants token refresh failed: missing token")
        self._token = new_token
        return new_token

    # ------------------------------------------------------------------
    # Data endpoints
    # ------------------------------------------------------------------
    def fetch_listed_info(self) -> Dict[str, Any]:
        response = self._authorized_request("GET", LISTED_ENDPOINT)
        return response.json()

    def fetch_daily_quotes(self, *, code: str, from_: str, to: str, page: int = 1) -> Dict[str, Any]:
        params = {"code": code, "from": from_, "to": to, "page": page}
        response = self._authorized_request("GET", DAILY_QUOTES_ENDPOINT, params=params)
        return response.json()

    def fetch_margin_daily(self, *, date: str) -> Dict[str, Any]:
        response = self._authorized_request("GET", MARGIN_DAILY_ENDPOINT, params={"date": date})
        return response.json()

    def fetch_margin_weekly(self, *, date: str) -> Dict[str, Any]:
        response = self._authorized_request("GET", MARGIN_WEEKLY_ENDPOINT, params={"date": date})
        return response.json()

    # ------------------------------------------------------------------
    # Pagination helpers
    # ------------------------------------------------------------------
    def fetch_quotes_paginated(self, *, code: str, from_: str, to: str) -> List[Dict[str, Any]]:
        page = 1
        all_rows: List[Dict[str, Any]] = []
        while True:
            payload = self.fetch_daily_quotes(code=code, from_=from_, to=to, page=page)
            rows = payload.get("daily_quotes", [])
            if not rows:
                break
            all_rows.extend(rows)
            if len(rows) < payload.get("paginationLimit", len(rows)):
                break
            page += 1
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
    def _authorized_request(
        self,
        method: str,
        endpoint: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
    ):
        token = self._ensure_token()
        headers = {"Authorization": f"Bearer {token}"}
        return self.request(method, endpoint, params=params, json=json, headers=headers)
