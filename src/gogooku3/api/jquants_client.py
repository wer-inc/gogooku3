"""
Base J-Quants API client with authentication and request handling.

Extracted from legacy JQuantsAsyncFetcher for better separation of concerns.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from contextlib import nullcontext
from typing import Any, Iterable, Tuple

import aiohttp

from gogooku3.config import JQuantsAPIConfig

logger = logging.getLogger(__name__)


class JQuantsClient:
    """
    Base client for J-Quants API with authentication and throttling.

    Responsibilities:
    - Authentication (refresh token → ID token)
    - Common request handling with retry/throttle logic
    - Adaptive concurrency management
    - Session health checking
    """

    def __init__(self, config: JQuantsAPIConfig):
        """
        Initialize J-Quants client.

        Args:
            config: J-Quants API configuration
        """
        self.config = config
        self.base_url = "https://api.jquants.com/v1"
        self.id_token: str | None = None

        # Concurrency control
        self._current_concurrency = config.max_concurrent_fetch
        self._max_concurrency_ceiling = config.max_concurrent_fetch
        self._min_concurrency = config.min_concurrency
        self.semaphore = asyncio.Semaphore(self._current_concurrency)

        # Throttling state
        self._throttle_backoff = config.throttle_backoff
        self._throttle_sleep_seconds = config.throttle_sleep
        self._recovery_step = config.throttle_step
        self._success_threshold = config.throttle_recovery_success
        self._retry_statuses: Tuple[int, ...] = (429, 503)
        self._success_streak = 0
        self._throttle_hits = 0
        self._throttle_recoveries = 0
        self._throttle_history: list[dict[str, Any]] = []

    async def authenticate(self, session: aiohttp.ClientSession) -> None:
        """
        Authenticate with J-Quants API and store ID token.

        Args:
            session: aiohttp client session

        Raises:
            Exception: If authentication fails
        """
        # Step 1: Get refresh token
        auth_url = f"{self.base_url}/token/auth_user"
        payload = {
            "mailaddress": self.config.auth_email,
            "password": self.config.auth_password,
        }

        async with session.post(auth_url, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
            refresh_token = data["refreshToken"]

        # Step 2: Get ID token
        refresh_url = f"{self.base_url}/token/auth_refresh"
        params = {"refreshtoken": refresh_token}

        async with session.post(refresh_url, params=params) as resp:
            resp.raise_for_status()
            data = await resp.json()
            self.id_token = data["idToken"]

        logger.info("✅ J-Quants authentication successful")

    async def _ensure_session_health(self, session: aiohttp.ClientSession) -> bool:
        """
        Check if session is healthy and can be used.

        Args:
            session: aiohttp client session

        Returns:
            True if session is healthy
        """
        try:
            return not session.closed
        except Exception:
            return False

    def _apply_concurrency(self, new_limit: int, *, reason: str) -> None:
        """
        Adjust concurrency limit.

        Args:
            new_limit: New concurrency limit
            reason: Reason for adjustment (for logging)
        """
        new_limit = max(
            self._min_concurrency,
            min(self._max_concurrency_ceiling, new_limit),
        )
        if new_limit == self._current_concurrency:
            return

        self._current_concurrency = new_limit
        self.semaphore = asyncio.Semaphore(new_limit)
        logger.info(
            "Adjusted J-Quants concurrency → %s (reason=%s)",
            new_limit,
            reason,
        )

    def _record_success(self) -> None:
        """Record successful request and potentially recover concurrency."""
        self._success_streak += 1

        # Check if we can recover concurrency
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
        """
        Handle throttle response (429/503) with backoff and sleep.

        Args:
            label: Request label (for logging)
            resp: Response with throttle status
        """
        status = resp.status
        self._throttle_hits += 1
        self._success_streak = 0

        # Check Retry-After header
        retry_after = resp.headers.get("Retry-After")
        delay = self._throttle_sleep_seconds
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except (TypeError, ValueError):
                pass

        # Reduce concurrency
        scaled = max(1, math.floor(self._current_concurrency * self._throttle_backoff))
        if scaled < self._current_concurrency:
            self._apply_concurrency(scaled, reason=f"throttle:{label}:{status}")

        # Record throttle event
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

        logger.warning(
            "J-Quants throttle detected (%s, status=%s). Sleeping %.1fs.",
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
        """
        Make HTTP request with retry and throttle handling.

        Args:
            session: aiohttp client session
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            label: Request label for logging/metrics
            params: Query parameters
            headers: HTTP headers
            json_payload: JSON request body
            data: Request body
            expected_statuses: Expected successful status codes
            decode_json: Whether to decode JSON response
            use_semaphore: Whether to use semaphore for concurrency control

        Returns:
            Tuple of (status_code, response_data)
        """
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

                        # Handle throttle
                        if status in self._retry_statuses:
                            await self._handle_throttle(label, resp)
                            continue  # Retry

                        # Decode response
                        payload: Any = None
                        if decode_json:
                            try:
                                payload = await resp.json()
                            except Exception:
                                payload = None

                        # Record success
                        self._record_success()
                        return status, payload

                except aiohttp.ClientError:
                    self._success_streak = 0
                    raise

    def throttle_metrics(self) -> dict[str, Any]:
        """
        Get current throttle metrics.

        Returns:
            Dictionary with throttle statistics
        """
        return {
            "hits": self._throttle_hits,
            "recoveries": self._throttle_recoveries,
            "current_concurrency": self._current_concurrency,
            "history": list(self._throttle_history),
        }

    @staticmethod
    def format_date_param(date_str: str | None) -> str | None:
        """
        Normalize date to YYYYMMDD format required by J-Quants.

        Args:
            date_str: Date string (YYYY-MM-DD or YYYYMMDD)

        Returns:
            Formatted date string (YYYYMMDD) or None

        Raises:
            ValueError: If date format is invalid
        """
        import datetime as _dt

        if date_str is None:
            return None

        value = date_str.strip()
        if not value:
            return value

        # Already in YYYYMMDD format
        if len(value) == 8 and value.isdigit():
            return value

        # Convert from YYYY-MM-DD
        if len(value) == 10 and value.count("-") == 2:
            parsed = _dt.datetime.strptime(value, "%Y-%m-%d")
            return parsed.strftime("%Y%m%d")

        raise ValueError(f"Unsupported date format: '{date_str}'")
