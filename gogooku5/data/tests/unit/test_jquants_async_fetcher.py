from __future__ import annotations

import asyncio
import datetime as dt
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import pytest
from builder.api.jquants_async_fetcher import (
    _TOKEN_CACHE,
    _TOKEN_CACHE_LOCK,
    JQuantsAsyncFetcher,
)


def _clear_token_cache(cache_file: Path | None, *, drop_file: bool = True) -> None:
    with _TOKEN_CACHE_LOCK:
        _TOKEN_CACHE.clear()
    if not drop_file or cache_file is None:
        return
    cache_file.unlink(missing_ok=True)
    cache_file.with_name(cache_file.name + ".lock").unlink(missing_ok=True)


@pytest.fixture
def token_cache_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "tokens.json"
    monkeypatch.setenv("JQUANTS_TOKEN_CACHE_FILE", str(path))
    return path


class _FakeResponse:
    def __init__(self, status: int, payload: dict | None = None, headers: dict | None = None):
        self.status = status
        self._payload = payload or {}
        self.headers = headers or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        if self.status >= 400:
            request_info = SimpleNamespace(real_url="https://api.test.local/token")
            raise aiohttp.ClientResponseError(
                request_info=request_info,
                history=(),
                status=self.status,
                message="error",
                headers=self.headers,
            )

    async def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse]):
        self._responses = responses
        self.closed = False

    def post(self, *args, **kwargs):
        if not self._responses:
            raise AssertionError("No more fake responses available")
        return self._responses.pop(0)


def test_token_cache_hydrates_new_instances(token_cache_file: Path) -> None:
    _clear_token_cache(token_cache_file)
    first = JQuantsAsyncFetcher(email="user@example.com", password="secret", max_concurrent=2)
    expiry = dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=10)
    first.id_token = "cached-id"
    first._refresh_token = "cached-refresh"
    first._token_expiry = expiry
    first._persist_tokens()

    second = JQuantsAsyncFetcher(email="user@example.com", password="secret", max_concurrent=2)
    assert second.id_token == "cached-id"
    assert second._refresh_token == "cached-refresh"
    assert second._token_expiry == expiry


def test_authenticate_short_circuits_with_valid_cache(token_cache_file: Path) -> None:
    _clear_token_cache(token_cache_file)
    bootstrap = JQuantsAsyncFetcher(email="user@example.com", password="secret", max_concurrent=2)
    bootstrap.id_token = "cached-id"
    bootstrap._refresh_token = "cached-refresh"
    bootstrap._token_expiry = dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=10)
    bootstrap._persist_tokens()

    fetcher = JQuantsAsyncFetcher(email="user@example.com", password="secret", max_concurrent=2)

    class _NoPostSession:
        def post(self, *args, **kwargs):
            raise AssertionError("authenticate() should not hit auth endpoints when cache is valid")

    asyncio.run(fetcher.authenticate(_NoPostSession()))


def test_token_cache_persists_across_processes(token_cache_file: Path) -> None:
    _clear_token_cache(token_cache_file)
    first = JQuantsAsyncFetcher(email="user@example.com", password="secret", max_concurrent=2)
    expiry = dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=5)
    first.id_token = "disk-id"
    first._refresh_token = "disk-refresh"
    first._token_expiry = expiry
    first._persist_tokens()

    # Simulate a new process by clearing the in-memory cache but keeping the file
    _clear_token_cache(token_cache_file, drop_file=False)

    restored = JQuantsAsyncFetcher(email="user@example.com", password="secret", max_concurrent=2)
    assert restored.id_token == "disk-id"
    assert restored._refresh_token == "disk-refresh"
    assert restored._token_expiry == expiry


def test_authenticate_retries_after_rate_limit(token_cache_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_token_cache(token_cache_file)
    fetcher = JQuantsAsyncFetcher(email="user@example.com", password="secret", max_concurrent=2)
    fetcher.id_token = None
    fetcher._refresh_token = None
    fetcher._token_expiry = None

    responses = [
        _FakeResponse(429, headers={"Retry-After": "0"}),  # auth_user attempt 1
        _FakeResponse(200, payload={"refreshToken": "refresh-token"}),  # auth_user retry
        _FakeResponse(200, payload={"idToken": "fresh-id"}),  # auth_refresh
    ]
    session = _FakeSession(responses)

    sleep_calls: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr("builder.api.jquants_async_fetcher.asyncio.sleep", _fake_sleep)
    monkeypatch.setattr("builder.api.jquants_async_fetcher.random.uniform", lambda *_args: 0)

    asyncio.run(fetcher.authenticate(session))

    assert fetcher.id_token == "fresh-id"
    assert fetcher._refresh_token == "refresh-token"
    assert len(sleep_calls) == 1
