"""Advanced J-Quants fetchers wrapping legacy async client."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import polars as pl

from ..config import DatasetBuilderSettings, get_settings
from ..utils.asyncio import run_sync

try:
    import aiohttp  # type: ignore
except ImportError as exc:  # pragma: no cover - defensive
    raise RuntimeError("aiohttp is required for advanced J-Quants fetchers") from exc


@dataclass
class AdvancedJQuantsFetcher:
    """Expose advanced J-Quants endpoints via synchronous helpers."""

    settings: DatasetBuilderSettings = field(default_factory=get_settings)
    max_concurrent: int | None = None

    def _create_async_fetcher(self):
        from .jquants_async_fetcher import JQuantsAsyncFetcher  # type: ignore

        return JQuantsAsyncFetcher(
            email=self.settings.jquants_auth_email,
            password=self.settings.jquants_auth_password,
            max_concurrent=self.max_concurrent or self.settings.index_option_parallel_concurrency,
            enable_parallel_fetch=self.settings.index_option_parallel_fetch,
        )

    async def _run_with_session(self, coro_builder):
        fetcher = self._create_async_fetcher()
        async with aiohttp.ClientSession() as session:
            await fetcher.authenticate(session)
            return await coro_builder(fetcher, session)

    def fetch_topix(self, *, start: str, end: str) -> pl.DataFrame:
        """Fetch TOPIX history for date range."""

        async def _runner(fetcher, session):
            return await fetcher.fetch_topix_data(session, start, end)

        return run_sync(self._run_with_session(_runner))

    def fetch_indices(self, *, start: str, end: str, codes: Sequence[str] | None = None) -> pl.DataFrame:
        """Fetch other indices OHLC."""

        async def _runner(fetcher, session):
            return await fetcher.fetch_indices_ohlc(
                session,
                from_date=start,
                to_date=end,
                codes=list(codes) if codes else None,
            )

        return run_sync(self._run_with_session(_runner))

    def fetch_trades_spec(self, *, start: str, end: str) -> pl.DataFrame:
        """Fetch investor flow data (trades spec)."""

        async def _runner(fetcher, session):
            return await fetcher.get_trades_spec(session, start, end)

        return run_sync(self._run_with_session(_runner))

    def fetch_listed_info(self, *, as_of: str | None = None) -> pl.DataFrame:
        """Fetch listed securities metadata."""

        async def _runner(fetcher, session):
            return await fetcher.get_listed_info(session, date=as_of)

        return run_sync(self._run_with_session(_runner))

    def fetch_margin_weekly(self, *, start: str, end: str | None = None) -> pl.DataFrame:
        """Fetch weekly margin interest snapshots within a range."""

        to_date = end or start

        async def _runner(fetcher, session):
            return await fetcher.get_weekly_margin_interest(session, from_date=start, to_date=to_date)

        return run_sync(self._run_with_session(_runner))

    def fetch_margin_daily(self, *, start: str, end: str) -> pl.DataFrame:
        """Fetch daily margin interest window."""

        async def _runner(fetcher, session):
            return await fetcher.get_daily_margin_interest(session, start, end)

        return run_sync(self._run_with_session(_runner))

    def fetch_futures(self, *, start: str, end: str) -> pl.DataFrame:
        """Fetch futures daily data."""

        async def _runner(fetcher, session):
            return await fetcher.get_futures_daily(session, from_date=start, to_date=end)

        return run_sync(self._run_with_session(_runner))

    def fetch_options(self, *, start: str, end: str) -> pl.DataFrame:
        """Fetch index option aggregated data."""

        async def _runner(fetcher, session):
            return await fetcher.get_index_option(session, from_date=start, to_date=end)

        return run_sync(self._run_with_session(_runner))

    def fetch_short_selling(
        self,
        *,
        start: str,
        end: str,
        business_days: Sequence[str] | None = None,
    ) -> pl.DataFrame:
        """Fetch daily short selling aggregates.

        When business_days is provided, the underlying async client will
        restrict API calls to those dates (to honour trading calendar
        filters and avoid unnecessary requests).
        """

        async def _runner(fetcher, session):
            return await fetcher.get_short_selling(
                session,
                from_date=start,
                to_date=end,
                business_days=list(business_days) if business_days is not None else None,
            )

        return run_sync(self._run_with_session(_runner))

    def fetch_short_positions(self, *, start: str, end: str) -> pl.DataFrame:
        """Fetch detailed short selling positions."""

        async def _runner(fetcher, session):
            return await fetcher.get_short_selling_positions(session, from_date=start, to_date=end)

        return run_sync(self._run_with_session(_runner))

    def fetch_sector_short_selling(
        self, *, start: str, end: str, business_days: list[str] | None = None
    ) -> pl.DataFrame:
        """Fetch sector-level short selling metrics."""

        async def _runner(fetcher, session):
            return await fetcher.get_sector_short_selling(
                session,
                from_date=start,
                to_date=end,
                business_days=business_days,
            )

        return run_sync(self._run_with_session(_runner))

    def fetch_trading_breakdown(self, *, start: str, end: str) -> pl.DataFrame:
        """Fetch breakdown data (institutional categories)."""

        async def _runner(fetcher, session):
            return await fetcher.get_breakdown(session, start, end)

        return run_sync(self._run_with_session(_runner))

    def fetch_prices_am(self, *, start: str, end: str) -> pl.DataFrame:
        """Fetch morning session prices."""

        async def _runner(fetcher, session):
            return await fetcher.get_prices_am(session, start, end)

        return run_sync(self._run_with_session(_runner))

    def fetch_dividends(self, *, start: str, end: str) -> pl.DataFrame:
        """Fetch dividend announcements."""

        async def _runner(fetcher, session):
            return await fetcher.get_dividends(session, start, end)

        return run_sync(self._run_with_session(_runner))

    def fetch_fs_details(self, *, start: str, end: str) -> pl.DataFrame:
        """Fetch financial statement details."""

        async def _runner(fetcher, session):
            return await fetcher.get_fs_details(session, start, end)

        return run_sync(self._run_with_session(_runner))

    def fetch_earnings(self, *, start: str, end: str) -> pl.DataFrame:
        """Fetch earnings announcements."""

        async def _runner(fetcher, session):
            return await fetcher.get_earnings_announcements(session, start, end)

        return run_sync(self._run_with_session(_runner))
