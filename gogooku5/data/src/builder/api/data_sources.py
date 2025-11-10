"""Higher-level data source helpers backed by J-Quants advanced fetcher."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import polars as pl

from ..config import DatasetBuilderSettings, get_settings
from ..features.macro.futures_topix import build_futures_features, load_futures
from ..features.macro.global_regime import (
    load_global_regime_data,
    prepare_vvmd_features,
)
from ..features.macro.index_option_225 import (
    build_index_option_225_features,
    load_index_option_225,
)
from ..features.macro.options_asof import build_option_signals, load_options
from ..features.macro.vix import load_vix_history, prepare_vix_features
from ..utils import CacheManager
from .advanced_fetcher import AdvancedJQuantsFetcher


@dataclass
class DataSourceManager:
    """Provide cached access to enriched J-Quants datasets."""

    settings: DatasetBuilderSettings = field(default_factory=get_settings)
    cache: CacheManager = field(default_factory=CacheManager)
    fetcher: AdvancedJQuantsFetcher = field(default_factory=AdvancedJQuantsFetcher)

    def margin_daily(self, *, start: str, end: str) -> pl.DataFrame:
        """Return normalized daily margin balances."""

        cache_key = f"margin_daily_{start}_{end}"
        ttl = self.settings.margin_daily_cache_ttl_days

        def _fetch() -> pl.DataFrame:
            raw = self.fetcher.fetch_margin_daily(start=start, end=end)
            return self._normalize_margin_daily(raw)

        df, _ = self.cache.get_or_fetch_dataframe(cache_key, _fetch, ttl_days=ttl, allow_empty=False)
        return df

    def macro_vix(self, *, start: str, end: str, force_refresh: bool = False) -> pl.DataFrame:
        """Return VIX-based macro features."""

        cache_key = f"macro_vix_{start}_{end}"
        ttl = self.settings.macro_cache_ttl_days

        def _fetch() -> pl.DataFrame:
            history = load_vix_history(
                start,
                end,
                parquet_path=self._macro_cache_file("vix", start, end),
                force_refresh=force_refresh,
            )
            return prepare_vix_features(history)

        try:
            features, _ = self.cache.get_or_fetch_dataframe(cache_key, _fetch, ttl_days=ttl, allow_empty=False)
        except ValueError as exc:
            logging.getLogger(__name__).warning(
                "VIX features unavailable for %s→%s (%s); returning empty frame",
                start,
                end,
                exc,
            )
            return pl.DataFrame()
        return features

    def macro_global_regime(self, *, start: str, end: str, force_refresh: bool = False) -> pl.DataFrame:
        """Return VVMD global regime features.

        Phase 1: 14 features from US and global markets:
        - SPY/QQQ volatility and momentum
        - VIX z-score
        - DXY (US Dollar) z-score
        - BTC relative momentum and volatility
        """
        cache_key = f"macro_global_regime_{start}_{end}"
        ttl = self.settings.macro_cache_ttl_days

        def _fetch() -> pl.DataFrame:
            history = load_global_regime_data(
                start,
                end,
                parquet_path=self._macro_cache_file("global_regime", start, end),
                force_refresh=force_refresh,
            )
            return prepare_vvmd_features(history)

        try:
            features, _ = self.cache.get_or_fetch_dataframe(cache_key, _fetch, ttl_days=ttl, allow_empty=False)
        except ValueError as exc:
            logging.getLogger(__name__).warning(
                "Global regime features unavailable for %s→%s (%s); returning empty frame",
                start,
                end,
                exc,
            )
            return pl.DataFrame()
        return features

    def margin_weekly(self, *, start: str, end: str) -> pl.DataFrame:
        """Return cached weekly margin interest."""

        cache_key = f"margin_weekly_{start}_{end}"
        ttl = self.settings.margin_weekly_cache_ttl_days

        def _fetch() -> pl.DataFrame:
            return self.fetcher.fetch_margin_weekly(start=start, end=end)

        df, _ = self.cache.get_or_fetch_dataframe(cache_key, _fetch, ttl_days=ttl, allow_empty=False)
        return df

    def dividends(self, *, start: str, end: str) -> pl.DataFrame:
        """Return dividend announcements enriched with availability metadata."""

        cache_key = f"dividend_{start}_{end}"
        ttl = self.settings.macro_cache_ttl_days

        def _fetch() -> pl.DataFrame:
            df = self.fetcher.fetch_dividends(start=start, end=end)
            if df.is_empty():
                logging.getLogger(__name__).warning("Dividend API returned zero rows for %s → %s", start, end)
                return df

            # Ensure dates are proper Date types
            if "RecordDate" in df.columns:
                df = df.with_columns(pl.col("RecordDate").cast(pl.Date, strict=False))

            return df

        df, _ = self.cache.get_or_fetch_dataframe(cache_key, _fetch, ttl_days=ttl, allow_empty=False)
        return df

    def fs_details(self, *, start: str, end: str) -> pl.DataFrame:
        """Return financial statement details."""

        cache_key = f"fs_details_{start}_{end}"
        ttl = self.settings.macro_cache_ttl_days

        def _fetch() -> pl.DataFrame:
            return self.fetcher.fetch_fs_details(start=start, end=end)

        df, _ = self.cache.get_or_fetch_dataframe(cache_key, _fetch, ttl_days=ttl, allow_empty=False)
        return df

    def listed_info(self, *, start: str, end: str) -> pl.DataFrame:
        """Return listed info daily snapshots (日次スナップショット).

        Note: J-Quants API supports date parameter for daily snapshots.
        This method fetches listed info for each date in the range.
        """
        cache_key = f"listed_info_{start}_{end}"
        ttl = self.settings.macro_cache_ttl_days

        def _fetch() -> pl.DataFrame:
            # Fetch listed info for each date in the range
            # J-Quants API supports date parameter for daily snapshots
            from ..utils import business_date_range

            dates = business_date_range(start, end)
            all_rows = []
            for date_str in dates:
                try:
                    # fetch_listed_info accepts date parameter (YYYY-MM-DD format)
                    raw = self.fetcher.fetch_listed_info(as_of=date_str)
                    if not raw.is_empty():
                        # Ensure Date column exists
                        if "Date" not in raw.columns:
                            raw = raw.with_columns(pl.lit(date_str, dtype=pl.Date).alias("Date"))
                        all_rows.append(raw)
                except Exception as exc:
                    # Log and continue for other dates
                    logger = logging.getLogger(__name__)
                    logger.debug(f"Failed to fetch listed_info for {date_str}: {exc}")
                    continue

            if not all_rows:
                return pl.DataFrame()

            # Combine all dates
            combined = pl.concat(all_rows)
            # Ensure Date column is Date type
            if "Date" in combined.columns:
                combined = combined.with_columns(pl.col("Date").cast(pl.Date, strict=False))
            return combined

        df, _ = self.cache.get_or_fetch_dataframe(cache_key, _fetch, ttl_days=ttl, allow_empty=False)
        return df

    def topix(self, *, start: str, end: str) -> pl.DataFrame:
        """Return TOPIX history for the given range."""

        cache_key = f"topix_{start}_{end}"
        ttl = self.settings.topix_cache_ttl_days

        def _fetch() -> pl.DataFrame:
            return self.fetcher.fetch_topix(start=start, end=end)

        df, _ = self.cache.get_or_fetch_dataframe(cache_key, _fetch, ttl_days=ttl)
        return df

    def futures_topix(
        self,
        *,
        start: str,
        end: str,
        topix_df: pl.DataFrame | None = None,
        trading_calendar: pl.DataFrame | None = None,
        force_refresh: bool = False,
    ) -> pl.DataFrame:
        """Return TOPIX futures features (P0: minimal set).

        Fetches TOPIXF futures data and generates features:
        - Market regime (direction/change)
        - Overnight/intraday decomposition
        - Volatility indicators
        - Open interest pressure
        - Term structure (front vs next)
        - Basis (futures vs spot)
        - Roll window flags

        All features use T+1 09:00 JST as-of availability.
        """
        cache_key = f"futures_topix_{start}_{end}"
        ttl = self.settings.macro_cache_ttl_days

        def _fetch() -> pl.DataFrame:
            raw_futures = self.fetcher.fetch_futures(start=start, end=end)
            normalized = load_futures(raw_futures, category="TOPIXF")
            features = build_futures_features(
                normalized,
                topix_df=topix_df,
                trading_calendar=trading_calendar,
            )
            return features

        try:
            features, _ = self.cache.get_or_fetch_dataframe(cache_key, _fetch, ttl_days=ttl, allow_empty=False)
        except ValueError as exc:
            logging.getLogger(__name__).warning(
                "TOPIX futures features unavailable for %s→%s (%s); returning empty frame",
                start,
                end,
                exc,
            )
            return pl.DataFrame()
        return features

    def options_daily(
        self,
        *,
        start: str,
        end: str,
        topix_df: pl.DataFrame | None = None,
        nk225_df: pl.DataFrame | None = None,
        trading_calendar: pl.DataFrame | None = None,
        force_refresh: bool = False,
    ) -> pl.DataFrame:
        """Return index option features (P0: minimal set).

        Fetches index option data (TOPIXE, NK225E) and generates features:
        - IV level (30D, 90D)
        - IV term structure (90D - 30D)
        - IV skew (±5% moneyness)
        - Put-Call ratio (OI and Volume)
        - VRP (Variance Risk Premium) - TODO: implement

        All features use T+1 09:00 JST as-of availability.
        """
        cache_key = f"options_daily_{start}_{end}"
        ttl = self.settings.macro_cache_ttl_days

        def _fetch() -> pl.DataFrame:
            raw_options = self.fetcher.fetch_options(start=start, end=end)
            normalized = load_options(raw_options, categories=["TOPIXE", "NK225E"])
            features = build_option_signals(
                normalized,
                topix_df=topix_df,
                nk225_df=nk225_df,
                trading_calendar=trading_calendar,
            )
            return features

        try:
            features, _ = self.cache.get_or_fetch_dataframe(cache_key, _fetch, ttl_days=ttl, allow_empty=False)
        except ValueError as exc:
            logging.getLogger(__name__).warning(
                "Index option daily features unavailable for %s→%s (%s); returning empty frame",
                start,
                end,
                exc,
            )
            return pl.DataFrame()
        return features

    def index_option_225(
        self,
        *,
        start: str,
        end: str,
        topix_df: pl.DataFrame | None = None,
        trading_calendar: pl.DataFrame | None = None,
        force_refresh: bool = False,
    ) -> pl.DataFrame:
        """Return Nikkei225 index option features (P0: minimal set).

        Fetches /option/index_option data and generates features:
        - ATM IV (near month, 30D synthetic)
        - VRP (IV - RV, IV / RV)
        - IV Term Structure (near vs next)
        - Put/Call sentiment (OI ratio, volume ratio)
        - Skew (25Δ approximation)
        - Days to expiration
        - Night → Day IV jump

        All features use T+0 15:10 JST as-of availability (day session close).
        Night session features use T+0 06:00 JST (separate flag).

        Performance: Raw API response cached with IPC format (Quick Wins optimization).
        - First fetch: ~2h42m (6+ years of data)
        - Subsequent fetches: <5s (cache hit)
        """
        # Cache raw API response separately (expensive operation)
        raw_cache_key = f"index_option_raw_{start}_{end}"
        ttl = self.settings.macro_cache_ttl_days

        def _fetch_raw() -> pl.DataFrame:
            logger = logging.getLogger(__name__)
            logger.info(
                "[INDEX OPTION] Cache miss for %s, fetching from API (this may take 2-3 hours for 6+ years)",
                raw_cache_key,
            )
            return self.fetcher.fetch_options(start=start, end=end)

        # Get or fetch raw options data (cached with IPC for 5x speedup)
        raw_options, cache_hit = self.cache.get_or_fetch_dataframe(
            raw_cache_key,
            _fetch_raw,
            ttl_days=ttl,
            prefer_ipc=True,  # Use Arrow IPC for faster reads (Quick Wins Task 1)
            allow_empty=False,
        )

        if cache_hit:
            logger = logging.getLogger(__name__)
            logger.info(
                "[INDEX OPTION] ✅ Cache hit for %s, loaded %d rows in <5s (saved ~2h42m)",
                raw_cache_key,
                len(raw_options),
            )

        # Process raw options into features (fast, <10s)
        normalized = load_index_option_225(raw_options)
        features = build_index_option_225_features(
            normalized,
            topix_df=topix_df,
            trading_calendar=trading_calendar,
        )

        return features

    def indices(self, *, start: str, end: str, codes: Sequence[str]) -> pl.DataFrame:
        """Return OHLC history for a set of index codes."""

        normalized_codes = ",".join(sorted(codes))
        cache_key = f"indices_{normalized_codes}_{start}_{end}"
        ttl = self.settings.topix_cache_ttl_days

        def _fetch() -> pl.DataFrame:
            return self.fetcher.fetch_indices(start=start, end=end, codes=codes)

        df, _ = self.cache.get_or_fetch_dataframe(cache_key, _fetch, ttl_days=ttl)
        return df

    def trading_breakdown(self, *, start: str, end: str) -> pl.DataFrame:
        """Return investor breakdown (buy/sell detail) data."""

        cache_key = f"breakdown_{start}_{end}"
        ttl = self.settings.trades_spec_cache_ttl_days

        def _fetch() -> pl.DataFrame:
            return self.fetcher.fetch_trading_breakdown(start=start, end=end)

        df, _ = self.cache.get_or_fetch_dataframe(cache_key, _fetch, ttl_days=ttl)
        return df

    def trades_spec(self, *, start: str, end: str) -> pl.DataFrame:
        """Return investor flow (trades spec) data."""

        cache_key = f"trades_spec_{start}_{end}"
        ttl = self.settings.trades_spec_cache_ttl_days

        def _fetch() -> pl.DataFrame:
            return self.fetcher.fetch_trades_spec(start=start, end=end)

        df, _ = self.cache.get_or_fetch_dataframe(cache_key, _fetch, ttl_days=ttl)
        return df

    def earnings(self, *, start: str, end: str) -> pl.DataFrame:
        """Return earnings announcement schedule."""

        cache_key = f"earnings_{start}_{end}"
        ttl = self.settings.cache_ttl_days_default

        def _fetch() -> pl.DataFrame:
            return self.fetcher.fetch_earnings(start=start, end=end)

        df, _ = self.cache.get_or_fetch_dataframe(cache_key, _fetch, ttl_days=ttl)
        return df

    def short_selling(
        self,
        *,
        start: str,
        end: str,
        business_days: Optional[list[str]] = None,
    ) -> pl.DataFrame:
        """Return short selling aggregates."""

        cache_key = f"short_{start}_{end}"
        ttl = self.settings.short_selling_cache_ttl_days

        def _fetch() -> pl.DataFrame:
            return self.fetcher.fetch_short_selling(start=start, end=end)

        df, _ = self.cache.get_or_fetch_dataframe(cache_key, _fetch, ttl_days=ttl)
        return df

    def short_positions(
        self,
        *,
        start: str,
        end: str,
    ) -> pl.DataFrame:
        """Return short selling positions (outstanding positions ≥0.5% disclosure threshold).

        Data from /markets/short_selling_positions endpoint.
        Published on day T at 17:30/18:00/19:00 JST, available at T+1 09:00 JST.
        """

        cache_key = f"short_positions_{start}_{end}"
        ttl = self.settings.short_selling_cache_ttl_days

        def _fetch() -> pl.DataFrame:
            return self.fetcher.fetch_short_positions(start=start, end=end)

        df, _ = self.cache.get_or_fetch_dataframe(cache_key, _fetch, ttl_days=ttl)
        return df

    def sector_short_selling(
        self,
        *,
        start: str,
        end: str,
        business_days: Optional[list[str]] = None,
    ) -> pl.DataFrame:
        """Return sector-level short selling metrics."""

        cache_key = f"sector_short_{start}_{end}"
        ttl = self.settings.sector_short_cache_ttl_days

        def _fetch() -> pl.DataFrame:
            return self.fetcher.fetch_sector_short_selling(
                start=start,
                end=end,
                business_days=business_days,
            )

        df, _ = self.cache.get_or_fetch_dataframe(cache_key, _fetch, ttl_days=ttl)
        return df

    def prices_am(self, *, start: str, end: str) -> pl.DataFrame:
        """Return morning session (AM) price snapshots."""

        cache_key = f"prices_am_{start}_{end}"
        ttl = self.settings.cache_ttl_days_default

        def _fetch() -> pl.DataFrame:
            return self.fetcher.fetch_prices_am(start=start, end=end)

        df, _ = self.cache.get_or_fetch_dataframe(cache_key, _fetch, ttl_days=ttl)
        return df

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_margin_daily(df: pl.DataFrame) -> pl.DataFrame:
        """Normalize margin daily DataFrame to dataset schema."""

        if df.is_empty():
            return pl.DataFrame(
                {
                    "code": pl.Series([], dtype=pl.Utf8),
                    "date": pl.Series([], dtype=pl.Date),
                    "margin_balance": pl.Series([], dtype=pl.Float64),
                    "short_balance": pl.Series([], dtype=pl.Float64),
                }
            )

        rename_map: dict[str, str] = {}
        if "Code" in df.columns:
            rename_map["Code"] = "code"
        if "ApplicationDate" in df.columns:
            rename_map["ApplicationDate"] = "application_date"
        elif "Date" in df.columns:
            rename_map["Date"] = "application_date"
        if "PublishedDate" in df.columns:
            rename_map["PublishedDate"] = "published_date"
        if "LongMarginOutstanding" in df.columns:
            rename_map["LongMarginOutstanding"] = "margin_balance"
        if "ShortMarginOutstanding" in df.columns:
            rename_map["ShortMarginOutstanding"] = "short_balance"

        out = df.rename(rename_map)

        # Ensure required columns exist
        if "code" not in out.columns:
            out = out.with_columns(pl.lit(None).cast(pl.Utf8).alias("code"))
        if "application_date" not in out.columns:
            out = out.with_columns(pl.lit(None).cast(pl.Date).alias("application_date"))

        if "application_date" in out.columns:
            out = out.with_columns(
                pl.col("application_date")
                .cast(pl.Utf8, strict=False)
                .str.strptime(pl.Date, strict=False)
                .alias("application_date")
            )

        if "code" in out.columns:
            out = out.with_columns(pl.col("code").cast(pl.Utf8).alias("code"))

        if "published_date" in out.columns:
            out = out.with_columns(
                pl.col("published_date")
                .cast(pl.Utf8, strict=False)
                .str.strptime(pl.Date, strict=False)
                .alias("published_date")
            )

        for column, alias in [
            ("margin_balance", "margin_balance"),
            ("short_balance", "short_balance"),
        ]:
            if column in out.columns:
                out = out.with_columns(pl.col(column).cast(pl.Float64, strict=False).alias(alias))
            else:
                out = out.with_columns(pl.lit(None).cast(pl.Float64).alias(alias))

        if "published_date" in out.columns:
            out = out.with_columns(
                pl.col("published_date")
                .cast(pl.Utf8, strict=False)
                .str.strptime(pl.Date, strict=False)
                .alias("published_date")
            )

        out = out.with_columns(
            pl.when(pl.col("published_date").is_not_null())
            .then(pl.col("published_date"))
            .otherwise(pl.col("application_date"))
            .alias("date")
        )

        order_by = ["code", "application_date", "date"]
        if "published_date" in out.columns:
            order_by.append("published_date")
        ordered = out.sort(order_by)
        dedup_subset = ["code", "application_date"] if "application_date" in ordered.columns else ["code", "date"]
        ordered = ordered.unique(subset=dedup_subset, keep="last")

        select_cols = ["code", "date", "margin_balance", "short_balance"]
        if "application_date" in ordered.columns:
            select_cols.append("application_date")
        if "published_date" in ordered.columns:
            select_cols.append("published_date")
        return ordered.select(select_cols).sort(["code", "date"])

    def _macro_cache_file(self, prefix: str, start: str, end: str) -> Path:
        cache_dir = self.settings.data_cache_dir / "macro"
        cache_dir.mkdir(parents=True, exist_ok=True)
        name = f"{prefix}_{start.replace('-', '')}_{end.replace('-', '')}.parquet"
        return cache_dir / name
