"""Higher-level data source helpers backed by J-Quants advanced fetcher."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import polars as pl

from ..config import DatasetBuilderSettings, get_settings
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

        df, _ = self.cache.get_or_fetch_dataframe(cache_key, _fetch, ttl_days=ttl)
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

        features, _ = self.cache.get_or_fetch_dataframe(cache_key, _fetch, ttl_days=ttl)
        return features

    def margin_weekly(self, *, start: str, end: str) -> pl.DataFrame:
        """Return cached weekly margin interest."""

        cache_key = f"margin_weekly_{start}_{end}"
        ttl = self.settings.margin_weekly_cache_ttl_days

        def _fetch() -> pl.DataFrame:
            return self.fetcher.fetch_margin_weekly(start=start, end=end)

        df, _ = self.cache.get_or_fetch_dataframe(cache_key, _fetch, ttl_days=ttl)
        return df

    def topix(self, *, start: str, end: str) -> pl.DataFrame:
        """Return TOPIX history for the given range."""

        cache_key = f"topix_{start}_{end}"
        ttl = self.settings.topix_cache_ttl_days

        def _fetch() -> pl.DataFrame:
            return self.fetcher.fetch_topix(start=start, end=end)

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
            rename_map["ApplicationDate"] = "date"
        elif "Date" in df.columns:
            rename_map["Date"] = "date"
        if "LongMarginOutstanding" in df.columns:
            rename_map["LongMarginOutstanding"] = "margin_balance"
        if "ShortMarginOutstanding" in df.columns:
            rename_map["ShortMarginOutstanding"] = "short_balance"

        out = df.rename(rename_map)

        # Ensure required columns exist
        if "code" not in out.columns:
            out = out.with_columns(pl.lit(None).cast(pl.Utf8).alias("code"))
        if "date" not in out.columns:
            out = out.with_columns(pl.lit(None).cast(pl.Date).alias("date"))

        date_dtype = out.schema.get("date")
        if date_dtype != pl.Date:
            out = out.with_columns(
                pl.col("date").cast(pl.Utf8, strict=False).str.strptime(pl.Date, strict=False).alias("date")
            )

        if "code" in out.columns:
            out = out.with_columns(pl.col("code").cast(pl.Utf8).alias("code"))

        for column, alias in [
            ("margin_balance", "margin_balance"),
            ("short_balance", "short_balance"),
        ]:
            if column in out.columns:
                out = out.with_columns(pl.col(column).cast(pl.Float64, strict=False).alias(alias))
            else:
                out = out.with_columns(pl.lit(0.0).cast(pl.Float64).alias(alias))

        return out.select(["code", "date", "margin_balance", "short_balance"]).sort(["code", "date"])

    def _macro_cache_file(self, prefix: str, start: str, end: str) -> Path:
        cache_dir = self.settings.data_cache_dir / "macro"
        cache_dir.mkdir(parents=True, exist_ok=True)
        name = f"{prefix}_{start.replace('-', '')}_{end.replace('-', '')}.parquet"
        return cache_dir / name
