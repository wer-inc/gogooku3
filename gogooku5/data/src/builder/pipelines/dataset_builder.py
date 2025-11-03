"""High-level orchestration for dataset creation."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import polars as pl
from polars.datatypes import Date as PlDateType
from polars.datatypes import Datetime as PlDatetimeType

from ..api import (
    AxisDecider,
    DataSourceManager,
    JQuantsFetcher,
    ListedManager,
    QuotesFetcher,
    TradingCalendarFetcher,
)
from ..config import DatasetBuilderSettings, get_settings
from ..features.core.advanced import AdvancedFeatureEngineer
from ..features.core.flow.enhanced import FlowFeatureEngineer
from ..features.core.graph.features import GraphFeatureEngineer
from ..features.core.index.features import IndexFeatureEngineer
from ..features.core.margin.daily import MarginDailyFeatureEngineer
from ..features.core.peer.features import PeerFeatureEngineer
from ..features.core.quality_features_polars import (
    QualityFinancialFeaturesGeneratorPolars,
)
from ..features.core.sector.aggregation import SectorAggregationFeatures
from ..features.core.technical import TechnicalFeatureEngineer
from ..features.core.volatility import AdvancedVolatilityFeatures
from ..features.macro.engineer import MacroFeatureEngineer
from ..features.utils import (
    add_asof_timestamp,
    apply_adv_filter,
    compute_adv60_from_raw,
    get_raw_quotes_paths,
    interval_join_pl,
    prepare_snapshot_pl,
)
from ..utils import (
    CacheManager,
    DatasetArtifact,
    StorageClient,
    business_date_range,
    configure_logger,
    shift_trading_days,
)

LOGGER = configure_logger("builder.pipeline")


def _compute_auto_warmup(
    max_window: int = 252,
    horizon_max: int = 20,
    embargo: int = 5,
    asof_lag: int = 1,
    safety: int = 60,
) -> int:
    """Compute automatic warmup period for VVMD features.

    Args:
        max_window: Maximum rolling window (252d for yearly features)
        horizon_max: Maximum prediction horizon (20d default)
        embargo: Walk-forward embargo period (5d default)
        asof_lag: T+1 data availability lag (1d for US market data)
        safety: Safety margin for incomplete data (60d default)

    Returns:
        Total warmup period in trading days
    """
    return max_window + horizon_max + embargo + asof_lag + safety


@dataclass
class DatasetBuilder:
    """Coordinate data fetching, feature engineering, and parquet export."""

    settings: DatasetBuilderSettings = field(default_factory=get_settings)
    fetcher: JQuantsFetcher = field(default_factory=JQuantsFetcher)
    cache: CacheManager = field(default_factory=CacheManager)
    storage: StorageClient = field(default_factory=StorageClient)
    data_sources: DataSourceManager = field(default_factory=DataSourceManager)
    calendar_fetcher: TradingCalendarFetcher = field(default_factory=TradingCalendarFetcher)
    quality_features: QualityFinancialFeaturesGeneratorPolars = field(
        default_factory=QualityFinancialFeaturesGeneratorPolars
    )
    index_features: IndexFeatureEngineer = field(default_factory=IndexFeatureEngineer)
    margin_features: MarginDailyFeatureEngineer = field(default_factory=MarginDailyFeatureEngineer)
    sector_features: SectorAggregationFeatures = field(default_factory=SectorAggregationFeatures)
    peer_features: PeerFeatureEngineer = field(default_factory=PeerFeatureEngineer)
    volatility_features: AdvancedVolatilityFeatures = field(default_factory=AdvancedVolatilityFeatures)
    graph_features: GraphFeatureEngineer = field(default_factory=GraphFeatureEngineer)
    flow_features: FlowFeatureEngineer = field(default_factory=FlowFeatureEngineer)
    macro_features: MacroFeatureEngineer = field(default_factory=MacroFeatureEngineer)
    advanced_features: AdvancedFeatureEngineer = field(default_factory=AdvancedFeatureEngineer)
    technical_features: TechnicalFeatureEngineer = field(default_factory=TechnicalFeatureEngineer)

    def build(self, *, start: str, end: str, refresh_listed: bool = False) -> Path:
        """Build the dataset for the given date range."""

        # Phase 2 Patch A: Warmup period + final slice
        # Save output range
        start_out, end_out = start, end

        # Calculate warmup period with env var override support
        # Default horizons: [1, 5, 10, 20] → max = 20
        horizon_max = 20
        default_lookback = max(60, 20) + horizon_max + 5  # = 60 + 20 + 5 = 85 (legacy default)

        # Priority: WARMUP_DAYS env var > default
        env_warm = os.getenv("WARMUP_DAYS")
        if env_warm:
            try:
                if str(env_warm).lower() in ("auto", "automatic"):
                    lookback_days = _compute_auto_warmup(max_window=252, horizon_max=horizon_max)
                    warmup_policy = f"env:auto->{lookback_days}"
                else:
                    lookback_days = int(env_warm)
                    warmup_policy = f"env:{lookback_days}"
            except ValueError as e:
                raise SystemExit(f"WARMUP_DAYS must be int or 'auto', got: {env_warm}") from e
        else:
            lookback_days = default_lookback
            warmup_policy = f"default:{lookback_days}"

        LOGGER.info("[WARMUP] policy=%s (horizon_max=%d)", warmup_policy, horizon_max)

        # Calculate context start date (warmup period)
        try:
            start_ctx = shift_trading_days(start_out, -lookback_days)
            LOGGER.info("[WARMUP] Output range: %s to %s", start_out, end_out)
            LOGGER.info("[WARMUP] Context range: %s to %s (warmup: %d days)", start_ctx, end_out, lookback_days)
        except Exception as e:
            LOGGER.warning("[WARMUP] Failed to calculate warmup period: %s. Using original start date.", e)
            start_ctx = start_out

        # Use context dates for data fetching
        start, end = start_ctx, end_out

        LOGGER.info("Starting dataset build from %s to %s", start, end)
        LOGGER.info("[DEBUG] Step 1: Creating ListedManager...")
        listed_manager = ListedManager(fetcher=self.fetcher)
        LOGGER.info("[DEBUG] Step 2: Fetching listed symbols...")
        listed = listed_manager.refresh() if refresh_listed else listed_manager.listed()
        LOGGER.info("[DEBUG] Step 2 complete: Got %d symbols", len(listed) if listed else 0)
        if not listed:
            listed = listed_manager.refresh()
        LOGGER.info("[DEBUG] Step 3: Creating AxisDecider...")
        decider = AxisDecider.from_listed_symbols(listed)
        LOGGER.info("[DEBUG] Step 4: Choosing symbols...")
        symbols = decider.choose_symbols()
        LOGGER.info("[DEBUG] Step 4 complete: Chose %d symbols", len(symbols) if symbols else 0)

        # Phase 1-2 Fix: Fail fast if no symbols available
        if not symbols:
            error_msg = f"No listed symbols available for date range {start} to {end}. Cannot build dataset."
            LOGGER.error(error_msg)
            raise ValueError(error_msg)

        LOGGER.info("[DEBUG] Step 5: Preparing listed dataframe...")
        listed_df = self._prepare_listed_dataframe(listed)
        listed_df = listed_df.filter(pl.col("code").is_in(symbols))

        LOGGER.info("[DEBUG] Step 6: Building calendar...")
        calendar_df = self._business_calendar(start=start, end=end)
        LOGGER.info("[DEBUG] Step 6 complete: %d business days", len(calendar_df))

        LOGGER.info("[DEBUG] Step 7: Checking quotes cache...")
        cache_key = self._quotes_cache_key(symbols=symbols, start=start, end=end)
        quotes_df = self.cache.load_dataframe(cache_key)
        if quotes_df is None:
            LOGGER.info(
                "[DEBUG] Cache miss for key %s; fetching quotes for %d symbols",
                cache_key,
                len(symbols) if symbols else 0,
            )
            quotes_payload = self._fetch_quotes(symbols, start=start, end=end)
            LOGGER.info("[DEBUG] Step 7: Got %d quote records", len(quotes_payload))
            quotes_df = self._format_quotes(quotes_payload)

            # Phase 1-2 Fix: Fail fast if no quotes returned
            if quotes_df.height == 0:
                error_msg = f"No quotes data returned for {len(symbols)} symbols from {start} to {end}. Check API access or date range."
                LOGGER.error(error_msg)
                raise ValueError(error_msg)

            self.cache.save_dataframe(cache_key, quotes_df)
            cache_index = self.cache.load_index()
            cache_index[cache_key] = {
                "start": start,
                "end": end,
                "rows": quotes_df.height,
                "updated_at": datetime.utcnow().isoformat(),
            }
            self.cache.save_index(cache_index)
        else:
            LOGGER.info("Cache hit for key %s", cache_key)

        aligned_quotes = self._align_quotes_with_calendar(quotes_df, calendar_df, listed_df)

        # Phase 2 Patch D: Add as-of timestamp for T+1 data availability
        # This enables proper temporal joins (e.g., weekly margin, statements)
        # Data disclosed on day T becomes available at T 15:00 JST
        combined_df = add_asof_timestamp(aligned_quotes, date_col="date")
        LOGGER.info("[PATCH D] Added asof_ts column for temporal joins (15:00 JST)")

        margin_df = self._fetch_margin_data(start=start, end=end)
        combined_df = self._join_margin_data(combined_df, margin_df)
        combined_df = self._add_return_targets(combined_df)
        combined_df = self.sector_features.add_features(combined_df)
        combined_df = self.peer_features.add_features(combined_df)

        flow_df = self.data_sources.trades_spec(start=start, end=end)
        combined_df = self.flow_features.add_features(combined_df, flow_df)

        vix_features = self.data_sources.macro_vix(start=start, end=end)
        combined_df = self.macro_features.add_vix(combined_df, vix_features)

        global_regime_features = self.data_sources.macro_global_regime(start=start, end=end)
        combined_df = self.macro_features.add_global_regime(combined_df, global_regime_features)

        combined_df = self._shift_macro_features(combined_df)

        combined_df = self.volatility_features.add_features(combined_df)
        combined_df = self.graph_features.add_features(combined_df)
        combined_df = self.advanced_features.add_features(combined_df)
        combined_df = self.technical_features.add_features(combined_df)

        # Phase 2 Patch D: As-of joins for T+1 data availability
        combined_df = self._add_weekly_margin_features_asof(combined_df, calendar_df=calendar_df, start=start, end=end)
        combined_df = self._add_short_selling_features_asof(combined_df, calendar_df=calendar_df, start=start, end=end)

        combined_df = self._add_index_features(combined_df, start=start, end=end)
        enriched_df = self.quality_features.generate_quality_features(combined_df)

        # Phase 2 Patch A: Slice to output range (after all features computed with warmup context)
        LOGGER.info(
            "[WARMUP] Slicing from context (%d rows) to output range: %s to %s", enriched_df.height, start_out, end_out
        )

        # Convert date strings to Date type for comparison
        if "date" in enriched_df.columns:
            try:
                start_bound = datetime.strptime(start_out, "%Y-%m-%d").date()
                end_bound = datetime.strptime(end_out, "%Y-%m-%d").date()
            except ValueError as exc:  # pragma: no cover - defensive guard for misconfigured inputs
                raise ValueError(f"Invalid output date bounds: start={start_out!r}, end={end_out!r}") from exc
            enriched_df = enriched_df.filter(
                (pl.col("date") >= pl.lit(start_bound)) & (pl.col("date") <= pl.lit(end_bound))
            )
            LOGGER.info("[WARMUP] After slicing: %d rows", enriched_df.height)
        else:
            LOGGER.warning("[WARMUP] No 'date' column found, skipping slice")

        # Phase 2 Patch D: T-leak detection (skeleton)
        # TODO: Implement when as-of joins are used for weekly/snapshot data
        # from ..features.utils import _detect_temporal_leaks
        # _detect_temporal_leaks(enriched_df, ts_col="asof_ts", suffix="_snap")
        LOGGER.debug("[PATCH D] T-leak detection: Not yet implemented (skeleton only)")

        # Phase 2 Patch E: ADV filter (optional, controlled by MIN_ADV_YEN env var)
        enriched_df = self._apply_adv_filter(enriched_df, start=start, end=end)

        finalized = self._finalize_for_output(enriched_df)
        artifact = self._persist_dataset(finalized, start=start_out, end=end_out)
        self.storage.ensure_remote_symlink(target=str(artifact.latest_symlink))
        return artifact.latest_symlink

    def _fetch_quotes(self, codes: Iterable[str], *, start: str, end: str) -> List[dict[str, str]]:
        codes_list = list(codes)
        if not codes_list:
            LOGGER.warning("No symbols provided for quote fetch between %s and %s", start, end)
            return []
        fetcher = QuotesFetcher(client=self.fetcher)
        LOGGER.info("🚀 Using optimized quote fetching (auto-selects by-date or by-code axis)")
        return fetcher.fetch_batch_optimized(codes=codes_list, start=start, end=end)

    def _format_quotes(self, quotes: List[dict[str, str]]) -> pl.DataFrame:
        if not quotes:
            LOGGER.warning("No quotes returned for requested range")
            return pl.DataFrame({"code": [], "date": [], "close": []})

        df = pl.DataFrame(quotes)
        rename_map = {col: col.lower() for col in df.columns}
        df = df.rename(rename_map)
        if "sectorcode" in df.columns:
            df = df.rename({"sectorcode": "sector_code"})
        if "sector_code" not in df.columns:
            df = df.with_columns(pl.lit("UNKNOWN").alias("sector_code"))

        numeric_cols = [
            "close",
            "open",
            "high",
            "low",
            "volume",
            "turnovervalue",
            "adjustmentclose",
            "adjustmentopen",
            "adjustmenthigh",
            "adjustmentlow",
            "adjustmentvolume",
        ]
        present_numeric = [col for col in numeric_cols if col in df.columns]
        if present_numeric:
            df = df.with_columns([pl.col(col).cast(pl.Float64, strict=False).alias(col) for col in present_numeric])

        date_dtype = df.schema.get("date")
        if isinstance(date_dtype, PlDatetimeType):
            df = df.with_columns(pl.col("date").dt.date().alias("date"))
        elif isinstance(date_dtype, PlDateType):
            pass
        else:
            df = df.with_columns(pl.col("date").str.strptime(pl.Date, strict=False).alias("date"))

        columns = [
            "code",
            "sector_code",
            "date",
            "close",
            "open",
            "high",
            "low",
            "volume",
            "turnovervalue",
            "adjustmentclose",
            "adjustmentopen",
            "adjustmenthigh",
            "adjustmentlow",
            "adjustmentvolume",
        ]
        existing_columns = [col for col in columns if col in df.columns]
        return df.select(existing_columns)

    def _prepare_listed_dataframe(self, listed: List[dict[str, str]]) -> pl.DataFrame:
        if not listed:
            return pl.DataFrame(
                {
                    "code": pl.Series([], dtype=pl.Utf8),
                    "sector_code": pl.Series([], dtype=pl.Utf8),
                    "market_code": pl.Series([], dtype=pl.Utf8),
                }
            )

        df = pl.DataFrame(listed)
        # Normalize column names, keeping only lowercase version if duplicates exist
        lower_cols = {}
        for col in df.columns:
            lower_name = col.lower()
            if lower_name not in lower_cols:
                lower_cols[lower_name] = col
        # Select columns (preferring uppercase if both exist, then rename all to lowercase)
        df = df.select([lower_cols[name] for name in sorted(lower_cols.keys())])
        rename_map = {col: col.lower() for col in df.columns if col != col.lower()}
        if rename_map:
            df = df.rename(rename_map)
        # Rename market code (only if target doesn't exist)
        if "marketcode" in df.columns and "market_code" not in df.columns:
            df = df.rename({"marketcode": "market_code"})

        # Rename sector code (only if target doesn't exist, prefer sector33 > sector17 > section)
        if "sector_code" not in df.columns:
            if "sector33code" in df.columns:
                df = df.rename({"sector33code": "sector_code"})
            elif "sector17code" in df.columns:
                df = df.rename({"sector17code": "sector_code"})
            elif "section" in df.columns:
                df = df.rename({"section": "sector_code"})
        if "code" not in df.columns:
            raise ValueError("Listed metadata missing 'code' field")

        df = df.with_columns(pl.col("code").cast(pl.Utf8, strict=False).alias("code"))

        if "sector_code" in df.columns:
            df = df.with_columns(
                pl.col("sector_code").cast(pl.Utf8, strict=False).fill_null("UNKNOWN").alias("sector_code")
            )
        else:
            df = df.with_columns(pl.lit("UNKNOWN").alias("sector_code"))

        if "market_code" in df.columns:
            df = df.with_columns(pl.col("market_code").cast(pl.Utf8, strict=False).alias("market_code"))
        else:
            df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias("market_code"))

        return df.select(["code", "sector_code", "market_code"]).unique(subset=["code"], keep="first")

    def _business_calendar(self, *, start: str, end: str) -> pl.DataFrame:
        cache_key = f"calendar_{start}_{end}"

        def _fetch() -> pl.DataFrame:
            days = self._trading_calendar_dates(start, end)
            if not days:
                days = business_date_range(start, end)
            df = pl.DataFrame({"date": days})
            return df.with_columns(pl.col("date").str.strptime(pl.Date, strict=False))

        calendar_df, _ = self.cache.get_or_fetch_dataframe(
            cache_key,
            _fetch,
            ttl_days=self.settings.calendar_cache_ttl_days,
        )
        return calendar_df.select("date").unique().sort("date")

    def _trading_calendar_dates(self, start: str, end: str) -> list[str]:
        """Fetch trading days from J-Quants trading calendar API."""

        try:
            start_year = int(start[:4])
            end_year = int(end[:4])
        except ValueError:
            return []

        dates: set[str] = set()
        for year in range(start_year, end_year + 1):
            try:
                payload = self.calendar_fetcher.fetch_calendar(year=year)
            except Exception as exc:  # pragma: no cover - network/runtime failures
                LOGGER.debug("Failed to fetch trading calendar for %s: %s", year, exc)
                return []

            entries = payload.get("trading_calendar") or []
            for entry in entries:
                date_str = entry.get("Date") or entry.get("date")
                division = entry.get("HolidayDivision") or entry.get("holidayDivision")
                if not date_str:
                    continue
                if str(division) not in {"1", "2"}:  # 1: business day, 2: half session
                    continue
                if start <= date_str <= end:
                    dates.add(date_str)

        return sorted(dates)

    def _align_quotes_with_calendar(
        self,
        quotes: pl.DataFrame,
        calendar: pl.DataFrame,
        listed: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Phase 1-3 Fix: Use quotes-based approach instead of cross-join.

        Old behavior:
        - Cross-join all symbols × all dates → millions of rows
        - Left join quotes → most rows have NULL price data

        New behavior:
        - Start with actual quotes (real data only)
        - Enrich with listed metadata (sector_code, market_code)
        - No cross-join → only rows with actual trading data
        """
        # If no quotes, return empty DataFrame with correct schema
        if quotes.is_empty():
            return pl.DataFrame(
                {
                    "code": pl.Series([], dtype=pl.Utf8),
                    "date": pl.Series([], dtype=pl.Utf8),
                    "sector_code": pl.Series([], dtype=pl.Utf8),
                    "market_code": pl.Series([], dtype=pl.Utf8),
                    "close": pl.Series([], dtype=pl.Float64),
                }
            )

        # Prepare listed metadata
        if listed.is_empty():
            # Fallback: use codes from quotes
            listed = (
                quotes.select("code")
                .unique()
                .with_columns(
                    pl.lit("UNKNOWN").alias("sector_code"),
                    pl.lit(None).cast(pl.Utf8).alias("market_code"),
                )
            )
        else:
            # Rename to avoid conflicts
            listed = listed.rename({"sector_code": "sector_code_listed"})
            listed = listed.with_columns(
                pl.col("sector_code_listed").fill_null("UNKNOWN"),
                pl.col("market_code").cast(pl.Utf8, strict=False),
            )

        # Join quotes with listed metadata
        aligned = quotes.join(listed.select(["code", "sector_code_listed", "market_code"]), on="code", how="left")

        # Fill missing sector_code from listed or use UNKNOWN
        if "sector_code" in aligned.columns:
            # If quotes already has sector_code, prefer it, then fallback to listed
            aligned = aligned.with_columns(
                pl.when(pl.col("sector_code").is_null() | (pl.col("sector_code") == ""))
                .then(pl.coalesce(["sector_code_listed", pl.lit("UNKNOWN")]))
                .otherwise(pl.col("sector_code"))
                .alias("sector_code")
            )
        else:
            # Quotes doesn't have sector_code, use listed or UNKNOWN
            aligned = aligned.with_columns(pl.coalesce(["sector_code_listed", pl.lit("UNKNOWN")]).alias("sector_code"))

        # Clean up temporary columns
        if "sector_code_listed" in aligned.columns:
            aligned = aligned.drop("sector_code_listed")

        return aligned.sort(["code", "date"])

    def _fetch_margin_data(self, *, start: str, end: str) -> pl.DataFrame:
        return self.data_sources.margin_daily(start=start, end=end)

    def _add_return_targets(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Phase 2 Patch B: Add PAST returns only (no look-ahead).

        IMPORTANT: This now generates ret_prev_* (past returns) for features.
        Forward returns (ret_fwd_*) are generated separately as labels.
        """
        if df.is_empty() or "close" not in df.columns:
            return df

        if "adjustmentclose" in df.columns:
            base_price = pl.col("adjustmentclose").fill_null(pl.col("close"))
        else:
            base_price = pl.col("close")

        # Generate PAST returns (safe for features - no look-ahead)
        horizons = {
            "ret_prev_1d": 1,
            "ret_prev_5d": 5,
            "ret_prev_10d": 10,
            "ret_prev_20d": 20,
            "ret_prev_60d": 60,
            "ret_prev_120d": 120,
        }
        exprs = []
        for name, horizon in horizons.items():
            # shift(+horizon) = look backward (safe!)
            past = base_price.shift(+horizon).over("code")
            exprs.append(((base_price / (past + 1e-12)) - 1.0).alias(name))

        if "turnovervalue" in df.columns:
            dollar_volume = pl.col("turnovervalue").fill_null(pl.col("volume") * pl.col("close")).alias("dollar_volume")
        else:
            dollar_volume = (pl.col("volume") * pl.col("close")).alias("dollar_volume")

        return df.sort(["code", "date"]).with_columns(exprs + [dollar_volume])

    def _add_index_features(self, df: pl.DataFrame, *, start: str, end: str) -> pl.DataFrame:
        index_frames: list[tuple[str, pl.DataFrame]] = []

        try:
            topix_df = self.data_sources.topix(start=start, end=end)
            if not topix_df.is_empty():
                index_frames.append(("topix", topix_df))
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.debug("Failed to fetch TOPIX history: %s", exc)

        try:
            extra_indices = self.data_sources.indices(start=start, end=end, codes=["0101"])
            if not extra_indices.is_empty():
                index_frames.append(("nk225", extra_indices))
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.debug("Failed to fetch additional indices: %s", exc)

        if not index_frames:
            return df

        joined = df
        for prefix, frame in index_frames:
            idx = frame
            if "Date" in idx.columns:
                idx = idx.rename({col: col.lower() for col in idx.columns})

            if "date" not in idx.columns:
                continue

            if "code" in idx.columns:
                idx = idx.with_columns(pl.col("code").cast(pl.Utf8))
            else:
                idx = idx.with_columns(pl.lit(prefix.upper()).alias("code"))

            idx = idx.filter(pl.col("code").is_not_null())

            idx = idx.with_columns(pl.col("date").cast(pl.Date, strict=False))
            processed = self.index_features.build_features(idx)
            processed = processed.drop("code", strict=False)
            rename_map = {col: f"{prefix}_{col}" for col in processed.columns if col != "date"}
            processed = processed.rename(rename_map)
            joined = joined.join(processed, on="date", how="left")

        return joined

    def _add_short_selling_features(
        self,
        df: pl.DataFrame,
        *,
        start: str,
        end: str,
    ) -> pl.DataFrame:
        try:
            short_df = self.data_sources.short_selling(start=start, end=end)
            sector_short_df = self.data_sources.sector_short_selling(start=start, end=end)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.debug("Failed to fetch short selling data: %s", exc)
            return df

        features = df
        if "sector_code" in features.columns:
            features = features.with_columns(
                pl.col("sector_code").cast(pl.Utf8).str.to_uppercase().alias("sector_code")
            )
            features = features.with_columns(
                pl.when(pl.col("sector_code") == "UNKNOWN")
                .then(None)
                .otherwise(pl.col("sector_code"))
                .alias("_sector_code_clean")
            )
            features = features.with_columns(
                pl.col("_sector_code_clean").forward_fill().over("code").alias("_sector_code_clean")
            )
            features = (
                features.drop("sector_code")
                .with_columns(pl.col("_sector_code_clean").fill_null("UNKNOWN").alias("sector_code"))
                .drop("_sector_code_clean")
            )

        if not short_df.is_empty():
            if "Date" in short_df.columns:
                short_df = short_df.rename({col: col.lower() for col in short_df.columns})

            required = {
                "date",
                "sellingexcludingshortsellingturnovervalue",
                "shortsellingwithrestrictionsturnovervalue",
                "shortsellingwithoutrestrictionsturnovervalue",
            }
            if required.issubset(short_df.columns):
                market = (
                    short_df.group_by("date")
                    .agg(
                        [
                            pl.col("sellingexcludingshortsellingturnovervalue").sum().alias("sell_ex_short"),
                            pl.col("shortsellingwithrestrictionsturnovervalue").sum().alias("short_with"),
                            pl.col("shortsellingwithoutrestrictionsturnovervalue").sum().alias("short_without"),
                        ]
                    )
                    .with_columns(pl.col("date").cast(pl.Date, strict=False))
                )
                market = market.with_columns(
                    [
                        (
                            pl.when(
                                (pl.col("sell_ex_short") + pl.col("short_with") + pl.col("short_without")).abs() > 1e-6
                            )
                            .then(
                                (pl.col("short_with") + pl.col("short_without"))
                                / (pl.col("sell_ex_short") + pl.col("short_with") + pl.col("short_without"))
                            )
                            .otherwise(0.0)
                        ).alias("short_selling_ratio_market"),
                        (
                            pl.when(
                                (pl.col("sell_ex_short") + pl.col("short_with") + pl.col("short_without")).abs() > 1e-6
                            )
                            .then(
                                pl.col("short_with")
                                / (pl.col("sell_ex_short") + pl.col("short_with") + pl.col("short_without"))
                            )
                            .otherwise(0.0)
                        ).alias("short_selling_with_restrictions_ratio"),
                        (
                            pl.when(
                                (pl.col("sell_ex_short") + pl.col("short_with") + pl.col("short_without")).abs() > 1e-6
                            )
                            .then(
                                pl.col("short_without")
                                / (pl.col("sell_ex_short") + pl.col("short_with") + pl.col("short_without"))
                            )
                            .otherwise(0.0)
                        ).alias("short_selling_without_restrictions_ratio"),
                    ]
                ).select(
                    [
                        "date",
                        "short_selling_ratio_market",
                        "short_selling_with_restrictions_ratio",
                        "short_selling_without_restrictions_ratio",
                    ]
                )
                features = features.join(market, on="date", how="left")

        if not sector_short_df.is_empty():
            if "Date" in sector_short_df.columns:
                sector_short_df = sector_short_df.rename({col: col.lower() for col in sector_short_df.columns})

            required_sector = {
                "date",
                "sector33code",
                "sellingexcludingshortsellingturnovervalue",
                "shortsellingwithrestrictionsturnovervalue",
                "shortsellingwithoutrestrictionsturnovervalue",
            }
            if required_sector.issubset(sector_short_df.columns):
                sector = sector_short_df.select(
                    [
                        pl.col("date").cast(pl.Date, strict=False),
                        pl.col("sector33code").cast(pl.Utf8).str.to_uppercase().alias("sector_code"),
                        pl.col("sellingexcludingshortsellingturnovervalue")
                        .cast(pl.Float64)
                        .alias("sector_sell_ex_short"),
                        pl.col("shortsellingwithrestrictionsturnovervalue").cast(pl.Float64).alias("sector_short_with"),
                        pl.col("shortsellingwithoutrestrictionsturnovervalue")
                        .cast(pl.Float64)
                        .alias("sector_short_without"),
                    ]
                )
                sector = sector.with_columns(
                    [
                        (
                            pl.when(
                                (
                                    pl.col("sector_sell_ex_short")
                                    + pl.col("sector_short_with")
                                    + pl.col("sector_short_without")
                                ).abs()
                                > 1e-6
                            )
                            .then(
                                (pl.col("sector_short_with") + pl.col("sector_short_without"))
                                / (
                                    pl.col("sector_sell_ex_short")
                                    + pl.col("sector_short_with")
                                    + pl.col("sector_short_without")
                                )
                            )
                            .otherwise(0.0)
                        ).alias("sector_short_selling_ratio"),
                        (
                            pl.when(
                                (
                                    pl.col("sector_sell_ex_short")
                                    + pl.col("sector_short_with")
                                    + pl.col("sector_short_without")
                                ).abs()
                                > 1e-6
                            )
                            .then(
                                pl.col("sector_short_with")
                                / (
                                    pl.col("sector_sell_ex_short")
                                    + pl.col("sector_short_with")
                                    + pl.col("sector_short_without")
                                )
                            )
                            .otherwise(0.0)
                        ).alias("sector_short_with_ratio"),
                        (
                            pl.when(
                                (
                                    pl.col("sector_sell_ex_short")
                                    + pl.col("sector_short_with")
                                    + pl.col("sector_short_without")
                                ).abs()
                                > 1e-6
                            )
                            .then(
                                pl.col("sector_short_without")
                                / (
                                    pl.col("sector_sell_ex_short")
                                    + pl.col("sector_short_with")
                                    + pl.col("sector_short_without")
                                )
                            )
                            .otherwise(0.0)
                        ).alias("sector_short_without_ratio"),
                    ]
                ).select(
                    [
                        "date",
                        "sector_code",
                        "sector_short_selling_ratio",
                        "sector_short_with_ratio",
                        "sector_short_without_ratio",
                    ]
                )

                features = features.join(sector, on=["date", "sector_code"], how="left")
                features = (
                    features.sort(["sector_code", "date"])
                    .with_columns(
                        [
                            pl.col("sector_short_selling_ratio").forward_fill().over("sector_code"),
                            pl.col("sector_short_with_ratio").forward_fill().over("sector_code"),
                            pl.col("sector_short_without_ratio").forward_fill().over("sector_code"),
                        ]
                    )
                    .sort(["date", "code"])
                )

        return features

    def _add_weekly_margin_features(
        self,
        df: pl.DataFrame,
        *,
        start: str,
        end: str,
    ) -> pl.DataFrame:
        try:
            weekly = self.data_sources.margin_weekly(start=start, end=end)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.debug("Failed to fetch weekly margin data: %s", exc)
            return df

        if weekly.is_empty():
            return df

        if "Date" in weekly.columns:
            weekly = weekly.rename({col: col.lower() for col in weekly.columns})

        required = {
            "date",
            "code",
            "shortmargintradevolume",
            "longmargintradevolume",
        }
        if not required.issubset(set(weekly.columns)):
            return df

        weekly = weekly.select(
            [
                pl.col("date").cast(pl.Date, strict=False),
                pl.col("code").cast(pl.Utf8),
                pl.col("shortmargintradevolume").cast(pl.Float64),
                pl.col("longmargintradevolume").cast(pl.Float64),
            ]
        )

        weekly = weekly.rename(
            {
                "shortmargintradevolume": "weekly_margin_short_volume",
                "longmargintradevolume": "weekly_margin_long_volume",
            }
        )

        weekly = weekly.with_columns(
            [
                (pl.col("weekly_margin_long_volume") - pl.col("weekly_margin_short_volume")).alias(
                    "weekly_margin_net_volume"
                ),
                (
                    (pl.col("weekly_margin_long_volume") - pl.col("weekly_margin_short_volume"))
                    / (pl.col("weekly_margin_long_volume") + pl.col("weekly_margin_short_volume") + 1e-12)
                ).alias("weekly_margin_imbalance"),
                (pl.col("weekly_margin_long_volume") / (pl.col("weekly_margin_short_volume") + 1e-12)).alias(
                    "weekly_margin_long_short_ratio"
                ),
            ]
        )

        joined = df.join(weekly, on=["date", "code"], how="left")
        fill_cols = [
            "weekly_margin_short_volume",
            "weekly_margin_long_volume",
            "weekly_margin_net_volume",
            "weekly_margin_imbalance",
            "weekly_margin_long_short_ratio",
        ]
        return joined.with_columns([pl.col(col).forward_fill().over("code") for col in fill_cols])

    def _shift_macro_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply T+1 shift to all macro features.

        CRITICAL: Enforces T+1 availability for macro features.
        US market close (16:00 EST) = 06:00 JST next day → available for Japanese trading at 09:00 JST.
        """
        macro_cols = [col for col in df.columns if col.startswith("macro_")]
        if not macro_cols:
            return df

        macro = (
            df.select(["date"] + macro_cols)
            .unique(subset=["date"])
            .sort("date")
            .with_columns([pl.col(col).shift(1).alias(col) for col in macro_cols])
        )
        return df.drop(macro_cols, strict=False).join(macro, on="date", how="left")

    # ========================================================================
    # Phase 2 Patch D: As-of Join Methods (T+1 Data Availability)
    # ========================================================================

    def _add_weekly_margin_features_asof(
        self,
        df: pl.DataFrame,
        *,
        calendar_df: pl.DataFrame,
        start: str,
        end: str,
    ) -> pl.DataFrame:
        """
        Add weekly margin features with T+1 as-of join.

        Data published on day T becomes available at T+1 09:00 JST.
        Uses prepare_snapshot_pl + interval_join_pl for correct temporal alignment.
        """
        try:
            weekly = self.data_sources.margin_weekly(start=start, end=end)
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("Failed to fetch weekly margin data: %s", exc)
            return df

        if weekly.is_empty():
            LOGGER.debug("Weekly margin data is empty, skipping as-of join")
            return df

        # Normalize column names to lowercase
        if "Date" in weekly.columns:
            weekly = weekly.rename({col: col.lower() for col in weekly.columns})

        # Check for PublishedDate column (critical for T+1 join)
        if "publisheddate" not in weekly.columns:
            LOGGER.warning("Weekly margin data missing 'publisheddate', falling back to direct join")
            return self._add_weekly_margin_features(df, start=start, end=end)

        # Prepare margin data with T+1 availability
        weekly_prepared = prepare_snapshot_pl(
            weekly,
            published_date_col="publisheddate",
            trading_calendar=calendar_df,
            availability_hour=9,
            availability_minute=0,
        )

        # Select and rename columns
        required = {
            "code",
            "shortmargintradevolume",
            "longmargintradevolume",
            "available_ts",
        }
        if not required.issubset(set(weekly_prepared.columns)):
            LOGGER.warning("Weekly margin data missing required columns, skipping")
            return df

        weekly_processed = weekly_prepared.select(
            [
                pl.col("code").cast(pl.Utf8),
                pl.col("shortmargintradevolume").cast(pl.Float64).alias("weekly_margin_short_volume"),
                pl.col("longmargintradevolume").cast(pl.Float64).alias("weekly_margin_long_volume"),
                pl.col("available_ts"),
            ]
        )

        # Compute margin features
        weekly_processed = weekly_processed.with_columns(
            [
                (pl.col("weekly_margin_long_volume") - pl.col("weekly_margin_short_volume")).alias(
                    "weekly_margin_net_volume"
                ),
                (
                    (pl.col("weekly_margin_long_volume") - pl.col("weekly_margin_short_volume"))
                    / (pl.col("weekly_margin_long_volume") + pl.col("weekly_margin_short_volume") + 1e-12)
                ).alias("weekly_margin_imbalance"),
                (pl.col("weekly_margin_long_volume") / (pl.col("weekly_margin_short_volume") + 1e-12)).alias(
                    "weekly_margin_long_short_ratio"
                ),
            ]
        )

        # As-of join with T+1 availability
        result = interval_join_pl(
            backbone=df,
            snapshot=weekly_processed,
            on_code="code",
            backbone_ts="asof_ts",
            snapshot_ts="available_ts",
            strategy="backward",
            suffix="_margin",
        )

        # Drop metadata column
        result = result.drop(["available_ts_margin"], strict=False)

        LOGGER.info("[PATCH D] Joined weekly margin features with T+1 as-of join")
        return result

    def _add_short_selling_features_asof(
        self,
        df: pl.DataFrame,
        *,
        calendar_df: pl.DataFrame,
        start: str,
        end: str,
    ) -> pl.DataFrame:
        """
        Add short selling features with T+1 as-of join.

        Data published on day T becomes available at T+1 09:00 JST.
        Market-wide and sector-level short selling ratios.
        """
        try:
            short_df = self.data_sources.short_selling(start=start, end=end)
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("Failed to fetch short selling data: %s", exc)
            return df

        if short_df.is_empty():
            LOGGER.debug("Short selling data is empty, skipping as-of join")
            return df

        # Normalize column names
        if "Date" in short_df.columns:
            short_df = short_df.rename({col: col.lower() for col in short_df.columns})

        # Check for PublishedDate
        if "publisheddate" not in short_df.columns:
            LOGGER.warning("Short selling data missing 'publisheddate', falling back to direct join")
            return self._add_short_selling_features(df, start=start, end=end)

        # Prepare short selling data with T+1 availability
        short_prepared = prepare_snapshot_pl(
            short_df,
            published_date_col="publisheddate",
            trading_calendar=calendar_df,
            availability_hour=9,
            availability_minute=0,
        )

        # Compute market-wide short selling ratio
        required = {
            "sellingexcludingshortsellingturnovervalue",
            "shortsellingwithrestrictionsturnovervalue",
            "shortsellingwithoutrestrictionsturnovervalue",
            "available_ts",
        }

        if required.issubset(short_prepared.columns):
            # Aggregate to daily market level
            market = short_prepared.group_by("available_ts").agg(
                [
                    pl.col("sellingexcludingshortsellingturnovervalue").sum().alias("sell_ex_short"),
                    pl.col("shortsellingwithrestrictionsturnovervalue").sum().alias("short_with"),
                    pl.col("shortsellingwithoutrestrictionsturnovervalue").sum().alias("short_without"),
                ]
            )

            market = market.with_columns(
                [
                    (
                        pl.when((pl.col("sell_ex_short") + pl.col("short_with") + pl.col("short_without")).abs() > 1e-6)
                        .then(
                            (pl.col("short_with") + pl.col("short_without"))
                            / (pl.col("sell_ex_short") + pl.col("short_with") + pl.col("short_without"))
                        )
                        .otherwise(0.0)
                    ).alias("short_selling_ratio_market"),
                    (
                        pl.when((pl.col("sell_ex_short") + pl.col("short_with") + pl.col("short_without")).abs() > 1e-6)
                        .then(
                            pl.col("short_with")
                            / (pl.col("sell_ex_short") + pl.col("short_with") + pl.col("short_without"))
                        )
                        .otherwise(0.0)
                    ).alias("short_selling_with_restrictions_ratio"),
                    (
                        pl.when((pl.col("sell_ex_short") + pl.col("short_with") + pl.col("short_without")).abs() > 1e-6)
                        .then(
                            pl.col("short_without")
                            / (pl.col("sell_ex_short") + pl.col("short_with") + pl.col("short_without"))
                        )
                        .otherwise(0.0)
                    ).alias("short_selling_without_restrictions_ratio"),
                ]
            ).select(
                [
                    "available_ts",
                    "short_selling_ratio_market",
                    "short_selling_with_restrictions_ratio",
                    "short_selling_without_restrictions_ratio",
                ]
            )

            # As-of join market-wide ratios (no code, just by timestamp)
            result = df.join(
                market.sort("available_ts"),
                left_on="asof_ts",
                right_on="available_ts",
                how="left",
            ).drop(["available_ts"], strict=False)

            LOGGER.info("[PATCH D] Joined short selling market features with T+1 as-of join")
            return result

        return df

    def _join_margin_data(self, quotes: pl.DataFrame, margin: pl.DataFrame) -> pl.DataFrame:
        if margin.is_empty():
            return quotes
        if "adjustmentfactor" not in quotes.columns:
            quotes = quotes.with_columns(pl.lit(1.0).alias("adjustmentfactor"))

        adjustment_lookup = (
            quotes.select("code", "date", "adjustmentfactor")
            .rename({"date": "application_date", "adjustmentfactor": "margin_adjustment_factor"})
            .with_columns(pl.col("application_date").cast(pl.Date, strict=False).alias("application_date"))
        )
        if "application_date" not in margin.columns:
            margin = margin.with_columns(pl.col("date").alias("application_date"))
        margin_cast_exprs = []
        if "application_date" in margin.columns:
            margin_cast_exprs.append(pl.col("application_date").cast(pl.Date, strict=False).alias("application_date"))
        if "date" in margin.columns:
            margin_cast_exprs.append(pl.col("date").cast(pl.Date, strict=False).alias("date"))
        if margin_cast_exprs:
            margin = margin.with_columns(margin_cast_exprs)

        enriched_margin = margin.join(adjustment_lookup, on=["code", "application_date"], how="left")
        enriched_margin = enriched_margin.with_columns(
            [
                pl.when((pl.col("margin_adjustment_factor").is_null()) | (pl.col("margin_adjustment_factor") == 0))
                .then(pl.col("margin_balance"))
                .otherwise(pl.col("margin_balance") / pl.col("margin_adjustment_factor"))
                .alias("margin_balance"),
                pl.when((pl.col("margin_adjustment_factor").is_null()) | (pl.col("margin_adjustment_factor") == 0))
                .then(pl.col("short_balance"))
                .otherwise(pl.col("short_balance") / pl.col("margin_adjustment_factor"))
                .alias("short_balance"),
            ]
        )
        margin_features = self.margin_features.build_features(enriched_margin)
        return quotes.join(margin_features, on=["code", "date"], how="left")

    def _apply_adv_filter(self, df: pl.DataFrame, *, start: str, end: str) -> pl.DataFrame:
        """
        Phase 2 Patch E: Apply ADV filter if MIN_ADV_YEN is configured.

        Filters dataset to include only stocks with sufficient average daily volume.
        Uses raw quote cache to compute 60-day trailing ADV (current day excluded).

        Args:
            df: ML features dataframe
            start: Start date (for locating raw cache files)
            end: End date (for locating raw cache files)

        Returns:
            Filtered dataframe (or original if MIN_ADV_YEN not set)
        """
        if self.settings.min_adv_yen is None:
            LOGGER.debug("[PATCH E] ADV filter disabled (MIN_ADV_YEN not set)")
            return df

        LOGGER.info(
            "[PATCH E] Applying ADV filter: min_adv_yen=%d, min_periods=%d",
            self.settings.min_adv_yen,
            self.settings.adv_min_periods,
        )

        try:
            # 1) Find raw quotes cache files
            raw_paths = get_raw_quotes_paths(
                cache_dir=self.settings.data_cache_dir,
                pattern="quotes_*.parquet",
            )

            # 2) Compute ADV from raw data
            adv_df = compute_adv60_from_raw(
                raw_paths=raw_paths,
                min_periods=self.settings.adv_min_periods,
            )

            # 3) Apply filter
            filtered_df = apply_adv_filter(
                ml_df=df,
                adv_df=adv_df,
                min_adv_yen=self.settings.min_adv_yen,
                on_code="code",
                on_date="date",
            )

            return filtered_df

        except Exception as e:
            LOGGER.error("[PATCH E] ADV filter failed: %s", e, exc_info=True)
            LOGGER.warning("[PATCH E] Returning unfiltered data (ADV filter skipped)")
            return df

    def _quotes_cache_key(self, *, symbols: Iterable[str], start: str, end: str) -> str:
        sorted_symbols = sorted({str(symbol) for symbol in symbols}) if symbols else []
        if not sorted_symbols:
            return f"quotes_{start}_{end}_all"
        digest_input = ",".join(sorted_symbols).encode("utf-8")

        try:
            import hashlib
        except ImportError:  # pragma: no cover
            suffix = "_".join(sorted_symbols[:10])
            return f"quotes_{start}_{end}_{suffix}"

        digest = hashlib.md5(digest_input).hexdigest()  # nosec: cache key only
        return f"quotes_{start}_{end}_{digest}"

    def _finalize_for_output(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Normalize schema before persistence.

        Phase 2 Patch B+I: Comprehensive leak defense with deny list.
        Phase 2 Hotfix: Duplicate column handling (canonicalize OHLC, safe rename)
        """
        import re

        from ..features.utils.schema import (
            canonicalize_ohlc,
            enforce_unique_columns,
            safe_rename,
            validate_unique_columns,
        )

        # Phase 2 Patch I: Comprehensive feature deny list (regex-based)
        # Denies: forward returns, metadata columns, published dates, as-of timestamps
        deny_pattern = re.compile(
            r"^(ret_fwd_|returns_|feat_ret_)|"  # Forward-looking returns (including feat_ret_*)
            r"(_?available_ts|_?next_available_ts|_?disclosed_ts|_?asof_ts)|"  # As-of metadata
            r"(published.*date|application_date)|"  # Publication/application dates
            r"(_metadata|_leak_)$",  # Metadata/leak columns
            re.IGNORECASE,
        )

        # Find columns matching deny pattern
        denied_cols = [col for col in df.columns if deny_pattern.search(col)]

        if denied_cols:
            LOGGER.warning(
                "[PATCH I] Removing %d denied columns (leak defense): %s",
                len(denied_cols),
                denied_cols[:15],  # Log first 15
            )
            df = df.drop(denied_cols)

        # Hotfix Step 1: Canonicalize OHLC columns (coalesce Close/close/EndPrice → Close)
        df = canonicalize_ohlc(df)

        # Hotfix Step 2: Enforce unique columns (safety net for any remaining duplicates)
        df = enforce_unique_columns(df)

        # Hotfix Step 3: Safe rename (handles collision when rename target already exists)
        rename_map = {
            "code": "Code",
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "sector_code": "SectorCode",
            "market_code": "MarketCode",
        }
        out = safe_rename(df, rename_map)

        # Hotfix Step 4: Validate no duplicates remain (fail-fast)
        validate_unique_columns(out)

        if "Date" in out.columns:
            out = out.with_columns(pl.col("Date").cast(pl.Date, strict=False).alias("Date"))
        return out

    def _persist_dataset(self, df: pl.DataFrame, *, start: str, end: str) -> DatasetArtifact:
        """
        Phase 2 Patch F: Safe artifact writer with Fail-fast validation.

        Validates dataset before persistence:
        - Non-empty dataframe
        - Required core columns present
        - Core columns have sufficient non-null values (≥90%)
        """
        # Patch F: Fail-fast validation
        if df.is_empty():
            raise ValueError(
                "Cannot persist empty dataset. " "Check data pipeline for issues (start=%s, end=%s)" % (start, end)
            )

        # Check core columns
        core_columns = ["Close", "Open", "Volume"]  # Capital case (after finalize)
        missing_cores = [col for col in core_columns if col not in df.columns]
        if missing_cores:
            raise ValueError(
                f"Missing required core columns: {missing_cores}. " f"Available columns: {df.columns[:20]}"
            )

        # Check non-null rates for core columns
        for col in core_columns:
            non_null_rate = float(df.select(pl.col(col).is_not_null().mean()).item())
            if non_null_rate < 0.90:
                raise ValueError(
                    f"Core column '{col}' has insufficient non-null rate: {non_null_rate:.2%} < 90%. "
                    f"Check data quality for date range {start} to {end}"
                )

        LOGGER.info(
            "[PATCH F] Validated dataset: %d rows × %d cols, core columns ≥90%% non-null", df.height, len(df.columns)
        )

        # Persist dataset (atomic write)
        artifact = self.storage.write_dataset(df, start_date=start, end_date=end)
        LOGGER.info("Dataset stored at %s (metadata=%s)", artifact.parquet_path.name, artifact.metadata_path.name)
        return artifact
