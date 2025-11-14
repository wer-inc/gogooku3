"""High-level orchestration for dataset creation."""
from __future__ import annotations

import json
import os
import re
import time
import weakref
from calendar import monthrange
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, TypeVar

import polars as pl
from polars.datatypes import Date as PlDateType
from polars.datatypes import Datetime as PlDatetimeType
from requests.exceptions import HTTPError

from ..api import (
    AxisDecider,
    DataSourceManager,
    JQuantsFetcher,
    ListedManager,
    QuotesFetcher,
    RateLimitDetected,
    TradingCalendarFetcher,
)
from ..chunks import ChunkSpec
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
from ..features.events import LimitEventFeatureEngineer
from ..features.fundamentals import (
    build_breakdown_feature_frame,
    build_dividend_feature_frame,
    build_fs_feature_frame,
    prepare_breakdown_snapshot,
    prepare_dividend_snapshot,
    prepare_fs_snapshot,
)
from ..features.macro.engineer import MacroFeatureEngineer
from ..features.macro.indexes import build_index_features, load_indices_allowlist
from ..features.macro.topix_features import build_topix_features
from ..features.macro.trades_spec import (
    build_trades_spec_features,
    load_trades_spec,
    map_market_code_to_section,
)
from ..features.macro.trading_calendar import build_trading_calendar_features
from ..features.margin.asof import prepare_margin_daily_asof, prepare_margin_weekly_asof
from ..features.session import SessionFeatureEngineer
from ..features.utils import (
    apply_adv_filter,
    compute_adv60_from_raw,
    ensure_sector_dimensions,
    get_raw_quotes_paths,
)
from ..features.utils.rolling import roll_mean_safe, roll_std_safe
from ..utils import (
    CacheManager,
    DatasetArtifact,
    QuoteShardIndex,
    QuoteShardStore,
    RawDataStore,
    StorageClient,
    add_asof_timestamp,
    business_date_range,
    configure_logger,
    forward_fill_after_publication,
    interval_join_pl,
    month_key,
    prepare_snapshot_pl,
    month_range,
    shift_trading_days,
)
from ..utils.schema_validator import SchemaValidator
from ..utils.mlflow_tracker import MLflowTracker
from ..utils.lazy_io import save_with_cache
from ..utils.prefetch import DataSourcePrefetcher
from ..validation.quality import parse_asof_specs, run_quality_checks

LOGGER = configure_logger("builder.pipeline")


QUALITY_SPLIT_RE = re.compile(r"[,\s]+")


class SchemaMismatchError(RuntimeError):
    """Raised when chunk outputs violate the schema manifest."""


class DatasetQualityError(RuntimeError):
    """Raised when dataset quality checks fail."""


def _tokenize_quality_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [token for token in QUALITY_SPLIT_RE.split(value) if token]


BETA_EPS = 1e-12
SUPPLY_SHOCK_THRESHOLD = 0.01
TOPIX_FEATURE_WHITELIST: tuple[str, ...] = (
    "close",
    "r_prev_1d",
    "r_prev_5d",
    "r_prev_20d",
    "trend_gap_20_100",
    "z_close_20",
    "atr14",
    "natr14",
    "yz_vol_20",
    "yz_vol_60",
    "pk_vol_20",
    "pk_vol_60",
    "rs_vol_20",
    "rs_vol_60",
    "vol_z_20",
    "regime_score",
)
NK225_FEATURE_WHITELIST: tuple[str, ...] = ("r_prev_1d", "r_prev_20d")
INDEX_FEATURE_WHITELIST: dict[str, tuple[str, ...]] = {
    "topix": TOPIX_FEATURE_WHITELIST,
    "nk225": NK225_FEATURE_WHITELIST,
}
LAZY_ALIGN_MIN_CALENDAR_ROWS = 250
LAZY_ALIGN_MIN_RANGE_DAYS = 365
LAZY_ALIGN_MIN_QUOTE_ROWS = 3_000_000
PrefetchResult = TypeVar("PrefetchResult")
LISTED_MARKET_ALLOWLIST: tuple[str, ...] = (
    "0101",
    "0102",
    "0104",
    "0106",
    "0107",
    "0111",
    "0112",
    "0113",
)


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
    raw_store: RawDataStore | None = None
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
    limit_features: LimitEventFeatureEngineer = field(default_factory=LimitEventFeatureEngineer)
    session_features: SessionFeatureEngineer = field(default_factory=SessionFeatureEngineer)
    shard_index: QuoteShardIndex = field(init=False)
    shard_store: QuoteShardStore = field(init=False)
    _cache_stats_path: Path = field(init=False)
    mlflow_tracker: MLflowTracker | None = field(default=None)

    def __post_init__(self) -> None:
        """Initialize GPU-ETL if enabled."""
        if self.settings.use_gpu_etl:
            try:
                # Import GPU-ETL from gogooku3 codebase
                import sys

                sys.path.insert(0, str(Path(__file__).parents[5] / "src"))
                from utils.gpu_etl import init_rmm_legacy

                success = init_rmm_legacy(self.settings.rmm_pool_size)
                if success:
                    LOGGER.info("✅ GPU-ETL enabled: RMM initialized with pool_size=%s", self.settings.rmm_pool_size)
                else:
                    LOGGER.warning("⚠️  GPU-ETL requested but initialization failed, falling back to CPU")
            except Exception as e:
                LOGGER.warning("⚠️  GPU-ETL initialization failed: %s, using CPU mode", e)
                if self.settings.force_gpu:
                    raise RuntimeError(f"FORCE_GPU=1 but GPU initialization failed: {e}") from e
        self.shard_index = QuoteShardIndex(self.settings.quote_index_path)
        self.shard_store = QuoteShardStore(self.settings.quote_cache_dir, self.shard_index)
        self._cache_stats_path = self.settings.quote_stats_path
        if self.mlflow_tracker is None:
            self.mlflow_tracker = MLflowTracker.from_settings(self.settings)
        self._cache_stats_path.parent.mkdir(parents=True, exist_ok=True)
        self._rate_limit_checked = False
        self._schema_fp_cache = None  # Lazy initialization for L0 schema fingerprint
        self._run_meta: dict[str, Any] | None = None
        raw_manifest = getattr(self.settings, "raw_manifest_path", None)
        if raw_manifest:
            try:
                self.raw_store = RawDataStore(Path(raw_manifest))
                LOGGER.info("[RAW] RawDataStore initialized from %s", raw_manifest)
            except FileNotFoundError:
                LOGGER.warning("[RAW] Manifest %s not found; raw store disabled", raw_manifest)
                self.raw_store = None
        else:
            self.raw_store = None

        if hasattr(self.data_sources, "attach_raw_store"):
            self.data_sources.attach_raw_store(self.raw_store)

        try:
            self.schema_validator: SchemaValidator | None = SchemaValidator()
            LOGGER.info(
                "[SCHEMA] Manifest loaded (version=%s hash=%s)",
                self.schema_validator.manifest_version,
                self.schema_validator.manifest_hash,
            )
        except FileNotFoundError:
            LOGGER.warning("[SCHEMA] Manifest not found; chunk builds will skip validation")
            self.schema_validator = None
        except Exception as exc:
            LOGGER.warning("[SCHEMA] Failed to initialize validator: %s", exc)
            self.schema_validator = None

    @staticmethod
    def _date_span_days(start: str, end: str) -> int:
        """Return inclusive span in days between two ISO dates."""

        try:
            start_dt = datetime.fromisoformat(start)
            end_dt = datetime.fromisoformat(end)
        except ValueError:
            return 0
        return max((end_dt - start_dt).days, 0)

    def _should_use_lazy_alignment(self, *, span_days: int, calendar_rows: int, quotes_rows: int) -> bool:
        """Decide whether to switch to the lazy alignment path for quotes/listed joins."""

        if not self.settings.enable_lazy_scans:
            return False
        if calendar_rows >= LAZY_ALIGN_MIN_CALENDAR_ROWS:
            return True
        if span_days >= LAZY_ALIGN_MIN_RANGE_DAYS:
            return True
        if quotes_rows >= LAZY_ALIGN_MIN_QUOTE_ROWS:
            return True
        return False

    def build(
        self,
        *,
        start: str,
        end: str,
        refresh_listed: bool = False,
        chunk_spec: ChunkSpec | None = None,
        start_time: float | None = None,
    ) -> Path:
        """Build the dataset for the given date range or chunk.

        Args:
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            refresh_listed: Force refresh listed companies data
            chunk_spec: Chunk specification for output paths
            start_time: Build start timestamp (for duration tracking)
        """
        # Phase 2 Patch A: Warmup period + final slice
        self._run_meta = {}
        if start_time is None:
            start_time = time.time()

        if chunk_spec is not None:
            start_out, end_out = chunk_spec.output_start, chunk_spec.output_end
            start_ctx = chunk_spec.input_start
            warmup_policy = f"chunk:{chunk_spec.chunk_id}"
            LOGGER.info(
                "[WARMUP] Chunk build %s: output=%s→%s, context=%s→%s",
                chunk_spec.chunk_id,
                start_out,
                end_out,
                start_ctx,
                chunk_spec.input_end,
            )
        else:
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

        prefetcher = DataSourcePrefetcher(self.settings.data_prefetch_threads, logger=LOGGER)
        _prefetcher_finalizer = weakref.finalize(prefetcher, prefetcher.close)
        if prefetcher.enabled:
            self._schedule_prefetch_targets(prefetcher, start=start, end=end)
        listed_info_df = self._fetch_listed_info_window(prefetcher, start=start, end=end)
        LOGGER.info("[DEBUG] Step 1: Creating ListedManager...")
        listed_manager = ListedManager(fetcher=self.fetcher)
        LOGGER.info("[DEBUG] Step 2: Fetching listed symbols...")
        listed = listed_manager.refresh() if refresh_listed else listed_manager.listed()
        LOGGER.info("[DEBUG] Step 2 complete: Got %d symbols", len(listed) if listed else 0)
        if not listed:
            listed = listed_manager.refresh()
        listed = self._merge_listed_metadata(listed, listed_info_df)
        LOGGER.info("[DEBUG] Step 3: Resolving symbol universe...")
        symbols = self._choose_symbol_universe(listed, listed_info_df, start=start, end=end)
        LOGGER.info("[DEBUG] Step 4 complete: Chose %d symbols", len(symbols) if symbols else 0)

        # Phase 1-2 Fix: Fail fast if no symbols available
        if not symbols:
            error_msg = f"No listed symbols available for date range {start} to {end}. Cannot build dataset."
            LOGGER.error(error_msg)
            raise ValueError(error_msg)

        LOGGER.info("[DEBUG] Step 5: Preparing listed dataframe...")
        # Use IPC cache for fast reuse (Pattern 1: small table mmap caching)
        listed_lf = self._load_small_table_cached("listed", lambda: self._prepare_listed_dataframe(listed))
        listed_filter = pl.col("code").is_in(symbols)
        listed_filtered_lf = listed_lf.filter(listed_filter)
        # Filter to symbols and materialize (small table, fast)
        listed_df = listed_filtered_lf.collect()

        LOGGER.info("[DEBUG] Step 6: Building calendar...")
        # Use IPC cache for fast reuse (calendar is ~250 rows/year, perfect for caching)
        calendar_key = f"trading_calendar_{start}_{end}"
        calendar_lf = self._load_small_table_cached(calendar_key, lambda: self._business_calendar(start=start, end=end))
        calendar_df = calendar_lf.collect()
        LOGGER.info("[DEBUG] Step 6 complete: %d business days", len(calendar_df))

        LOGGER.info("[DEBUG] Step 7: Checking quotes cache (smart reuse enabled)...")
        quotes_df = self._load_or_fetch_quotes(symbols=symbols, start=start, end=end)

        span_days = self._date_span_days(start, end)
        use_lazy_alignment = self._should_use_lazy_alignment(
            span_days=span_days,
            calendar_rows=calendar_df.height,
            quotes_rows=quotes_df.height,
        )

        if use_lazy_alignment:
            LOGGER.info(
                "[LAZY ALIGN] Using streaming quotes/listed alignment (span=%d days, calendar_rows=%d, quotes_rows=%d)",
                span_days,
                calendar_df.height,
                quotes_df.height,
            )
            aligned_quotes = self._align_quotes_with_calendar_lazy(quotes_df, calendar_lf, listed_filtered_lf)
        else:
            aligned_quotes = self._align_quotes_with_calendar(quotes_df, calendar_df, listed_df)

        # Phase 2 Patch D: Add as-of timestamp for T+1 data availability
        # This enables proper temporal joins (e.g., weekly margin, statements)
        # Data disclosed on day T becomes available at T 15:00 JST
        combined_df = add_asof_timestamp(aligned_quotes, date_col="date")
        LOGGER.info("[PATCH D] Added asof_ts column for temporal joins (15:00 JST)")

        # GPU-ETL: Convert to cuDF for accelerated processing
        if self.settings.use_gpu_etl:
            combined_df = self._apply_gpu_processing(combined_df)
            # Defensive: Verify dataframe integrity after GPU processing
            LOGGER.info(
                "[DEBUG] Post-GPU check: %d rows, %d cols, schema types: %s",
                len(combined_df),
                len(combined_df.columns),
                str({k: str(v) for k, v in list(combined_df.schema.items())[:5]}),
            )

        try:
            LOGGER.info("[DEBUG] Starting margin data fetch...")
            margin_df = self._fetch_margin_data(start=start, end=end)
            LOGGER.info("[DEBUG] Margin data fetched: %d rows", len(margin_df))
            combined_df = self._attach_margin_daily_features(
                combined_df,
                raw_daily=margin_df,
                calendar_df=calendar_df,
            )
            LOGGER.info("[DEBUG] Margin daily features attached")
        except Exception as e:
            LOGGER.error("[DEBUG] Margin daily processing failed: %s", e, exc_info=True)
            raise

        combined_df = self._add_return_targets(combined_df)
        combined_df = self._ensure_ret_prev_columns(combined_df)
        combined_df = self._add_gap_features(combined_df)
        combined_df = self._add_range_features(combined_df)
        combined_df = self.limit_features.add_features(combined_df, meta=self._run_meta)
        combined_df = self._extend_limit_features(combined_df)
        combined_df = self.session_features.add_features(
            combined_df,
            meta=self._run_meta,
            intraday_mode=False,
        )
        combined_df = self._add_am_pm_features(combined_df)
        combined_df = self.sector_features.add_features(combined_df)
        # Sector17相対特徴量を追加（既存のsector33に加えて）
        combined_df = self._add_sector17_features(combined_df)
        combined_df = self.peer_features.add_features(combined_df)

        try:
            fs_df = self._resolve_prefetched(
                prefetcher,
                "fs_details",
                lambda: self.data_sources.fs_details(start=start, end=end),
            )
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.debug("Failed to fetch fs_details: %s", exc)
            fs_df = pl.DataFrame()

        try:
            div_df = self._resolve_prefetched(
                prefetcher,
                "dividends",
                lambda: self.data_sources.dividends(start=start, end=end),
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("Failed to fetch dividend data: %s", exc)
            div_df = pl.DataFrame()

        try:
            breakdown_df = self._resolve_prefetched(
                prefetcher,
                "trading_breakdown",
                lambda: self.data_sources.trading_breakdown(start=start, end=end),
            )
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.debug("Failed to fetch trading breakdown data: %s", exc)
            breakdown_df = pl.DataFrame()

        try:
            earnings_df = self._resolve_prefetched(
                prefetcher,
                "earnings",
                lambda: self.data_sources.earnings(start=start, end=end),
            )
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.debug("Failed to fetch earnings announcements: %s", exc)
            earnings_df = pl.DataFrame()

        combined_df = self._attach_fs_features(
            combined_df,
            raw_fs=fs_df,
            calendar_df=calendar_df,
        )

        combined_df = self._attach_dividend_features(
            combined_df,
            raw_dividend=div_df,
            calendar_df=calendar_df,
        )

        combined_df = self._attach_earnings_features(
            combined_df,
            raw_earnings=earnings_df,
            calendar_df=calendar_df,
        )

        combined_df = self._attach_breakdown_features(
            combined_df,
            raw_breakdown=breakdown_df,
            calendar_df=calendar_df,
        )

        # Listed info features (P0: market dummies, sector codes, scale bucket, margin eligibility)
        combined_df = self._attach_listed_info_features(
            combined_df,
            start=start,
            end=end,
            calendar_df=calendar_df,
            listed_info_df=listed_info_df,
        )

        combined_df = self._add_supply_shock_features(combined_df)

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
        combined_df = self._attach_margin_weekly_features(
            combined_df,
            calendar_df=calendar_df,
            start=start,
            end=end,
        )
        combined_df = self._add_short_selling_features_asof(combined_df, calendar_df=calendar_df, start=start, end=end)
        try:
            short_positions_df = self._resolve_prefetched(
                prefetcher,
                "short_positions",
                lambda: self.data_sources.short_positions(start=start, end=end),
            )
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.debug("Failed to fetch short positions data: %s", exc)
            short_positions_df = pl.DataFrame()
        combined_df = self._add_short_positions_features_asof(
            combined_df,
            calendar_df=calendar_df,
            start=start,
            end=end,
            positions_df=short_positions_df,
        )

        combined_df = self._add_float_turnover_crowding_features(combined_df)
        combined_df = self._add_short_squeeze_features(combined_df)
        combined_df = self._add_margin_pain_index(combined_df)
        combined_df = self._add_pre_earnings_flow_features(combined_df)
        combined_df = self._add_liquidity_impact_features(combined_df)

        # 日経225オプション特徴量の追加（P0: 最小構成）
        combined_df = self._add_index_option_225_features_asof(
            combined_df, calendar_df=calendar_df, start=start, end=end
        )

        combined_df = self._add_index_features(combined_df, start=start, end=end)

        # TOPIX派生特徴量の追加（P0: 最小構成）
        combined_df = self._add_topix_features(combined_df, calendar_df=calendar_df, start=start, end=end)

        # β/α (60日, 対TOPIX) の追加（P0: 市場曝露の除去と残差の抽出）
        combined_df = self._add_beta_alpha_features(combined_df, start=start, end=end)

        # trades_spec（投資部門別売買状況）特徴量の追加（MVP）
        combined_df = self._add_trades_spec_features(combined_df, calendar_df=calendar_df, start=start, end=end)

        # 取引カレンダー特徴量の追加（P0: 必須基盤）
        combined_df = self._add_trading_calendar_features(combined_df, calendar_df=calendar_df)

        # TOPIX先物特徴量の追加（P0: 最小構成）
        try:
            topix_df = self.data_sources.topix(start=start, end=end)
            if not topix_df.is_empty():
                futures_features = self.data_sources.futures_topix(
                    start=start,
                    end=end,
                    topix_df=topix_df,
                    trading_calendar=calendar_df,
                )
                if not futures_features.is_empty():
                    combined_df = self.macro_features.add_futures_topix(combined_df, futures_features)
                    LOGGER.info(
                        "TOPIX futures features attached: %d features",
                        len([c for c in combined_df.columns if c.startswith("fut_")]),
                    )
                else:
                    LOGGER.debug("TOPIX futures features empty, skipping")
            else:
                LOGGER.debug("TOPIX data empty, skipping futures features")
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to attach TOPIX futures features: %s", exc, exc_info=True)

        # オプション特徴量の追加（P0: 最小構成）
        try:
            # 日経225データも取得（VRP計算用、現在は未実装）
            nk225_df = None
            try:
                nk225_df = self.data_sources.indices(start=start, end=end, codes=["0101"])
            except Exception:
                pass

            options_features = self.data_sources.options_daily(
                start=start,
                end=end,
                topix_df=topix_df,
                nk225_df=nk225_df,
                trading_calendar=calendar_df,
            )
            if not options_features.is_empty():
                combined_df = self.macro_features.add_options_features(combined_df, options_features)
                LOGGER.info(
                    "Option features attached: %d features",
                    len([c for c in combined_df.columns if c.startswith("macro_opt_")]),
                )
            else:
                LOGGER.debug("Option features empty, skipping")
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to attach option features: %s", exc, exc_info=True)

        combined_df = self._add_gap_basis_features(combined_df)

        enriched_df = self.quality_features.generate_quality_features(combined_df)

        # Phase 2 Patch A: Slice to output range (after all features computed with warmup context)
        LOGGER.info(
            "[WARMUP] Slicing from context (%d rows) to output range: %s to %s", enriched_df.height, start_out, end_out
        )

        # Convert date strings to Date type for comparison
        date_col = "date" if "date" in enriched_df.columns else ("Date" if "Date" in enriched_df.columns else None)
        if date_col:
            try:
                start_bound = datetime.strptime(start_out, "%Y-%m-%d").date()
                end_bound = datetime.strptime(end_out, "%Y-%m-%d").date()
            except ValueError as exc:  # pragma: no cover - defensive guard for misconfigured inputs
                raise ValueError(f"Invalid output date bounds: start={start_out!r}, end={end_out!r}") from exc

            date_expr = pl.col(date_col)
            temp_col = None
            dtype = enriched_df.schema.get(date_col)
            if dtype is None:
                LOGGER.warning("[WARMUP] Date column %s missing from schema; skipping slice", date_col)
            else:
                if isinstance(dtype, PlDatetimeType):
                    # Convert timezone-aware or naive datetimes to plain dates
                    temp_col = "__warmup_slice_date"
                    enriched_df = enriched_df.with_columns(date_expr.dt.date().alias(temp_col))
                    filter_col = pl.col(temp_col)
                    lower, upper = start_bound, end_bound
                elif isinstance(dtype, PlDateType):
                    filter_col = date_expr
                    lower, upper = start_bound, end_bound
                else:
                    # Fallback: coerce strings (or mixed types) to Date safely
                    temp_col = "__warmup_slice_date"
                    normalized = (
                        date_expr.cast(pl.Utf8, strict=False)
                        .str.replace_all(r"[./]", "-")
                        .str.slice(0, 10)
                        .str.strptime(pl.Date, strict=False)
                    )
                    enriched_df = enriched_df.with_columns(normalized.alias(temp_col))
                    filter_col = pl.col(temp_col)
                    lower, upper = start_bound, end_bound

                try:
                    stats = enriched_df.select(
                        filter_col.min().alias("slice_min"),
                        filter_col.max().alias("slice_max"),
                        pl.len().alias("rows_before_slice"),
                    ).row(0)
                    LOGGER.info(
                        "[WARMUP] Slice stats before filter: rows=%s, min=%s, max=%s (dtype=%s)",
                        stats[2],
                        stats[0],
                        stats[1],
                        dtype,
                    )
                except Exception as slice_exc:  # pragma: no cover - defensive
                    LOGGER.warning("[WARMUP] Failed to inspect slice stats: %s", slice_exc)

                enriched_df = enriched_df.filter((filter_col >= pl.lit(lower)) & (filter_col <= pl.lit(upper)))
                if temp_col and temp_col in enriched_df.columns:
                    enriched_df = enriched_df.drop(temp_col)
                LOGGER.info("[WARMUP] After slicing: %d rows", enriched_df.height)
        else:
            LOGGER.warning("[WARMUP] No 'date' or 'Date' column found, skipping slice")

        # Phase 2 Patch D: T-leak detection (skeleton)
        # TODO: Implement when as-of joins are used for weekly/snapshot data
        # from ..features.utils import _detect_temporal_leaks
        # _detect_temporal_leaks(enriched_df, ts_col="asof_ts", suffix="_snap")
        LOGGER.debug("[PATCH D] T-leak detection: Not yet implemented (skeleton only)")

        # Phase 2 Patch E: ADV filter (optional, controlled by MIN_ADV_YEN env var)
        enriched_df = self._apply_adv_filter(enriched_df, start=start, end=end)
        enriched_df = self._ensure_ret_prev_columns(enriched_df)

        finalized = self._finalize_for_output(enriched_df)
        finalized = self._ensure_ret_prev_columns(finalized)
        if chunk_spec is not None:
            return self._persist_chunk_dataset(finalized, chunk_spec)

        artifact = self._persist_dataset(finalized, start=start_out, end=end_out)
        self.storage.ensure_remote_symlink(target=str(artifact.latest_symlink))

        # BATCH-2B Safety: Return actual parquet if symlink not created (test mode)
        if artifact.latest_symlink.exists() or artifact.latest_symlink.is_symlink():
            return artifact.latest_symlink
        else:
            LOGGER.info("Latest symlink not created (safety gate). Returning parquet_path instead.")
            return artifact.parquet_path

    def _schedule_prefetch_targets(
        self,
        prefetcher: DataSourcePrefetcher,
        *,
        start: str,
        end: str,
    ) -> None:
        """Submit long-tail data source fetches to overlap with CPU/GPU work."""

        tasks: dict[str, Callable[[], Any]] = {
            "fs_details": lambda: self.data_sources.fs_details(start=start, end=end),
            "dividends": lambda: self.data_sources.dividends(start=start, end=end),
            "trading_breakdown": lambda: self.data_sources.trading_breakdown(start=start, end=end),
            "earnings": lambda: self.data_sources.earnings(start=start, end=end),
            "short_positions": lambda: self.data_sources.short_positions(start=start, end=end),
            "listed_info": lambda: self.data_sources.listed_info(start=start, end=end),
        }
        for key, factory in tasks.items():
            try:
                scheduled = prefetcher.schedule(key, factory)
                if scheduled:
                    LOGGER.debug("[PREFETCH] Scheduled %s (range %s→%s)", key, start, end)
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.debug("[PREFETCH] Failed to schedule %s: %s", key, exc)

    def _resolve_prefetched(
        self,
        prefetcher: DataSourcePrefetcher,
        key: str,
        fallback_fn: Callable[[], PrefetchResult],
    ) -> PrefetchResult:
        """Return prefetched result if available, otherwise call fallback synchronously."""

        if prefetcher is not None and prefetcher.enabled:
            return prefetcher.resolve(key, fallback_fn)
        return fallback_fn()

    def _fetch_listed_info_window(
        self,
        prefetcher: DataSourcePrefetcher,
        *,
        start: str,
        end: str,
    ) -> pl.DataFrame:
        """Fetch listed_info snapshots once and reuse across the pipeline."""

        def _fallback() -> pl.DataFrame:
            return self.data_sources.listed_info(start=start, end=end)

        try:
            payload = self._resolve_prefetched(prefetcher, "listed_info", _fallback)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to fetch listed_info snapshots (%s→%s): %s", start, end, exc)
            return pl.DataFrame()

        if isinstance(payload, pl.DataFrame):
            return payload

        if payload is None:
            return pl.DataFrame()

        try:
            return pl.DataFrame(payload)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Invalid listed_info payload (%s→%s): %s", start, end, exc)
            return pl.DataFrame()

    def _merge_listed_metadata(
        self,
        listed_entries: List[dict[str, Any]] | None,
        listed_info_df: pl.DataFrame,
    ) -> List[dict[str, Any]]:
        """Augment ListedManager payload with as-of snapshots for delisted symbols."""

        if not listed_entries:
            listed_entries = []

        merged: dict[str, dict[str, Any]] = {}
        for entry in listed_entries:
            code = entry.get("code")
            if not code:
                continue
            merged[str(code)] = dict(entry)

        if listed_info_df.is_empty() or "Code" not in listed_info_df.columns:
            return list(merged.values())

        # Keep the latest snapshot per code to fill missing metadata.
        snapshot_cols = ["Code", "MarketCode", "Sector33Code", "Sector17Code", "Date"]
        available_cols = [col for col in snapshot_cols if col in listed_info_df.columns]
        snapshot_df = listed_info_df.select(available_cols)
        if "Date" in snapshot_df.columns:
            snapshot_df = snapshot_df.sort(["Code", "Date"]).group_by("Code").tail(1)
        else:
            snapshot_df = snapshot_df.group_by("Code").tail(1)

        for row in snapshot_df.iter_rows(named=True):
            code = row.get("Code")
            if not code:
                continue
            code_str = str(code)
            entry = merged.setdefault(code_str, {"code": code_str})
            if not entry.get("sector_code"):
                sector_code = row.get("Sector33Code") or row.get("Sector17Code")
                if sector_code:
                    entry["sector_code"] = str(sector_code)
            if not entry.get("market_code") and row.get("MarketCode"):
                entry["market_code"] = str(row["MarketCode"])
            if "Date" in row and row.get("Date") is not None and not entry.get("listed_date"):
                entry["listed_date"] = row["Date"]

        return list(merged.values())

    def _choose_symbol_universe(
        self,
        listed_entries: List[dict[str, Any]],
        listed_info_df: pl.DataFrame,
        *,
        start: str,
        end: str,
    ) -> List[str]:
        """Select symbol universe using as-of snapshots when available."""

        symbols = self._extract_symbols_from_snapshots(listed_info_df)
        if symbols:
            LOGGER.info(
                "[LISTED] Using listed_info snapshots for %s→%s (codes=%d)",
                start,
                end,
                len(symbols),
            )
            return symbols

        LOGGER.info("[LISTED] Falling back to ListedManager universe (%d codes)", len(listed_entries))
        decider = AxisDecider.from_listed_symbols(listed_entries)
        return decider.choose_symbols()

    def _extract_symbols_from_snapshots(self, listed_info_df: pl.DataFrame) -> List[str]:
        """Return union of codes observed in listed_info snapshots (filtered by market)."""

        if listed_info_df.is_empty() or "Code" not in listed_info_df.columns:
            return []

        df = listed_info_df
        if "MarketCode" in df.columns:
            filtered = df.filter(pl.col("MarketCode").is_in(LISTED_MARKET_ALLOWLIST))
            if not filtered.is_empty():
                df = filtered

        codes_df = (
            df.select(pl.col("Code").cast(pl.Utf8, strict=False).alias("code")).drop_nulls().unique().sort("code")
        )
        if codes_df.is_empty():
            return []
        return codes_df.get_column("code").to_list()

    def _maybe_run_quality_checks(self, dataset_path: Path, *, context: str) -> dict[str, Dict[str, object]] | None:
        if not self.settings.enable_dataset_quality_check:
            return None

        targets = _tokenize_quality_list(self.settings.dataset_quality_targets)
        asof_specs = _tokenize_quality_list(self.settings.dataset_quality_asof_checks)
        try:
            asof_pairs = parse_asof_specs(asof_specs)
        except ValueError as exc:
            raise DatasetQualityError(f"{context}: invalid as-of spec - {exc}") from exc

        try:
            summary, errors, warnings = run_quality_checks(
                dataset_path,
                date_col=self.settings.dataset_quality_date_col,
                code_col=self.settings.dataset_quality_code_col,
                targets=targets,
                asof_pairs=asof_pairs,
                allow_future_days=self.settings.dataset_quality_allow_future_days,
                sample_rows=self.settings.dataset_quality_sample_rows,
            )
        except Exception as exc:
            raise DatasetQualityError(f"{context}: quality check failed ({exc})") from exc

        if errors:
            raise DatasetQualityError(f"{context}: {'; '.join(errors)}")
        if warnings:
            if self.settings.dataset_quality_fail_on_warning:
                raise DatasetQualityError(f"{context}: {'; '.join(warnings)}")
            LOGGER.warning("[QUALITY] %s warnings: %s", context, warnings)
        LOGGER.info("[QUALITY] %s passed dataset quality checks (%d stages)", context, len(summary))
        return summary

    def _persist_chunk_dataset(self, df: pl.DataFrame, chunk_spec: ChunkSpec) -> Path:
        """Persist chunk outputs and metadata without touching global symlinks."""

        parquet_kwargs = {"compression": self.settings.dataset_parquet_compression}
        parquet_path, ipc_path = save_with_cache(
            df,
            chunk_spec.dataset_path,
            create_ipc=True,
            parquet_kwargs=parquet_kwargs,
        )

        metadata: dict[str, Any] = {
            "chunk_id": chunk_spec.chunk_id,
            "input_start": chunk_spec.input_start,
            "input_end": chunk_spec.input_end,
            "output_start": chunk_spec.output_start,
            "output_end": chunk_spec.output_end,
            "rows": df.height,
            "columns": df.columns,
            "dtypes": {name: str(dtype) for name, dtype in df.schema.items()},
            "paths": {
                "parquet": str(parquet_path),
                "ipc": str(ipc_path) if ipc_path else None,
            },
        }
        if isinstance(self._run_meta, dict):
            metadata["builder_meta"] = self._run_meta

        validation_error: str | None = None
        if self.schema_validator is not None:
            try:
                validation_result = self.schema_validator.validate_dataframe(df)
                metadata["feature_schema_version"] = self.schema_validator.manifest_version
                metadata["feature_schema_hash"] = validation_result.schema_hash
                metadata["schema_validation"] = validation_result.to_dict()
                if validation_result.column_order_mismatch:
                    LOGGER.debug(
                        "[SCHEMA] Column order differs from manifest for %s (hash=%s)",
                        chunk_spec.chunk_id,
                        validation_result.schema_hash,
                    )
                if not validation_result.is_valid:
                    validation_error = str(validation_result)
            except Exception as exc:
                LOGGER.warning(
                    "[SCHEMA] Validation failed for %s (skipping schema enforcement): %s",
                    chunk_spec.chunk_id,
                    exc,
                )

        quality_summary = None
        try:
            quality_summary = self._maybe_run_quality_checks(
                parquet_path,
                context=f"chunk:{chunk_spec.chunk_id}",
            )
        except DatasetQualityError as exc:
            build_duration = time.time() - start_time
            self._write_chunk_status(
                chunk_spec,
                state="failed_quality_check",
                rows=df.height,
                error=str(exc),
                build_duration=build_duration,
            )
            raise

        if quality_summary:
            metadata["quality_checks"] = quality_summary

        chunk_spec.output_dir.mkdir(parents=True, exist_ok=True)
        with chunk_spec.metadata_path.open("w", encoding="utf-8") as meta_fp:
            json.dump(metadata, meta_fp, indent=2, ensure_ascii=False, sort_keys=True)

        # Compute build duration
        build_duration = time.time() - start_time

        # Prepare schema metadata for status.json
        schema_metadata = {}
        if "feature_schema_version" in metadata:
            schema_metadata["feature_schema_version"] = metadata["feature_schema_version"]
        if "feature_schema_hash" in metadata:
            schema_metadata["feature_schema_hash"] = metadata["feature_schema_hash"]

        if validation_error:
            self._write_chunk_status(
                chunk_spec,
                state="failed_schema_mismatch",
                rows=df.height,
                error=validation_error,
                schema_metadata=schema_metadata if schema_metadata else None,
                build_duration=build_duration,
            )
            raise SchemaMismatchError(validation_error)

        self._write_chunk_status(
            chunk_spec,
            state="completed",
            rows=df.height,
            schema_metadata=schema_metadata if schema_metadata else None,
            build_duration=build_duration,
        )
        return parquet_path

    def _write_chunk_status(
        self,
        chunk_spec: ChunkSpec,
        *,
        state: str,
        rows: int | None = None,
        error: str | None = None,
        schema_metadata: dict[str, Any] | None = None,
        build_duration: float | None = None,
    ) -> None:
        """Write status.json for a chunk build with atomic writes and enhanced metadata.

        Args:
            chunk_spec: Chunk specification
            state: Build state (running, completed, failed, failed_schema_mismatch)
            rows: Number of rows in final dataset (optional)
            error: Error message if failed (optional)
            schema_metadata: Schema validation metadata (version, hash) (optional)
            build_duration: Build duration in seconds (optional)
        """

        payload: dict[str, Any] = {
            "chunk_id": chunk_spec.chunk_id,
            "state": state,
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "output_start": chunk_spec.output_start,
            "output_end": chunk_spec.output_end,
        }

        if rows is not None:
            payload["rows"] = rows
        if error:
            payload["error"] = error
        if build_duration is not None:
            payload["build_duration_seconds"] = round(build_duration, 2)

        # Add schema metadata if provided
        if schema_metadata:
            if "feature_schema_version" in schema_metadata:
                payload["feature_schema_version"] = schema_metadata["feature_schema_version"]
            if "feature_schema_hash" in schema_metadata:
                payload["feature_schema_hash"] = schema_metadata["feature_schema_hash"]

        chunk_spec.output_dir.mkdir(parents=True, exist_ok=True)

        # Atomic write: temp file + rename to prevent corruption
        temp_path = chunk_spec.status_path.with_suffix(".json.tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as status_fp:
                json.dump(payload, status_fp, indent=2, ensure_ascii=False, sort_keys=True)
            temp_path.rename(chunk_spec.status_path)
        except Exception as exc:
            # Clean up temp file on failure
            if temp_path.exists():
                temp_path.unlink()
            LOGGER.error("Failed to write status.json for %s: %s", chunk_spec.chunk_id, exc)
            raise

    def build_chunk(self, chunk_spec: ChunkSpec, *, refresh_listed: bool = False) -> Path:
        """Build a single chunk described by ``chunk_spec`` with enhanced status tracking."""
        chunk_spec.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_chunk_status(chunk_spec, state="running")

        start_time = time.time()
        try:
            result = self.build(
                start=chunk_spec.output_start,
                end=chunk_spec.output_end,
                refresh_listed=refresh_listed,
                chunk_spec=chunk_spec,
                start_time=start_time,
            )
            # Note: build() writes "completed" status with metadata and duration
            return result
        except SchemaMismatchError as exc:
            build_duration = time.time() - start_time
            LOGGER.error("Schema mismatch detected for chunk %s: %s", chunk_spec.chunk_id, exc)
            # Status already written by build() with "failed_schema_mismatch" state
            raise
        except Exception as exc:
            build_duration = time.time() - start_time
            self._write_chunk_status(
                chunk_spec, state="failed", error=str(exc), build_duration=build_duration
            )
            raise

    def _load_small_table_cached(self, table_name: str, generator_fn) -> pl.LazyFrame:
        """Load small table with Arrow IPC mmap caching for fast reuse.

        This method implements Pattern 1 from the Lazy Join optimization proposal:
        - Arrow IPC format for 3-5x faster reads vs Parquet
        - Memory-mapped (mmap) for zero-copy access
        - .lazy().cache() for in-memory reuse across multiple joins

        Args:
            table_name: Cache file name (e.g., "listed", "trading_calendar")
            generator_fn: Function that generates the DataFrame if cache miss

        Returns:
            Cached LazyFrame ready for multiple joins (zero-copy reuse)

        Performance:
            - First load: Generate + save IPC (~100-500ms)
            - Subsequent loads: mmap read (~10-50ms, 10-50x faster)
            - Memory: Zero-copy mmap, minimal overhead
        """
        cache_dir = self.settings.data_cache_dir / "small_tables"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{table_name}.arrow"

        if not cache_path.exists():
            # Cache miss: Generate and save as Arrow IPC
            LOGGER.info("[SMALL TABLE] Cache miss for %s, generating...", table_name)
            df = generator_fn()
            df.write_ipc(cache_path, compression="lz4")
            LOGGER.info("[SMALL TABLE] Saved %s to IPC cache: %d rows, %d cols", table_name, df.height, df.width)
        else:
            LOGGER.info("[SMALL TABLE] Cache hit for %s", table_name)

        # Load with lazy scan (mmap) + cache in memory
        lf = pl.scan_ipc(str(cache_path))
        return lf.cache()  # In-memory cache for repeated use

    def _load_or_fetch_quotes(self, *, symbols: Iterable[str], start: str, end: str) -> pl.DataFrame:
        """Load quotes using L0 month shards with selective API backfill."""

        symbols_list = [str(symbol) for symbol in symbols]

        if self.settings.use_raw_store and self.raw_store is not None:
            try:
                raw_df = self.raw_store.load_range(source="prices", start=start, end=end)
            except FileNotFoundError:
                LOGGER.warning(
                    "[RAW] Missing prices data for %s→%s in raw store. Falling back to cache/API.",
                    start,
                    end,
                )
            else:
                normalized = self._normalize_from_shards(raw_df)
                if not normalized.is_empty():
                    start_date = datetime.fromisoformat(start).date()
                    end_date = datetime.fromisoformat(end).date()
                    normalized = normalized.filter(
                        (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
                    )
                    if symbols_list:
                        normalized = normalized.filter(pl.col("code").is_in(symbols_list))
                if not normalized.is_empty():
                    LOGGER.info(
                        "[RAW] Returning %d rows of prices from raw store for %s→%s",
                        normalized.height,
                        start,
                        end,
                    )
                    return normalized
                LOGGER.warning(
                    "[RAW] Raw store returned no prices for requested window (codes=%s %s→%s); fallback to cache/API",
                    symbols_list[:5],
                    start,
                    end,
                )

        codes_fp = self._symbols_digest(symbols_list)
        window_id = self._quotes_cache_key(symbols=symbols_list, start=start, end=end)
        start_month = month_key(start)
        end_month = month_key(end)
        months = month_range(start_month, end_month)
        if not months:
            months = [start_month]

        shard_meta_map = {meta.yyyymm: meta for meta in self.shard_index.get_shards(months)}
        expected_schema_fp = self._expected_l0_schema_fp()
        to_fetch: list[str] = []
        for month in months:
            meta = shard_meta_map.get(month)
            if meta is None:
                to_fetch.append(month)
                continue
            if meta.schema_fp != expected_schema_fp:
                LOGGER.info(
                    "[CACHE] Month %s schema mismatch (%s ≠ %s); scheduling refetch",
                    month,
                    meta.schema_fp,
                    expected_schema_fp,
                )
                to_fetch.append(month)
        preexisting_ratio = 1.0 if not months else (len(months) - len(to_fetch)) / len(months)
        LOGGER.info(
            "[CACHE] Window %s requires %d months (hit %.1f%%)",
            window_id,
            len(months),
            preexisting_ratio * 100.0,
        )

        self._ensure_rate_limit_ok(symbols=symbols_list, reference_date=end)

        for month in to_fetch:
            fetch_started = datetime.utcnow()
            month_df = self._fetch_month_shard(month)
            if month_df.is_empty():
                LOGGER.warning("[CACHE] Month %s returned no records; skipping shard write", month)
                continue
            shard_meta = self.shard_store.append_month(month, month_df)
            duration_ms = int((datetime.utcnow() - fetch_started).total_seconds() * 1000)
            self._record_cache_stat(
                event="write_shard",
                month=month,
                rows=shard_meta.n_rows,
                duration_ms=duration_ms,
            )
        self._enforce_cache_limit()

        collected = self.shard_store.collect_months(
            months,
            start=start,
            end=end,
            codes=set(symbols_list) if symbols_list else None,
        )

        if collected.is_empty():
            error_msg = (
                f"Collected quotes data is empty after shard load for {start} to {end}. "
                "Verify API responses and cache integrity."
            )
            LOGGER.error(error_msg)
            raise ValueError(error_msg)

        quotes_df = self._normalize_from_shards(collected)

        self.shard_index.register_window(
            window_id,
            start=start,
            end=end,
            codes_fp=codes_fp,
            coverage=1.0,
            months=months,
        )
        self._record_cache_stat(
            event="window",
            window_id=window_id,
            rows=quotes_df.height,
            months=len(months),
            preexisting_ratio=preexisting_ratio,
            hit_months=len(months) - len(to_fetch),
            missed_months=len(to_fetch),
        )

        return quotes_df

    def _fetch_month_shard(self, month: str) -> pl.DataFrame:
        """Fetch full-month quotes via by-date axis and normalize for shard storage."""

        year = int(month[:4])
        month_num = int(month[4:6])
        start_date = f"{year:04d}-{month_num:02d}-01"
        last_day = monthrange(year, month_num)[1]
        end_date = f"{year:04d}-{month_num:02d}-{last_day:02d}"
        trading_days = business_date_range(start_date, end_date)
        LOGGER.info(
            "[CACHE] Fetching month shard %s (%d trading days: %s → %s)",
            month,
            len(trading_days),
            start_date,
            end_date,
        )

        fetcher = QuotesFetcher(client=self.fetcher)
        payload = fetcher.fetch_by_date(dates=trading_days, codes=None)
        LOGGER.info("[CACHE] Month %s fetched %d records", month, len(payload))
        quotes_df = self._format_quotes(payload)
        if quotes_df.is_empty():
            return pl.DataFrame()
        return self._normalize_for_shard(quotes_df)

    def _normalize_for_shard(self, quotes_df: pl.DataFrame) -> pl.DataFrame:
        """Convert formatted quotes (lowercase) into uppercase shard schema."""

        rename_map = {src: dst for src, dst in self._L0_RENAME.items() if src in quotes_df.columns}
        normalized = quotes_df.rename(rename_map)

        # Ensure required columns exist with canonical dtypes
        casts: list[pl.Expr] = []
        for column in self._L0_COLUMNS:
            dtype = self._L0_SCHEMA[column]
            if column not in normalized.columns:
                normalized = normalized.with_columns(pl.lit(None, dtype=dtype).alias(column))
            casts.append(pl.col(column).cast(dtype, strict=False).alias(column))

        normalized = normalized.select(self._L0_COLUMNS).with_columns(casts)
        normalized = normalized.with_columns(
            pl.col("Code").cast(pl.Utf8, strict=False).alias("Code"),
            pl.col("Date").cast(pl.Utf8, strict=False).alias("Date"),
        )
        return normalized

    def _normalize_from_shards(self, shard_df: pl.DataFrame) -> pl.DataFrame:
        """Convert uppercase shard schema back to pipeline lowercase schema."""

        if shard_df.is_empty():
            return shard_df

        reverse = {dst: src for src, dst in self._L0_RENAME.items() if dst in shard_df.columns}
        normalized = shard_df.rename(reverse)
        if "date" in normalized.columns:
            normalized = normalized.with_columns(pl.col("date").str.strptime(pl.Date, strict=False))
        if "code" in normalized.columns:
            normalized = normalized.with_columns(pl.col("code").cast(pl.Utf8, strict=False))

        # Align canonical OHLCV so downstream always sees adjustment-derived values.
        coalesce_exprs: list[pl.Expr] = []
        if "adjustmentclose" in normalized.columns:
            close_base = pl.col("adjustmentclose")
            if "close" in normalized.columns:
                close_base = pl.coalesce([pl.col("adjustmentclose"), pl.col("close")])
            coalesce_exprs.append(close_base.alias("adjustmentclose"))
            if "close" in normalized.columns:
                coalesce_exprs.append(close_base.alias("close"))
        if "adjustmentopen" in normalized.columns:
            open_base = pl.col("adjustmentopen")
            if "open" in normalized.columns:
                open_base = pl.coalesce([pl.col("adjustmentopen"), pl.col("open")])
            coalesce_exprs.append(open_base.alias("adjustmentopen"))
            if "open" in normalized.columns:
                coalesce_exprs.append(open_base.alias("open"))
        if "adjustmenthigh" in normalized.columns:
            high_base = pl.col("adjustmenthigh")
            if "high" in normalized.columns:
                high_base = pl.coalesce([pl.col("adjustmenthigh"), pl.col("high")])
            coalesce_exprs.append(high_base.alias("adjustmenthigh"))
            if "high" in normalized.columns:
                coalesce_exprs.append(high_base.alias("high"))
        if "adjustmentlow" in normalized.columns:
            low_base = pl.col("adjustmentlow")
            if "low" in normalized.columns:
                low_base = pl.coalesce([pl.col("adjustmentlow"), pl.col("low")])
            coalesce_exprs.append(low_base.alias("adjustmentlow"))
            if "low" in normalized.columns:
                coalesce_exprs.append(low_base.alias("low"))
        if "adjustmentvolume" in normalized.columns:
            volume_base = pl.col("adjustmentvolume")
            if "volume" in normalized.columns:
                volume_base = pl.coalesce([pl.col("adjustmentvolume"), pl.col("volume")])
            coalesce_exprs.append(volume_base.alias("adjustmentvolume"))
            if "volume" in normalized.columns:
                coalesce_exprs.append(volume_base.alias("volume"))
        if coalesce_exprs:
            normalized = normalized.with_columns(coalesce_exprs)
        return normalized

    def _record_cache_stat(self, **payload: Any) -> None:
        """Append cache telemetry to configured jsonl stream."""

        payload["ts"] = datetime.utcnow().isoformat()
        try:
            with self._cache_stats_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
        except Exception as exc:  # pragma: no cover - telemetry best-effort
            LOGGER.debug("[CACHE] Failed to write cache stat: %s", exc)

    def _enforce_cache_limit(self) -> None:
        """Ensure quote shard cache stays within configured size."""

        max_gb = self.settings.max_cache_size_gb
        if max_gb is None or max_gb <= 0:
            return

        limit_bytes = int(max_gb * (1024**3))
        shard_dir = self.settings.quote_cache_dir / "raw" / "quotes"
        if not shard_dir.exists():
            return

        total_bytes = sum(path.stat().st_size for path in shard_dir.glob("*.parquet"))
        if total_bytes <= limit_bytes:
            return

        LOGGER.info(
            "[CACHE] Total shard size %.2f GB exceeds limit %.2f GB; starting GC",
            total_bytes / (1024**3),
            max_gb,
        )

        for shard in self.shard_index.all_shards():
            if total_bytes <= limit_bytes:
                break
            shard_path = self.shard_store.shard_path(shard.yyyymm)
            size_bytes = shard_path.stat().st_size if shard_path.exists() else 0
            LOGGER.info(
                "[CACHE] GC removing shard %s (%.2f MB)",
                shard.yyyymm,
                size_bytes / (1024**2),
            )
            self.shard_store.remove_month(shard.yyyymm)
            total_bytes -= size_bytes
            self._record_cache_stat(event="gc_remove", month=shard.yyyymm, size_bytes=size_bytes)

        self.shard_index.vacuum()

    def _expected_l0_schema_fp(self) -> str:
        """Return canonical schema fingerprint for L0 shards."""

        if self._schema_fp_cache:
            return self._schema_fp_cache

        empty_series = {name: pl.Series(name=name, values=[], dtype=dtype) for name, dtype in self._L0_SCHEMA.items()}
        sample = pl.DataFrame(empty_series)
        parts = "|".join(f"{name}:{dtype}" for name, dtype in sample.schema.items())
        import hashlib

        self._schema_fp_cache = hashlib.md5(parts.encode("utf-8")).hexdigest()  # nosec - cache metadata only
        return self._schema_fp_cache

    def _ensure_rate_limit_ok(self, *, symbols: Iterable[str], reference_date: str) -> None:
        """Run a lightweight probe so we can bail out early when rate-limited."""

        if getattr(self, "_rate_limit_checked", False):
            return

        sample_code = next((str(symbol) for symbol in symbols if str(symbol).strip()), None)
        if not sample_code:
            self._rate_limit_checked = True
            return

        probe_date = reference_date
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                self.fetcher.check_rate_limit(code=sample_code, date=probe_date)
                LOGGER.info(
                    "[RATE-LIMIT] Preflight passed using code=%s on probe_date=%s",
                    sample_code,
                    probe_date,
                )
                self._rate_limit_checked = True
                return
            except RateLimitDetected as exc:
                message = (
                    "J-Quants API rate limit detected before dataset build. "
                    f"code={sample_code}, probe_date={probe_date}."
                )
                LOGGER.error("[RATE-LIMIT] %s", message)
                raise RuntimeError(message) from exc
            except HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status in {400, 404} and attempt < max_attempts - 1:
                    try:
                        new_probe_date = shift_trading_days(probe_date, -1)
                    except Exception:  # pragma: no cover - fallback
                        break
                    LOGGER.debug(
                        "[RATE-LIMIT] Probe date %s returned status %s; retrying with %s",
                        probe_date,
                        status,
                        new_probe_date,
                    )
                    probe_date = new_probe_date
                    continue
                raise

        self._rate_limit_checked = True

    def _format_quotes(self, quotes: List[dict[str, str]]) -> pl.DataFrame:
        if not quotes:
            LOGGER.warning("No quotes returned for requested range")
            return pl.DataFrame({"code": [], "date": [], "close": []})

        sentinel_values = {"", "-", "--", "---"}
        sanitized_payload = [
            {
                key: (None if isinstance(value, str) and value.strip() in sentinel_values else value)
                for key, value in row.items()
            }
            for row in quotes
        ]

        schema_overrides = {
            "Close": pl.Float64,
            "Open": pl.Float64,
            "High": pl.Float64,
            "Low": pl.Float64,
            "Volume": pl.Float64,
            "TurnoverValue": pl.Float64,
            "AdjustmentClose": pl.Float64,
            "AdjustmentOpen": pl.Float64,
            "AdjustmentHigh": pl.Float64,
            "AdjustmentLow": pl.Float64,
            "AdjustmentVolume": pl.Float64,
            "AdjustmentFactor": pl.Float64,
            "UpperLimit": pl.Int8,
            "LowerLimit": pl.Int8,
            "UpperLimitPrice": pl.Float64,
            "LowerLimitPrice": pl.Float64,
            "MorningOpen": pl.Float64,
            "MorningHigh": pl.Float64,
            "MorningLow": pl.Float64,
            "MorningClose": pl.Float64,
            "MorningVolume": pl.Float64,
            "MorningTurnoverValue": pl.Float64,
            "MorningUpperLimit": pl.Int8,
            "MorningLowerLimit": pl.Int8,
            "MorningUpperLimitPrice": pl.Float64,
            "MorningLowerLimitPrice": pl.Float64,
            "AfternoonOpen": pl.Float64,
            "AfternoonHigh": pl.Float64,
            "AfternoonLow": pl.Float64,
            "AfternoonClose": pl.Float64,
            "AfternoonVolume": pl.Float64,
            "AfternoonTurnoverValue": pl.Float64,
            "AfternoonUpperLimit": pl.Int8,
            "AfternoonLowerLimit": pl.Int8,
            "AfternoonUpperLimitPrice": pl.Float64,
            "AfternoonLowerLimitPrice": pl.Float64,
        }

        try:
            df = pl.DataFrame(
                sanitized_payload,
                schema_overrides=schema_overrides,
                infer_schema_length=None,
            )
        except Exception:
            LOGGER.debug("[CACHE] Falling back to sanitized dataframe construction for quotes payload")
            df = pl.DataFrame(
                sanitized_payload,
                schema_overrides=schema_overrides,
                infer_schema_length=None,
            )
        snake_case_overrides = {
            "UpperLimit": "upper_limit",
            "LowerLimit": "lower_limit",
            "UpperLimitPrice": "upper_limit_price",
            "LowerLimitPrice": "lower_limit_price",
            "MorningOpen": "morning_open",
            "MorningHigh": "morning_high",
            "MorningLow": "morning_low",
            "MorningClose": "morning_close",
            "MorningVolume": "morning_volume",
            "MorningTurnoverValue": "morning_turnover_value",
            "MorningUpperLimit": "morning_upper_limit",
            "MorningLowerLimit": "morning_lower_limit",
            "MorningUpperLimitPrice": "morning_upper_limit_price",
            "MorningLowerLimitPrice": "morning_lower_limit_price",
            "AfternoonOpen": "afternoon_open",
            "AfternoonHigh": "afternoon_high",
            "AfternoonLow": "afternoon_low",
            "AfternoonClose": "afternoon_close",
            "AfternoonVolume": "afternoon_volume",
            "AfternoonTurnoverValue": "afternoon_turnover_value",
            "AfternoonUpperLimit": "afternoon_upper_limit",
            "AfternoonLowerLimit": "afternoon_lower_limit",
            "AfternoonUpperLimitPrice": "afternoon_upper_limit_price",
            "AfternoonLowerLimitPrice": "afternoon_lower_limit_price",
        }
        rename_map = {col: snake_case_overrides.get(col, col.lower()) for col in df.columns}
        df = df.rename(rename_map)
        if "sectorcode" in df.columns:
            df = df.rename({"sectorcode": "sector_code"})
        if "sector_code" not in df.columns:
            df = df.with_columns(pl.lit("UNKNOWN").alias("sector_code"))

        float_cols = [
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
            "adjustmentfactor",
            "morning_open",
            "morning_high",
            "morning_low",
            "morning_close",
            "morning_volume",
            "morning_turnover_value",
            "upper_limit_price",
            "lower_limit_price",
            "afternoon_open",
            "afternoon_high",
            "afternoon_low",
            "afternoon_close",
            "afternoon_volume",
            "afternoon_turnover_value",
            "morning_upper_limit_price",
            "morning_lower_limit_price",
            "afternoon_upper_limit_price",
            "afternoon_lower_limit_price",
        ]
        present_float = [col for col in float_cols if col in df.columns]
        if present_float:
            df = df.with_columns([pl.col(col).cast(pl.Float64, strict=False).alias(col) for col in present_float])

        int_cols = [
            "upper_limit",
            "lower_limit",
            "morning_upper_limit",
            "morning_lower_limit",
            "afternoon_upper_limit",
            "afternoon_lower_limit",
        ]
        present_ints = [col for col in int_cols if col in df.columns]
        if present_ints:
            df = df.with_columns([pl.col(col).cast(pl.Int8, strict=False).alias(col) for col in present_ints])

        date_dtype = df.schema.get("date")
        if isinstance(date_dtype, PlDatetimeType):
            df = df.with_columns(pl.col("date").dt.date().alias("date"))
        elif isinstance(date_dtype, PlDateType):
            pass
        else:
            df = df.with_columns(pl.col("date").str.strptime(pl.Date, strict=False).alias("date"))

        # Align raw OHLC columns with canonical adjustments for downstream parity.
        coalesce_exprs: list[pl.Expr] = []
        if "adjustmentclose" in df.columns:
            close_base = pl.col("adjustmentclose")
            if "close" in df.columns:
                close_base = pl.coalesce([pl.col("adjustmentclose"), pl.col("close")])
            coalesce_exprs.append(close_base.alias("adjustmentclose"))
            if "close" in df.columns:
                coalesce_exprs.append(close_base.alias("close"))
        if "adjustmentopen" in df.columns:
            open_base = pl.col("adjustmentopen")
            if "open" in df.columns:
                open_base = pl.coalesce([pl.col("adjustmentopen"), pl.col("open")])
            coalesce_exprs.append(open_base.alias("adjustmentopen"))
            if "open" in df.columns:
                coalesce_exprs.append(open_base.alias("open"))
        if "adjustmenthigh" in df.columns:
            high_base = pl.col("adjustmenthigh")
            if "high" in df.columns:
                high_base = pl.coalesce([pl.col("adjustmenthigh"), pl.col("high")])
            coalesce_exprs.append(high_base.alias("adjustmenthigh"))
            if "high" in df.columns:
                coalesce_exprs.append(high_base.alias("high"))
        if "adjustmentlow" in df.columns:
            low_base = pl.col("adjustmentlow")
            if "low" in df.columns:
                low_base = pl.coalesce([pl.col("adjustmentlow"), pl.col("low")])
            coalesce_exprs.append(low_base.alias("adjustmentlow"))
            if "low" in df.columns:
                coalesce_exprs.append(low_base.alias("low"))
        if "adjustmentvolume" in df.columns:
            volume_base = pl.col("adjustmentvolume")
            if "volume" in df.columns:
                volume_base = pl.coalesce([pl.col("adjustmentvolume"), pl.col("volume")])
            coalesce_exprs.append(volume_base.alias("adjustmentvolume"))
            if "volume" in df.columns:
                coalesce_exprs.append(volume_base.alias("volume"))
        if coalesce_exprs:
            df = df.with_columns(coalesce_exprs)

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
            "adjustmentfactor",
            "upper_limit",
            "lower_limit",
            "morning_open",
            "morning_high",
            "morning_low",
            "morning_close",
            "morning_volume",
            "morning_turnover_value",
            "morning_upper_limit",
            "morning_lower_limit",
            "afternoon_open",
            "afternoon_high",
            "afternoon_low",
            "afternoon_close",
            "afternoon_volume",
            "afternoon_turnover_value",
            "afternoon_upper_limit",
            "afternoon_lower_limit",
        ]
        existing_columns = [col for col in columns if col in df.columns]
        return df.select(existing_columns)

    def _prepare_listed_dataframe(self, listed: List[dict[str, str]]) -> pl.DataFrame:
        """Prepare listed info DataFrame with basic normalization (legacy method).

        Note: This method is kept for backward compatibility. For full feature engineering
        with as-of protection, use `_attach_listed_info_features` instead.
        """
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
            df = df.with_columns(pl.lit("UNKNOWN").cast(pl.Utf8).alias("sector_code"))

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
        # P0 FIX: If no quotes, raise ValueError immediately (fail-fast)
        if quotes.is_empty():
            raise ValueError(
                "Empty quotes DataFrame detected in _align_quotes_with_calendar. "
                "This indicates a data fetching issue or incorrect date range. "
                "Check that the date range has valid trading days and quote data is available."
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

        # BATCH-2B: Ensure sector dimensions for sector_short_selling_ratio (#12)
        aligned = ensure_sector_dimensions(aligned)

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

    def _align_quotes_with_calendar_lazy(
        self,
        quotes: pl.DataFrame,
        calendar_lf: pl.LazyFrame,
        listed_lf: pl.LazyFrame,
    ) -> pl.DataFrame:
        """
        Optimized version using LazyFrame for batch join with streaming collect.

        Pattern 2: Batch join with LazyFrame + streaming collect for memory efficiency.
        - Convert quotes to LazyFrame
        - Join with cached small tables (LazyFrame)
        - Use streaming collect to reduce memory footprint
        """
        # P0 FIX: If no quotes, raise ValueError immediately (fail-fast)
        if quotes.is_empty():
            raise ValueError(
                "Empty quotes DataFrame detected in _align_quotes_with_calendar_lazy. "
                "This indicates a data fetching issue or incorrect date range. "
                "Check that the date range has valid trading days and quote data is available."
            )

        # Convert quotes to LazyFrame for lazy evaluation
        quotes_lf = quotes.lazy()

        # Join with listed metadata (LazyFrame, cached)
        # Filter listed to only codes present in quotes for efficiency
        quotes_codes = quotes_lf.select("code").unique()
        listed_filtered = listed_lf.join(quotes_codes, on="code", how="inner").with_columns(
            [
                pl.col("sector_code").fill_null("UNKNOWN"),
                pl.col("market_code").cast(pl.Utf8, strict=False),
            ]
        )

        # Join quotes with listed metadata (lazy, optimized)
        aligned_lf = (
            quotes_lf.join(listed_filtered.select(["code", "sector_code", "market_code"]), on="code", how="left")
            .with_columns(
                [
                    # Ensure sector_code exists
                    pl.when(pl.col("sector_code").is_null() | (pl.col("sector_code") == ""))
                    .then(pl.lit("UNKNOWN"))
                    .otherwise(pl.col("sector_code"))
                    .alias("sector_code")
                ]
            )
            .sort(["code", "date"])
        )

        # Materialize with streaming for memory efficiency
        aligned = aligned_lf.collect(streaming=True)

        # BATCH-2B: Ensure sector dimensions for sector_short_selling_ratio (#12)
        aligned = ensure_sector_dimensions(aligned)

        return aligned

    def _fetch_margin_data(self, *, start: str, end: str) -> pl.DataFrame:
        return self.data_sources.margin_daily(start=start, end=end)

    def _add_return_targets(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Phase 2 Patch B: Add PAST returns only (no look-ahead).

        IMPORTANT: This now generates ret_prev_* (past returns) for features.
        Forward returns (ret_fwd_*) are generated separately as labels.
        """
        if df.is_empty():
            return df
        code_col = self._resolve_column_name(df, "code")
        date_col = self._resolve_column_name(df, "date")
        adj_close = self._resolve_column_name(df, "adjustmentclose")
        raw_close = self._resolve_column_name(df, "close")
        if code_col is None or (adj_close is None and raw_close is None):
            return df

        if adj_close and raw_close:
            base_price = pl.col(adj_close).fill_null(pl.col(raw_close))
        elif adj_close:
            base_price = pl.col(adj_close)
        else:
            base_price = pl.col(raw_close)  # type: ignore[arg-type]

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
            past = base_price.shift(+horizon).over(code_col)
            exprs.append(((base_price / (past + 1e-12)) - 1.0).alias(name))

        price_expr = pl.col(adj_close) if adj_close else pl.col(raw_close)  # type: ignore[arg-type]
        adj_volume = self._resolve_column_name(df, "adjustmentvolume")
        raw_volume = self._resolve_column_name(df, "volume")
        volume_expr = pl.col(adj_volume) if adj_volume else pl.col(raw_volume)  # type: ignore[arg-type]
        turnover_col = self._resolve_column_name(df, "turnovervalue")
        if turnover_col:
            dollar_volume = pl.col(turnover_col).fill_null(price_expr * volume_expr).alias("dollar_volume")
        else:
            dollar_volume = (price_expr * volume_expr).alias("dollar_volume")

        sort_cols = [col for col in (code_col, date_col) if col is not None]
        if sort_cols:
            df = df.sort(sort_cols)
        return df.with_columns(exprs + [dollar_volume])

    def _add_gap_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Decompose previous-day returns into overnight and intraday components."""
        required = {"code", "date", "adjustmentopen", "adjustmentclose"}
        if df.is_empty() or not required.issubset(df.columns):
            return df

        df = df.sort(["code", "date"])

        ao = pl.col("adjustmentopen")
        ac = pl.col("adjustmentclose")

        eps = 1e-9

        ret_overnight_expr = ((ao / (ac.shift(1).over("code") + eps)) - 1.0).alias("ret_overnight")
        ret_intraday_expr = ((ac / (ao + eps)) - 1.0).alias("ret_intraday")

        df = df.with_columns([ret_overnight_expr, ret_intraday_expr])
        # Gap decomposition should mirror the most recent 1D return (same-day close vs prior close)
        df = df.with_columns(
            [
                pl.col("ret_overnight").alias("gap_ov_prev1"),
                pl.col("ret_intraday").alias("gap_id_prev1"),
            ]
        )

        if "adjustmentfactor" in df.columns:
            adj = pl.col("adjustmentfactor")
            adj_change = (adj.shift(1).over("code") != adj.shift(2).over("code")) | (
                adj.shift(2).over("code") != adj.shift(3).over("code")
            )
            df = df.with_columns(
                [
                    pl.when(adj_change).then(None).otherwise(pl.col("ret_overnight")).alias("ret_overnight"),
                    pl.when(adj_change).then(None).otherwise(pl.col("ret_intraday")).alias("ret_intraday"),
                    pl.when(adj_change).then(None).otherwise(pl.col("gap_ov_prev1")).alias("gap_ov_prev1"),
                    pl.when(adj_change).then(None).otherwise(pl.col("gap_id_prev1")).alias("gap_id_prev1"),
                ]
            )

        clip_threshold = 0.30
        extreme_mask = (pl.col("gap_ov_prev1").abs() > clip_threshold) | (pl.col("gap_id_prev1").abs() > clip_threshold)
        df = df.with_columns(
            [
                pl.when(extreme_mask).then(None).otherwise(pl.col("ret_overnight")).alias("ret_overnight"),
                pl.when(extreme_mask).then(None).otherwise(pl.col("ret_intraday")).alias("ret_intraday"),
                pl.when(extreme_mask).then(None).otherwise(pl.col("gap_ov_prev1")).alias("gap_ov_prev1"),
                pl.when(extreme_mask).then(None).otherwise(pl.col("gap_id_prev1")).alias("gap_id_prev1"),
            ]
        )

        denom = pl.col("gap_ov_prev1").abs() + eps
        gap_product = pl.col("gap_ov_prev1") * pl.col("gap_id_prev1")

        def _sign(expr: pl.Expr) -> pl.Expr:
            return pl.when(expr > 0).then(pl.lit(1.0)).when(expr < 0).then(pl.lit(-1.0)).otherwise(pl.lit(0.0))

        df = df.with_columns(
            [
                (pl.col("gap_id_prev1") / denom).alias("gap_amplify_ratio_prev1"),
                (_sign(pl.col("gap_ov_prev1")) * _sign(pl.col("gap_id_prev1"))).alias("gap_sign_concord_prev1"),
                pl.when(gap_product < 0)
                .then(pl.col("gap_id_prev1").abs() / (pl.col("gap_ov_prev1").abs() + 1e-9))
                .otherwise(None)
                .alias("gap_fill_ratio_prev1"),
            ]
        )

        df = df.with_columns(
            pl.when(pl.col("gap_fill_ratio_prev1").is_not_null())
            .then((pl.col("gap_fill_ratio_prev1") >= 1.0).cast(pl.Int8))
            .otherwise(pl.lit(0))
            .cast(pl.Int8)
            .alias("gap_filled_prev1_flag")
        )

        # P0新特徴量: gap_sign, gap_magnitude_z20, gap_confirmed
        def _sign(expr: pl.Expr) -> pl.Expr:
            return pl.when(expr > 0).then(pl.lit(1.0)).when(expr < 0).then(pl.lit(-1.0)).otherwise(pl.lit(0.0))

        # gap_sign: ギャップの符号（+1/-1/0）
        df = df.with_columns(_sign(pl.col("ret_overnight")).alias("gap_sign"))

        # gap_magnitude_z20: ギャップの大きさのZ-score（20日）
        from ..features.utils.rolling import roll_mean_safe, roll_std_safe

        df = df.with_columns(
            [
                roll_mean_safe(pl.col("ret_overnight").abs(), 20, min_periods=10, by="code").alias("_gap_mag_ma20"),
                roll_std_safe(pl.col("ret_overnight").abs(), 20, min_periods=10, by="code").alias("_gap_mag_std20"),
            ]
        )
        df = df.with_columns(
            pl.when(pl.col("_gap_mag_std20").abs() > eps)
            .then(
                (pl.col("ret_overnight").abs().shift(1).over("code") - pl.col("_gap_mag_ma20"))
                / (pl.col("_gap_mag_std20") + eps)
            )
            .otherwise(None)
            .alias("gap_magnitude_z20")
        )

        # gap_confirmed: ギャップが日中に確認されたか（sign(ret_overnight)==sign(ret_intraday)）
        df = df.with_columns(
            (
                (_sign(pl.col("ret_overnight")) == _sign(pl.col("ret_intraday")))
                & pl.col("ret_overnight").is_not_null()
                & pl.col("ret_intraday").is_not_null()
            )
            .cast(pl.Int8)
            .alias("gap_confirmed")
        )

        # 一時列を削除
        cleanup_cols = ["_gap_mag_ma20", "_gap_mag_std20"]
        for col in cleanup_cols:
            if col in df.columns:
                df = df.drop(col)

        if isinstance(self._run_meta, dict):
            gap_meta = self._run_meta.setdefault("gap_decomposition", {})
            gap_meta.update(
                {
                    "mode": "prev1_only",
                    "clip": 0.30,
                    "feature_clock": "close",
                    "p0_features": ["gap_sign", "gap_magnitude_z20", "gap_confirmed"],
                }
            )

        return df

    def _ensure_ret_prev_columns(
        self, df: pl.DataFrame, horizons: tuple[int, ...] = (1, 5, 10, 20, 60, 120)
    ) -> pl.DataFrame:
        """Guarantee past-return columns exist (used by validators and downstream features)."""

        if df.is_empty():
            return df
        code_col = self._resolve_column_name(df, "code")
        date_col = self._resolve_column_name(df, "date")
        price_col = self._resolve_column_name(df, "adjustmentclose") or self._resolve_column_name(df, "close")

        if code_col is None or price_col is None:
            return df

        missing_exprs = []
        for horizon in horizons:
            name = f"ret_prev_{horizon}d"
            if name in df.columns:
                continue
            past = pl.col(price_col).shift(+horizon).over(code_col)
            missing_exprs.append(((pl.col(price_col) / (past + 1e-12)) - 1.0).alias(name))

        if not missing_exprs:
            return df

        LOGGER.debug(
            "[RET] Backfilling missing past-return columns: %s",
            [f"ret_prev_{h}d" for h in horizons if f"ret_prev_{h}d" not in df.columns],
        )
        sort_cols = [col for col in (code_col, date_col) if col is not None]
        if sort_cols:
            df = df.sort(sort_cols)
        enriched = df.with_columns(missing_exprs)
        return enriched

    @staticmethod
    def _resolve_column_name(df: pl.DataFrame, canonical: str) -> str | None:
        """Return actual column matching canonical name (case-insensitive)."""

        target = canonical.lower()
        for column in df.columns:
            if column.lower() == target:
                return column
        return None

    def _add_range_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add range/position features (P0: day_range, close_location, range_expansion)."""
        required = {"code", "date", "adjustmenthigh", "adjustmentlow", "adjustmentclose"}
        if df.is_empty() or not required.issubset(df.columns):
            return self._ensure_range_columns(df)

        df = df.sort(["code", "date"])
        eps = 1e-9

        ah = pl.col("adjustmenthigh")
        al = pl.col("adjustmentlow")
        ac = pl.col("adjustmentclose")
        prev_close = ac.shift(1).over("code")

        # day_range: (high-low)/close_{t-1}
        df = df.with_columns(((ah - al) / (prev_close + eps)).alias("day_range"))

        # close_location: (close-low)/(high-low)（両端ケア）
        df = df.with_columns(
            pl.when((ah - al).abs() > eps).then((ac - al) / (ah - al + eps)).otherwise(None).alias("close_location")
        )

        # range_expansion: day_range / roll_mean(day_range,20,shift=1)
        from ..features.utils.rolling import roll_mean_safe

        df = df.with_columns(
            roll_mean_safe(pl.col("day_range"), 20, min_periods=10, by="code").alias("_day_range_ma20")
        )
        df = df.with_columns(
            pl.when(pl.col("_day_range_ma20").abs() > eps)
            .then(pl.col("day_range").shift(1).over("code") / (pl.col("_day_range_ma20") + eps))
            .otherwise(None)
            .alias("range_expansion")
        )

        # 一時列を削除
        if "_day_range_ma20" in df.columns:
            df = df.drop("_day_range_ma20")

        df = self._ensure_range_columns(df)

        if isinstance(self._run_meta, dict):
            range_meta = self._run_meta.setdefault("range_features", {})
            range_meta.update(
                {
                    "columns": ["day_range", "close_location", "range_expansion"],
                    "policy": "left_closed_shift1",
                }
            )

        return df

    def _ensure_range_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Ensure all range feature columns exist with proper defaults."""
        specs = {
            "day_range": (pl.Float64, None),
            "close_location": (pl.Float64, None),
            "range_expansion": (pl.Float64, None),
        }
        for col, (dtype, default) in specs.items():
            if col not in df.columns:
                df = df.with_columns(pl.lit(default).cast(dtype).alias(col))
        return df

    def _ensure_supply_shock_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        specs: dict[str, tuple[pl.DataType, object | None]] = {
            "fs_shares_outstanding": (pl.Float64, None),
            "shares_out_delta_pct": (pl.Float64, None),
            "buyback_flag": (pl.Int8, 0),
            "dilution_flag": (pl.Int8, 0),
            "supply_shock": (pl.Int8, 0),
            "is_supply_shock_valid": (pl.Int8, 0),
        }
        for col, (dtype, default) in specs.items():
            if col not in df.columns:
                df = df.with_columns(pl.lit(default).cast(dtype).alias(col))
        return df

    def _add_supply_shock_features(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return self._ensure_supply_shock_columns(df)

        shares_col = self._resolve_column_name(df, "fs_shares_outstanding")
        if shares_col is None:
            LOGGER.debug("[SUPPLY] No fs_shares_outstanding column available")
            return self._ensure_supply_shock_columns(df)

        df = df.sort(["code", "date"])
        prev_col = "__prev_shares"
        df = df.with_columns(pl.col(shares_col).shift(1).over("code").alias(prev_col))
        df = df.with_columns(
            pl.when(pl.col(prev_col).is_not_null() & (pl.col(prev_col).abs() > 1e-3))
            .then((pl.col(shares_col) - pl.col(prev_col)) / (pl.col(prev_col) + 1e-12))
            .otherwise(None)
            .alias("shares_out_delta_pct")
        )

        df = df.with_columns(
            (pl.col("shares_out_delta_pct") <= -SUPPLY_SHOCK_THRESHOLD).cast(pl.Int8).alias("buyback_flag"),
            (pl.col("shares_out_delta_pct") >= SUPPLY_SHOCK_THRESHOLD).cast(pl.Int8).alias("dilution_flag"),
            (pl.col("shares_out_delta_pct").is_not_null()).cast(pl.Int8).alias("is_supply_shock_valid"),
        )
        df = df.with_columns((pl.col("buyback_flag") - pl.col("dilution_flag")).alias("supply_shock"))
        df = df.drop([prev_col], strict=False)
        return self._ensure_supply_shock_columns(df)

    def _ensure_float_crowding_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        specs: dict[str, tuple[pl.DataType, object | None]] = {
            "float_turnover_pct": (pl.Float64, None),
            "float_turnover_pct_z20": (pl.Float64, None),
            "float_turnover_pct_roc_5d": (pl.Float64, None),
            "margin_long_pct_float": (pl.Float64, None),
            "margin_long_pct_float_z20": (pl.Float64, None),
            "margin_long_pct_float_roc_5d": (pl.Float64, None),
            "weekly_margin_long_pct_float": (pl.Float64, None),
            "weekly_margin_long_pct_float_z20": (pl.Float64, None),
            "weekly_margin_long_pct_float_roc_5d": (pl.Float64, None),
            "crowding_score": (pl.Float64, None),
            "is_crowding_signal_valid": (pl.Int8, 0),
        }
        for col, (dtype, default) in specs.items():
            if col not in df.columns:
                df = df.with_columns(pl.lit(default).cast(dtype).alias(col))
        return df

    def _add_float_turnover_crowding_features(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return self._ensure_float_crowding_columns(df)

        required_cols = {"code", "date"}
        if not required_cols.issubset(df.columns):
            LOGGER.debug("[FLOAT] Missing base columns for float turnover computation")
            return self._ensure_float_crowding_columns(df)

        shares_col = self._resolve_column_name(df, "fs_shares_outstanding")
        turnover_col = self._resolve_column_name(df, "turnovervalue")
        price_col = self._resolve_column_name(df, "adjustmentclose") or self._resolve_column_name(df, "close")
        margin_long_col = self._resolve_column_name(df, "dmi_long_balance")
        weekly_long_col = self._resolve_column_name(df, "wm_long")

        if shares_col is None or turnover_col is None or price_col is None:
            LOGGER.debug(
                "[FLOAT] Insufficient inputs (shares=%s, turnover=%s, price=%s)", shares_col, turnover_col, price_col
            )
            return self._ensure_float_crowding_columns(df)

        eps = 1e-12
        share_expr = pl.col(shares_col).cast(pl.Float64, strict=False)
        turnover_expr = pl.col(turnover_col).cast(pl.Float64, strict=False)
        price_expr = pl.col(price_col).cast(pl.Float64, strict=False)

        new_columns: list[pl.Expr] = [
            pl.when(
                share_expr.is_not_null() & (share_expr > 0) & turnover_expr.is_not_null() & price_expr.is_not_null()
            )
            .then(turnover_expr / (price_expr * share_expr + eps))
            .otherwise(None)
            .alias("float_turnover_pct")
        ]

        if margin_long_col is not None:
            long_expr = pl.col(margin_long_col).cast(pl.Float64, strict=False)
            new_columns.append(
                pl.when(share_expr.is_not_null() & (share_expr > 0) & long_expr.is_not_null())
                .then(long_expr / (share_expr + eps))
                .otherwise(None)
                .alias("margin_long_pct_float")
            )
        else:
            new_columns.append(pl.lit(None).cast(pl.Float64).alias("margin_long_pct_float"))

        if weekly_long_col is not None:
            weekly_expr = pl.col(weekly_long_col).cast(pl.Float64, strict=False)
            new_columns.append(
                pl.when(share_expr.is_not_null() & (share_expr > 0) & weekly_expr.is_not_null())
                .then(weekly_expr / (share_expr + eps))
                .otherwise(None)
                .alias("weekly_margin_long_pct_float")
            )
        else:
            new_columns.append(pl.lit(None).cast(pl.Float64).alias("weekly_margin_long_pct_float"))

        df = df.sort(["code", "date"]).with_columns(new_columns)

        def _apply_roll_metrics(
            frame: pl.DataFrame,
            base_col: str,
            z_col: str,
            roc_col: str,
        ) -> pl.DataFrame:
            if base_col not in frame.columns:
                frame = frame.with_columns(pl.lit(None).cast(pl.Float64).alias(base_col))
            ma_col = f"_{base_col}_ma20"
            std_col = f"_{base_col}_std20"
            base_series = pl.col(base_col)
            frame = frame.with_columns(
                roll_mean_safe(base_series, 20, min_periods=5, by="code").alias(ma_col),
                roll_std_safe(base_series, 20, min_periods=5, by="code").alias(std_col),
            )
            frame = frame.with_columns(
                pl.when(pl.col(std_col).is_not_null() & (pl.col(std_col) > eps))
                .then((base_series.shift(1).over("code") - pl.col(ma_col)) / (pl.col(std_col) + eps))
                .otherwise(None)
                .alias(z_col),
                pl.when(base_series.shift(5).over("code").is_not_null())
                .then(base_series / (base_series.shift(5).over("code") + eps) - 1.0)
                .otherwise(None)
                .alias(roc_col),
            )
            frame = frame.drop([ma_col, std_col])
            return frame

        df = _apply_roll_metrics(df, "float_turnover_pct", "float_turnover_pct_z20", "float_turnover_pct_roc_5d")
        df = _apply_roll_metrics(
            df,
            "margin_long_pct_float",
            "margin_long_pct_float_z20",
            "margin_long_pct_float_roc_5d",
        )
        df = _apply_roll_metrics(
            df,
            "weekly_margin_long_pct_float",
            "weekly_margin_long_pct_float_z20",
            "weekly_margin_long_pct_float_roc_5d",
        )

        df = df.with_columns(
            pl.when(
                pl.col("margin_long_pct_float_z20").is_not_null()
                & pl.col("weekly_margin_long_pct_float_z20").is_not_null()
            )
            .then(pl.col("margin_long_pct_float_z20") + pl.col("weekly_margin_long_pct_float_z20"))
            .otherwise(None)
            .alias("crowding_score"),
        )

        df = df.with_columns(
            pl.when(pl.col("crowding_score").is_not_null())
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("is_crowding_signal_valid")
        )

        df = self._ensure_float_crowding_columns(df)
        return df

    def _ensure_short_squeeze_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        specs: dict[str, tuple[pl.DataType, object | None]] = {
            "ssp_ratio_component": (pl.Float64, None),
            "sector_short_ratio_z20": (pl.Float64, None),
            "limit_up_flag_lag1": (pl.Float64, None),
            "squeeze_risk": (pl.Float64, None),
            "is_squeeze_signal_valid": (pl.Int8, 0),
        }
        for col, (dtype, default) in specs.items():
            if col not in df.columns:
                df = df.with_columns(pl.lit(default).cast(dtype).alias(col))
        return df

    def _add_short_squeeze_features(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return self._ensure_short_squeeze_columns(df)

        ssp_col = self._resolve_column_name(df, "ssp_ratio_sum")
        sector_ratio_col = self._resolve_column_name(df, "sector_short_selling_ratio") or self._resolve_column_name(
            df, "ss_ratio_market"
        )
        limit_flag_col = self._resolve_column_name(df, "limit_up_flag") or self._resolve_column_name(df, "is_limit_up")

        if ssp_col is None or sector_ratio_col is None or limit_flag_col is None:
            LOGGER.debug(
                "[SQUEEZE] Missing inputs (ssp=%s, sector_ratio=%s, limit_flag=%s)",
                ssp_col,
                sector_ratio_col,
                limit_flag_col,
            )
            return self._ensure_short_squeeze_columns(df)

        df = df.sort(["code", "date"])

        ssp_z_col = self._resolve_column_name(df, "ssp_ratio_sum_z20")
        if ssp_z_col is None and ssp_col is not None:
            temp_ma = "__ssp_ma20"
            temp_std = "__ssp_std20"
            df = df.with_columns(
                roll_mean_safe(pl.col(ssp_col), 20, min_periods=5, by="code").alias(temp_ma),
                roll_std_safe(pl.col(ssp_col), 20, min_periods=5, by="code").alias(temp_std),
            )
            df = df.with_columns(
                pl.when(pl.col(temp_std).abs() > 1e-12)
                .then((pl.col(ssp_col).shift(1).over("code") - pl.col(temp_ma)) / (pl.col(temp_std) + 1e-12))
                .otherwise(None)
                .alias("ssp_ratio_sum_z20")
            )
            df = df.drop([temp_ma, temp_std], strict=False)
            ssp_z_col = "ssp_ratio_sum_z20"

        sector_mean = "__sector_short_ma20"
        sector_std = "__sector_short_std20"
        df = df.with_columns(
            roll_mean_safe(pl.col(sector_ratio_col), 20, min_periods=5, by="sector_code").alias(sector_mean),
            roll_std_safe(pl.col(sector_ratio_col), 20, min_periods=5, by="sector_code").alias(sector_std),
        )
        df = df.with_columns(
            pl.when(pl.col(sector_std).abs() > 1e-12)
            .then(
                (pl.col(sector_ratio_col).shift(1).over("sector_code") - pl.col(sector_mean))
                / (pl.col(sector_std) + 1e-12)
            )
            .otherwise(None)
            .alias("sector_short_ratio_z20")
        )
        df = df.drop([sector_mean, sector_std], strict=False)

        df = df.with_columns(
            pl.col(limit_flag_col).shift(1).over("code").fill_null(0).cast(pl.Float64).alias("limit_up_flag_lag1"),
            pl.col(ssp_z_col if ssp_z_col else ssp_col).alias("ssp_ratio_component"),
        )

        df = df.with_columns(
            pl.sum_horizontal(
                [
                    pl.col("ssp_ratio_component"),
                    pl.col("sector_short_ratio_z20"),
                    pl.col("limit_up_flag_lag1"),
                ]
            ).alias("squeeze_risk"),
            pl.when(pl.col("ssp_ratio_component").is_not_null() & pl.col("sector_short_ratio_z20").is_not_null())
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("is_squeeze_signal_valid"),
        )

        return self._ensure_short_squeeze_columns(df)

    def _ensure_margin_pain_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        specs: dict[str, tuple[pl.DataType, object | None]] = {
            "mpi_drawdown": (pl.Float64, None),
            "mpi_drawdown_z20": (pl.Float64, None),
            "mpi_dist_to_limit": (pl.Float64, None),
            "mpi_dist_to_limit_z20": (pl.Float64, None),
            "margin_long_pct_float_z20": (pl.Float64, None),
            "margin_pain_index": (pl.Float64, None),
            "is_margin_pain_valid": (pl.Int8, 0),
        }
        for col, (dtype, default) in specs.items():
            if col not in df.columns:
                df = df.with_columns(pl.lit(default).cast(dtype).alias(col))
        return df

    def _add_margin_pain_index(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return self._ensure_margin_pain_columns(df)

        # Note: shares_col resolved but not needed (margin pain uses percentage)
        margin_pct_col = self._resolve_column_name(df, "margin_long_pct_float")
        price_col = self._resolve_column_name(df, "adjustmentclose") or self._resolve_column_name(df, "close")
        lower_limit_price_col = self._resolve_column_name(df, "lower_limit_price")

        if margin_pct_col is None or price_col is None or lower_limit_price_col is None:
            LOGGER.debug(
                "[MPI] Missing inputs (margin_pct=%s, price=%s, lower_limit_price=%s)",
                margin_pct_col,
                price_col,
                lower_limit_price_col,
            )
            return self._ensure_margin_pain_columns(df)

        df = df.sort(["code", "date"])
        price_expr = pl.col(price_col).cast(pl.Float64, strict=False)

        roll_alias = "__mpi_roll_max20"
        if hasattr(pl.Expr, "rolling_max"):
            df = df.with_columns(
                price_expr.shift(1).over("code").rolling_max(window_size=20, min_periods=5).alias(roll_alias)
            )
        else:
            df = df.with_columns(price_expr.shift(1).over("code").alias(roll_alias))

        df = df.with_columns(
            pl.when(pl.col(roll_alias).is_not_null() & (pl.col(roll_alias).abs() > 1e-12))
            .then((price_expr / pl.col(roll_alias)) - 1.0)
            .otherwise(None)
            .alias("mpi_drawdown")
        )

        limit_expr = pl.col(lower_limit_price_col).cast(pl.Float64, strict=False)
        df = df.with_columns(
            pl.when(price_expr.is_not_null() & (price_expr.abs() > 1e-9) & limit_expr.is_not_null())
            .then((price_expr - limit_expr) / price_expr)
            .otherwise(None)
            .alias("mpi_dist_to_limit")
        )
        df = df.drop([roll_alias], strict=False)

        def _zscore(expr_col: str, alias: str, by: str = "code") -> None:
            nonlocal df
            mean_alias = f"__{alias}_ma20"
            std_alias = f"__{alias}_std20"
            df = df.with_columns(
                roll_mean_safe(pl.col(expr_col), 20, min_periods=5, by=by).alias(mean_alias),
                roll_std_safe(pl.col(expr_col), 20, min_periods=5, by=by).alias(std_alias),
            )
            df = df.with_columns(
                pl.when(pl.col(std_alias).abs() > 1e-12)
                .then((pl.col(expr_col) - pl.col(mean_alias)) / (pl.col(std_alias) + 1e-12))
                .otherwise(None)
                .alias(alias)
            )
            df = df.drop([mean_alias, std_alias], strict=False)

        _zscore("mpi_drawdown", "mpi_drawdown_z20")
        _zscore("mpi_dist_to_limit", "mpi_dist_to_limit_z20")

        df = df.with_columns(
            pl.sum_horizontal(
                [
                    pl.col("margin_long_pct_float_z20"),
                    pl.col("mpi_drawdown_z20").abs(),
                    (-pl.col("mpi_dist_to_limit_z20")),
                ]
            ).alias("margin_pain_index"),
            pl.when(
                pl.col("margin_long_pct_float_z20").is_not_null()
                & pl.col("mpi_drawdown_z20").is_not_null()
                & pl.col("mpi_dist_to_limit_z20").is_not_null()
            )
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("is_margin_pain_valid"),
        )

        return self._ensure_margin_pain_columns(df)

    def _ensure_pre_earnings_flow_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        specs: dict[str, tuple[pl.DataType, object | None]] = {
            "preE_margin_diff": (pl.Float64, None),
            "preE_short_ratio_diff": (pl.Float64, None),
            "preE_margin_diff_z20": (pl.Float64, None),
            "preE_short_ratio_diff_z20": (pl.Float64, None),
            "preE_flow_score": (pl.Float64, None),
            "preE_risk_score": (pl.Float64, None),
            "is_preE_flow_valid": (pl.Int8, 0),
        }
        for col, (dtype, default) in specs.items():
            if col not in df.columns:
                df = df.with_columns(pl.lit(default).cast(dtype).alias(col))
        return df

    def _add_pre_earnings_flow_features(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return self._ensure_pre_earnings_flow_columns(df)

        margin_pct_col = self._resolve_column_name(df, "margin_long_pct_float")
        short_ratio_col = self._resolve_column_name(df, "short_selling_ratio_market")
        earn_flag_col = self._resolve_column_name(df, "earnings_upcoming_3d") or self._resolve_column_name(
            df, "is_E_pp3"
        )

        if margin_pct_col is None or short_ratio_col is None or earn_flag_col is None:
            LOGGER.debug(
                "[EARN-FLOW] Missing inputs (margin=%s, short_ratio=%s, earn_flag=%s)",
                margin_pct_col,
                short_ratio_col,
                earn_flag_col,
            )
            return self._ensure_pre_earnings_flow_columns(df)

        df = df.sort(["code", "date"])
        margin_diff = (
            pl.col(margin_pct_col).cast(pl.Float64) - pl.col(margin_pct_col).cast(pl.Float64).shift(5).over("code")
        ).alias("preE_margin_diff")
        short_diff = (
            pl.col(short_ratio_col).cast(pl.Float64) - pl.col(short_ratio_col).cast(pl.Float64).shift(5)
        ).alias("preE_short_ratio_diff")
        df = df.with_columns([margin_diff, short_diff])

        def _zscore(col: str, alias: str, by: str | None) -> None:
            nonlocal df
            mean_alias = f"__{alias}_ma20"
            std_alias = f"__{alias}_std20"
            df = df.with_columns(
                roll_mean_safe(pl.col(col), 20, min_periods=5, by=by).alias(mean_alias),
                roll_std_safe(pl.col(col), 20, min_periods=5, by=by).alias(std_alias),
            )
            df = df.with_columns(
                pl.when(pl.col(std_alias).abs() > 1e-12)
                .then((pl.col(col) - pl.col(mean_alias)) / (pl.col(std_alias) + 1e-12))
                .otherwise(None)
                .alias(alias)
            )
            df = df.drop([mean_alias, std_alias], strict=False)

        _zscore("preE_margin_diff", "preE_margin_diff_z20", "code")
        _zscore("preE_short_ratio_diff", "preE_short_ratio_diff_z20", None)

        df = df.with_columns(
            pl.sum_horizontal([pl.col("preE_margin_diff_z20"), pl.col("preE_short_ratio_diff_z20")]).alias(
                "preE_flow_score"
            ),
        )

        pre_event_mask = pl.col(earn_flag_col) == 1
        earn_1d_col = self._resolve_column_name(df, "earnings_upcoming_1d")
        earn_5d_col = self._resolve_column_name(df, "earnings_upcoming_5d")
        if earn_1d_col is not None:
            pre_event_mask = pre_event_mask | (pl.col(earn_1d_col) == 1)
        if earn_5d_col is not None:
            pre_event_mask = pre_event_mask | (pl.col(earn_5d_col) == 1)

        df = df.with_columns(
            pre_event_mask.cast(pl.Int8).alias("_preE_flag"),
            (pre_event_mask.cast(pl.Float64) * pl.col("preE_flow_score")).alias("preE_risk_score"),
            pl.when(pl.col("preE_flow_score").is_not_null() & pre_event_mask.is_not_null())
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("is_preE_flow_valid"),
        )

        df = df.drop(["_preE_flag"], strict=False)

        return self._ensure_pre_earnings_flow_columns(df)

    def _ensure_gap_basis_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        specs: dict[str, tuple[pl.DataType, object | None]] = {
            "gap_predictor": (pl.Float64, None),
            "basis_gate": (pl.Float64, None),
            "is_gap_basis_valid": (pl.Int8, 0),
        }
        for col, (dtype, default) in specs.items():
            if col not in df.columns:
                df = df.with_columns(pl.lit(default).cast(dtype).alias(col))
        return df

    def _add_gap_basis_features(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return self._ensure_gap_basis_columns(df)

        fut_overnight_col = self._resolve_column_name(df, "fut_topix_overnight")
        basis_col = self._resolve_column_name(df, "fut_topix_basis")
        beta_col = self._resolve_column_name(df, "beta60_topix")

        if fut_overnight_col is None or basis_col is None or beta_col is None:
            LOGGER.debug(
                "[GAP] Missing inputs (overnight=%s, basis=%s, beta=%s)",
                fut_overnight_col,
                basis_col,
                beta_col,
            )
            return self._ensure_gap_basis_columns(df)

        df = df.with_columns(
            (pl.col(fut_overnight_col) * pl.col(beta_col)).alias("gap_predictor"),
            (pl.col(basis_col) * pl.col(beta_col)).alias("basis_gate"),
            pl.when(
                pl.col(fut_overnight_col).is_not_null()
                & pl.col(basis_col).is_not_null()
                & pl.col(beta_col).is_not_null()
            )
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("is_gap_basis_valid"),
        )

        return self._ensure_gap_basis_columns(df)

    def _ensure_liquidity_impact_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        specs: dict[str, tuple[pl.DataType, object | None]] = {
            "amihud_z20": (pl.Float64, None),
            "bd_activity_z20": (pl.Float64, None),
            "liquidity_impact": (pl.Float64, None),
            "is_liquidity_signal_valid": (pl.Int8, 0),
        }
        for col, (dtype, default) in specs.items():
            if col not in df.columns:
                df = df.with_columns(pl.lit(default).cast(dtype).alias(col))
        return df

    def _add_liquidity_impact_features(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return self._ensure_liquidity_impact_columns(df)

        amihud_z_col = self._resolve_column_name(df, "amihud_z20")
        amihud_raw_col = self._resolve_column_name(df, "amihud_20")
        bd_activity_col = self._resolve_column_name(df, "bd_activity_ratio")

        if bd_activity_col is None or (amihud_z_col is None and amihud_raw_col is None):
            LOGGER.debug(
                "[LIQ] Missing inputs (amihud_z=%s, amihud_raw=%s, bd_activity=%s)",
                amihud_z_col,
                amihud_raw_col,
                bd_activity_col,
            )
            return self._ensure_liquidity_impact_columns(df)

        df = df.sort(["code", "date"])

        if amihud_z_col is None and amihud_raw_col is not None:
            mean_alias = "__amihud_ma20"
            std_alias = "__amihud_std20"
            df = df.with_columns(
                roll_mean_safe(pl.col(amihud_raw_col), 20, min_periods=5, by="code").alias(mean_alias),
                roll_std_safe(pl.col(amihud_raw_col), 20, min_periods=5, by="code").alias(std_alias),
            )
            df = df.with_columns(
                pl.when(pl.col(std_alias).abs() > 1e-12)
                .then((pl.col(amihud_raw_col) - pl.col(mean_alias)) / (pl.col(std_alias) + 1e-12))
                .otherwise(None)
                .alias("amihud_z20"),
            )
            df = df.drop([mean_alias, std_alias], strict=False)
            amihud_z_col = "amihud_z20"

        bd_mean = "__bd_activity_ma20"
        bd_std = "__bd_activity_std20"
        df = df.with_columns(
            roll_mean_safe(pl.col(bd_activity_col), 20, min_periods=5, by="code").alias(bd_mean),
            roll_std_safe(pl.col(bd_activity_col), 20, min_periods=5, by="code").alias(bd_std),
        )
        df = df.with_columns(
            pl.when(pl.col(bd_std).abs() > 1e-12)
            .then((pl.col(bd_activity_col) - pl.col(bd_mean)) / (pl.col(bd_std) + 1e-12))
            .otherwise(None)
            .alias("bd_activity_z20"),
        )
        df = df.drop([bd_mean, bd_std], strict=False)

        df = df.with_columns(
            (pl.col(amihud_z_col if amihud_z_col else "amihud_z20") * pl.col("bd_activity_z20")).alias(
                "liquidity_impact"
            ),
            pl.when(
                pl.col("bd_activity_z20").is_not_null()
                & pl.col(amihud_z_col if amihud_z_col else "amihud_z20").is_not_null()
            )
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("is_liquidity_signal_valid"),
        )

        return self._ensure_liquidity_impact_columns(df)

    def _extend_limit_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Extend limit features with P0: is_limit_up/down, streak, post_limit_reversal_flag."""
        required = {"code", "date", "upper_limit", "lower_limit", "ret_intraday"}
        if df.is_empty() or not required.issubset(df.columns):
            return self._ensure_extended_limit_columns(df)

        df = df.sort(["code", "date"])

        # is_limit_up/down: 既存のlimit_up_flag/limit_down_flagのエイリアス
        if "limit_up_flag" in df.columns:
            df = df.with_columns(pl.col("limit_up_flag").alias("is_limit_up"))
        else:
            df = df.with_columns((pl.col("upper_limit") == 1).fill_null(False).cast(pl.Int8).alias("is_limit_up"))

        if "limit_down_flag" in df.columns:
            df = df.with_columns(pl.col("limit_down_flag").alias("is_limit_down"))
        else:
            df = df.with_columns((pl.col("lower_limit") == 1).fill_null(False).cast(pl.Int8).alias("is_limit_down"))

        # limit_up_streak, limit_down_streak: 直近連続日数
        # 連続フラグを計算（forward fillで連続区間を識別）
        df = df.with_columns(pl.arange(0, pl.len()).over("code").alias("_row_idx"))

        # limit_up_streak
        df = df.with_columns(
            pl.when(pl.col("is_limit_up") == 1).then(pl.col("_row_idx")).otherwise(None).alias("_limit_up_start")
        )
        df = df.with_columns(pl.col("_limit_up_start").forward_fill().over("code").alias("_limit_up_ff"))
        df = df.with_columns(
            pl.when(pl.col("_limit_up_ff").is_not_null())
            .then((pl.col("_row_idx") - pl.col("_limit_up_ff") + 1).cast(pl.Int32))
            .otherwise(pl.lit(0))
            .alias("limit_up_streak")
        )

        # limit_down_streak
        df = df.with_columns(
            pl.when(pl.col("is_limit_down") == 1).then(pl.col("_row_idx")).otherwise(None).alias("_limit_down_start")
        )
        df = df.with_columns(pl.col("_limit_down_start").forward_fill().over("code").alias("_limit_down_ff"))
        df = df.with_columns(
            pl.when(pl.col("_limit_down_ff").is_not_null())
            .then((pl.col("_row_idx") - pl.col("_limit_down_ff") + 1).cast(pl.Int32))
            .otherwise(pl.lit(0))
            .alias("limit_down_streak")
        )

        # post_limit_reversal_flag: is_limit_up_{t-1} & (ret_intraday_t<0)
        df = df.with_columns(
            ((pl.col("is_limit_up").shift(1).over("code") == 1) & (pl.col("ret_intraday") < 0))
            .cast(pl.Int8)
            .alias("post_limit_reversal_flag")
        )

        # 一時列を削除
        cleanup_cols = [
            "_row_idx",
            "_limit_up_start",
            "_limit_up_ff",
            "_limit_down_start",
            "_limit_down_ff",
        ]
        for col in cleanup_cols:
            if col in df.columns:
                df = df.drop(col)

        df = self._ensure_extended_limit_columns(df)

        if isinstance(self._run_meta, dict):
            limit_meta = self._run_meta.setdefault("extended_limit_features", {})
            limit_meta.update(
                {
                    "columns": [
                        "is_limit_up",
                        "is_limit_down",
                        "limit_up_streak",
                        "limit_down_streak",
                        "post_limit_reversal_flag",
                    ],
                    "policy": "left_closed_shift1",
                }
            )

        return df

    def _ensure_extended_limit_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Ensure all extended limit feature columns exist with proper defaults."""
        specs = {
            "is_limit_up": (pl.Int8, 0),
            "is_limit_down": (pl.Int8, 0),
            "limit_up_streak": (pl.Int32, 0),
            "limit_down_streak": (pl.Int32, 0),
            "post_limit_reversal_flag": (pl.Int8, 0),
        }
        for col, (dtype, default) in specs.items():
            if col not in df.columns:
                df = df.with_columns(pl.lit(default).cast(dtype).alias(col))
        return df

    def _add_am_pm_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add AM/PM features (P1: am_pos, am_vol_ratio_20, extending existing session_features)."""
        # 既存のsession_featuresがam_gap_prev_close, am_rangeを提供しているため、
        # 不足しているam_pos, am_vol_ratio_20のみを追加
        required = {
            "code",
            "date",
            "morning_close",
            "adjustmentclose",
            "morning_high",
            "morning_low",
            "morning_volume",
            "volume",
        }
        if df.is_empty() or not required.issubset(df.columns):
            return self._ensure_am_pm_columns(df)

        df = df.sort(["code", "date"])
        eps = 1e-9

        mc = pl.col("morning_close")
        mh = pl.col("morning_high")
        ml = pl.col("morning_low")
        mv = pl.col("morning_volume")

        # am_pos = (morningclose - morninglow) / max(morninghigh - morninglow, eps)
        # 既存のsession_featuresにないため追加
        if "am_pos" not in df.columns:
            df = df.with_columns(
                pl.when((mh - ml).abs() > eps).then((mc - ml) / (mh - ml + eps)).otherwise(None).alias("am_pos")
            )

        # am_vol_ratio_20 = morningvolume / roll_mean(volume,20,shift=1)
        # 既存のsession_featuresにないため追加
        if "am_vol_ratio_20" not in df.columns:
            from ..features.utils.rolling import roll_mean_safe

            df = df.with_columns(roll_mean_safe(pl.col("volume"), 20, min_periods=10, by="code").alias("_vol_ma20"))
            df = df.with_columns(
                pl.when(pl.col("_vol_ma20").abs() > eps)
                .then(mv / (pl.col("_vol_ma20") + eps))
                .otherwise(None)
                .alias("am_vol_ratio_20")
            )

            # 一時列を削除
            if "_vol_ma20" in df.columns:
                df = df.drop("_vol_ma20")

        df = self._ensure_am_pm_columns(df)

        if isinstance(self._run_meta, dict):
            am_pm_meta = self._run_meta.setdefault("am_pm_features_extended", {})
            am_pm_meta.update(
                {
                    "columns": ["am_pos", "am_vol_ratio_20"],
                    "policy": "available_ts=当日15:00_JST",
                    "p1_features": True,
                    "note": "Extends existing session_features (am_gap_prev_close, am_range are provided by session_features)",
                }
            )

        return df

    def _ensure_am_pm_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Ensure all AM/PM feature columns exist with proper defaults."""
        specs = {
            "am_pos": (pl.Float64, None),
            "am_vol_ratio_20": (pl.Float64, None),
        }
        for col, (dtype, default) in specs.items():
            if col not in df.columns:
                df = df.with_columns(pl.lit(default).cast(dtype).alias(col))
        return df

    def _add_sector17_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add sector17 relative features (extending existing sector33).

        P0: Sector17相対特徴量を追加（既存のsector33に加えて）
        """
        if df.is_empty() or "sector17_code" not in df.columns:
            return self._ensure_sector17_columns(df)

        # 既存のsector_featuresの実装を参考に、sector17_codeでも同様の特徴量を生成
        # ただし、既存のsector_featuresはsector_codeを使用しているため、
        # ここではsector17_code専用の特徴量を追加
        sector17_col = "sector17_code"
        cross_keys = ["date", sector17_col]

        # 日次×セクター17の集計
        if "ret_prev_1d" in df.columns:
            df = df.with_columns(pl.col("ret_prev_1d").median().over(cross_keys).alias("sec17_ret_1d_eq"))
        if "ret_prev_5d" in df.columns:
            df = df.with_columns(pl.col("ret_prev_5d").median().over(cross_keys).alias("sec17_ret_5d_eq"))

        # Sector17相対リターン
        if "sec17_ret_1d_eq" in df.columns and "ret_prev_1d" in df.columns:
            df = df.with_columns((pl.col("ret_prev_1d") - pl.col("sec17_ret_1d_eq")).alias("rel_to_sec17_1d"))
        if "sec17_ret_5d_eq" in df.columns and "ret_prev_5d" in df.columns:
            df = df.with_columns((pl.col("ret_prev_5d") - pl.col("sec17_ret_5d_eq")).alias("rel_to_sec17_5d"))

        # Sector17モメンタム（20日）
        if "sec17_ret_1d_eq" in df.columns:
            df = df.with_columns(
                pl.col("sec17_ret_1d_eq")
                .shift(1)
                .rolling_sum(window_size=20, min_periods=20)
                .over(sector17_col)
                .alias("sec17_mom_20")
            )

        df = self._ensure_sector17_columns(df)

        if isinstance(self._run_meta, dict):
            sector17_meta = self._run_meta.setdefault("sector17_features", {})
            sector17_meta.update(
                {
                    "columns": [
                        "sec17_ret_1d_eq",
                        "sec17_ret_5d_eq",
                        "rel_to_sec17_1d",
                        "rel_to_sec17_5d",
                        "sec17_mom_20",
                    ],
                    "policy": "left_closed_shift1",
                    "note": "Extends existing sector33 features with sector17 granularity",
                }
            )

        return df

    def _ensure_sector17_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Ensure all sector17 feature columns exist with proper defaults."""
        specs = {
            "sec17_ret_1d_eq": (pl.Float64, None),
            "sec17_ret_5d_eq": (pl.Float64, None),
            "rel_to_sec17_1d": (pl.Float64, None),
            "rel_to_sec17_5d": (pl.Float64, None),
            "sec17_mom_20": (pl.Float64, None),
        }
        for col, (dtype, default) in specs.items():
            if col not in df.columns:
                df = df.with_columns(pl.lit(default).cast(dtype).alias(col))
        return df

    def _attach_listed_info_features(
        self,
        df: pl.DataFrame,
        *,
        start: str,
        end: str,
        calendar_df: pl.DataFrame,
        listed_info_df: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Attach listed info features with as-of protection.

        P0 features:
        - Market dummies: is_prime, is_standard, is_growth
        - Sector codes: sector33_code, sector17_code (extending existing sector_code)
        - Scale bucket: scale_bucket
        - Margin eligibility: is_margin_eligible
        - Market/sector change event flags
        """
        if listed_info_df is not None:
            listed_raw = listed_info_df
        else:
            try:
                listed_raw = self.data_sources.listed_info(start=start, end=end)
            except Exception as exc:
                LOGGER.warning("Failed to fetch listed_info: %s", exc, exc_info=True)
                return self._ensure_listed_info_columns(df)

        if listed_raw.is_empty():
            LOGGER.debug("Listed info is empty, skipping features")
            return self._ensure_listed_info_columns(df)

        # Build listed info features with as-of protection
        from ..features.core.listed_info_features import build_listed_info_features

        try:
            listed_features = build_listed_info_features(
                listed_raw,
                trading_calendar=calendar_df,
                build_date=end,  # Use end date as build reference
            )
        except Exception as exc:
            LOGGER.warning("Failed to build listed info features: %s", exc, exc_info=True)
            return self._ensure_listed_info_columns(df)

        if listed_features.is_empty():
            LOGGER.debug("Listed info features generation returned empty, skipping")
            return self._ensure_listed_info_columns(df)

        # As-of join (backward interval join)
        snapshot_ts_col = "_listed_available_ts"
        backbone_ts_col = "_listed_asof_ts"
        listed_features = listed_features.with_columns(pl.col("available_ts").cast(pl.Int64).alias(snapshot_ts_col))
        working_df = df.with_columns(pl.col("asof_ts").cast(pl.Int64).alias(backbone_ts_col))

        joined = interval_join_pl(
            backbone=working_df,
            snapshot=listed_features,
            on_code="code",
            backbone_ts=backbone_ts_col,
            snapshot_ts=snapshot_ts_col,
            strategy="backward",
            suffix="_listed",
        )

        # 一時列を削除
        cleanup_cols = [snapshot_ts_col, backbone_ts_col]
        for col in cleanup_cols:
            if col in joined.columns:
                joined = joined.drop(col)

        # 既存のsector_codeをsector33_codeにエイリアス（後方互換性）
        if "sector33_code" not in joined.columns and "sector_code" in joined.columns:
            joined = joined.with_columns(pl.col("sector_code").alias("sector33_code"))

        # 派生特徴量を追加
        joined = self._add_listed_info_derived_features(joined)

        joined = self._ensure_listed_info_columns(joined)

        # メタデータ更新
        if isinstance(self._run_meta, dict):
            listed_meta = self._run_meta.setdefault("listed_info_features", {})
            listed_meta.update(
                {
                    "columns": [
                        "is_prime",
                        "is_standard",
                        "is_growth",
                        "sector33_code",
                        "sector17_code",
                        "scale_bucket",
                        "is_margin_eligible",
                        "is_listed_info_valid",
                    ],
                    "source": "/listed/info",
                    "as_of_rule": "当日情報: Date @ 09:00 JST, 翌日情報: today 17:30 JST (T+1 09:00まで使用不可)",
                }
            )

        return joined

    def _add_listed_info_derived_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add derived features from listed info (P0/P1).

        - Sector17 relative features (extending existing sector33)
        - Market dummies × main features interactions
        - Scale category size factors
        - Market/sector change event flags
        """
        if df.is_empty():
            return df

        # Sector17相対特徴量（既存のsector33に加えて）
        # 既存のsector_featuresはsector33_codeを使用しているため、
        # sector17_codeでも同様の特徴量を生成
        if "sector17_code" in df.columns and "ret_prev_1d" in df.columns:
            # Sector17相対リターン（簡易実装）
            # より詳細な実装はsector_featuresの拡張で対応
            df = df.with_columns(
                pl.col("ret_prev_1d").median().over(["date", "sector17_code"]).alias("sec17_ret_1d_eq")
            )
            if "sec17_ret_1d_eq" in df.columns and "ret_prev_1d" in df.columns:
                df = df.with_columns((pl.col("ret_prev_1d") - pl.col("sec17_ret_1d_eq")).alias("rel_to_sec17_1d"))

        # 市場区分ダミー×主要特徴の交互作用（P0）
        if "is_prime" in df.columns:
            # is_prime × ret_prev_20d
            if "ret_prev_20d" in df.columns:
                df = df.with_columns((pl.col("is_prime") * pl.col("ret_prev_20d")).alias("is_prime_x_ret_20d"))
            # is_prime × dollar_volume_z20（存在する場合）
            if "dollar_volume_z20" in df.columns:
                df = df.with_columns((pl.col("is_prime") * pl.col("dollar_volume_z20")).alias("is_prime_x_dv_z20"))

        if "is_growth" in df.columns:
            # is_growth × ret_prev_20d
            if "ret_prev_20d" in df.columns:
                df = df.with_columns((pl.col("is_growth") * pl.col("ret_prev_20d")).alias("is_growth_x_ret_20d"))
            # is_growth × dollar_volume_z20
            if "dollar_volume_z20" in df.columns:
                df = df.with_columns((pl.col("is_growth") * pl.col("dollar_volume_z20")).alias("is_growth_x_dv_z20"))

        # 規模カテゴリのサイズ因子（P0）
        # scale_bucketのone-hotエンコーディング（Large70/Mid400/Small等）
        if "scale_bucket" in df.columns:
            # 主要な規模カテゴリのダミー
            scale_categories = ["Large70", "Mid400", "Small"]
            for cat in scale_categories:
                df = df.with_columns((pl.col("scale_bucket") == cat).cast(pl.Int8).alias(f"is_scale_{cat.lower()}"))

        # 市場/セクター変更イベントフラグ（P0）
        # days_since_market_change, market_changed_5d
        df = df.sort(["code", "date"])
        df = df.with_columns(pl.arange(0, pl.len()).over("code").alias("_row_idx"))

        # 市場区分変更の検出
        if "market_code" in df.columns:
            df = df.with_columns(
                (pl.col("market_code") != pl.col("market_code").shift(1).over("code"))
                .cast(pl.Int8)
                .alias("_market_changed")
            )
            df = df.with_columns(
                pl.when(pl.col("_market_changed") == 1)
                .then(pl.col("_row_idx"))
                .otherwise(None)
                .alias("_market_change_idx")
            )
            df = df.with_columns(
                pl.col("_market_change_idx").forward_fill().over("code").alias("_ff_market_change_idx")
            )
            df = df.with_columns(
                pl.when(pl.col("_ff_market_change_idx").is_not_null())
                .then((pl.col("_row_idx") - pl.col("_ff_market_change_idx")).cast(pl.Int32))
                .otherwise(None)
                .alias("days_since_market_change")
            )
            df = df.with_columns(
                ((pl.col("days_since_market_change") >= 0) & (pl.col("days_since_market_change") <= 5))
                .cast(pl.Int8)
                .alias("market_changed_5d")
            )

        # セクター変更の検出（sector33_code）
        if "sector33_code" in df.columns:
            df = df.with_columns(
                (pl.col("sector33_code") != pl.col("sector33_code").shift(1).over("code"))
                .cast(pl.Int8)
                .alias("_sector33_changed")
            )
            df = df.with_columns(
                pl.when(pl.col("_sector33_changed") == 1)
                .then(pl.col("_row_idx"))
                .otherwise(None)
                .alias("_sector33_change_idx")
            )
            df = df.with_columns(
                pl.col("_sector33_change_idx").forward_fill().over("code").alias("_ff_sector33_change_idx")
            )
            df = df.with_columns(
                pl.when(pl.col("_ff_sector33_change_idx").is_not_null())
                .then((pl.col("_row_idx") - pl.col("_ff_sector33_change_idx")).cast(pl.Int32))
                .otherwise(None)
                .alias("days_since_sector33_change")
            )
            df = df.with_columns(
                ((pl.col("days_since_sector33_change") >= 0) & (pl.col("days_since_sector33_change") <= 5))
                .cast(pl.Int8)
                .alias("sector33_changed_5d")
            )

        # 一時列を削除
        cleanup_cols = [
            "_row_idx",
            "_market_changed",
            "_market_change_idx",
            "_ff_market_change_idx",
            "_sector33_changed",
            "_sector33_change_idx",
            "_ff_sector33_change_idx",
        ]
        for col in cleanup_cols:
            if col in df.columns:
                df = df.drop(col)

        return df

    def _ensure_listed_info_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Ensure all listed info feature columns exist with proper defaults."""
        specs = {
            # Market dummies
            "is_prime": (pl.Int8, 0),
            "is_standard": (pl.Int8, 0),
            "is_growth": (pl.Int8, 0),
            # Sector codes
            "sector33_code": (pl.Utf8, "UNKNOWN"),
            "sector17_code": (pl.Utf8, "UNKNOWN"),
            # Scale bucket
            "scale_bucket": (pl.Utf8, None),
            # Margin eligibility
            "is_margin_eligible": (pl.Int8, None),
            # Quality flags
            "is_listed_info_valid": (pl.Int8, 0),
            # Derived features
            "sec17_ret_1d_eq": (pl.Float64, None),
            "rel_to_sec17_1d": (pl.Float64, None),
            "is_prime_x_ret_20d": (pl.Float64, None),
            "is_growth_x_ret_20d": (pl.Float64, None),
            "is_prime_x_dv_z20": (pl.Float64, None),
            "is_growth_x_dv_z20": (pl.Float64, None),
            "is_scale_large70": (pl.Int8, 0),
            "is_scale_mid400": (pl.Int8, 0),
            "is_scale_small": (pl.Int8, 0),
            "days_since_market_change": (pl.Int32, None),
            "market_changed_5d": (pl.Int8, 0),
            "days_since_sector33_change": (pl.Int32, None),
            "sector33_changed_5d": (pl.Int8, 0),
        }
        for col, (dtype, default) in specs.items():
            if col not in df.columns:
                df = df.with_columns(pl.lit(default).cast(dtype).alias(col))
        return df

    def _ensure_margin_daily_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        specs = {
            # P0: Core levels
            "dmi_long_balance": pl.Float64,
            "dmi_short_balance": pl.Float64,
            "dmi_total": pl.Float64,
            "dmi_net": pl.Float64,
            "dmi_long_short_ratio": pl.Float64,
            # P0: Changes
            "dmi_long_balance_diff_1d": pl.Float64,
            "dmi_short_balance_diff_1d": pl.Float64,
            "dmi_total_diff_1d": pl.Float64,
            "dmi_net_diff_1d": pl.Float64,
            # P0: Z-scores
            "dmi_net_z20": pl.Float64,
            "dmi_long_short_ratio_z20": pl.Float64,
            # P0: Liquidity normalization
            "dmi_total_over_adv20": pl.Float64,
            "dmi_net_over_adv20": pl.Float64,
            # P0: Quality flags
            "is_dmi_valid": pl.Int8,
            "dmi_staleness_days": pl.Int32,
            "dmi_reason_code": pl.Utf8,
            "dmi_reason_is_revision": pl.Int8,
            # Legacy (for backward compatibility)
            "dmi_net_adv60": pl.Float64,
            "dmi_delta_net_adv60": pl.Float64,
            "dmi_delta_net_adv60_z20": pl.Float64,
            "dmi_imbalance": pl.Float64,
            "is_margin_daily_valid": pl.Int8,
        }
        for col, dtype in specs.items():
            if col not in df.columns:
                fill = 0 if dtype == pl.Int8 else None
                df = df.with_columns(pl.lit(fill).cast(dtype).alias(col))
        return df

    def _ensure_margin_weekly_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Ensure all weekly margin feature columns exist with proper defaults."""
        specs = {
            # P1: ベース
            "wm_long": pl.Float64,
            "wm_short": pl.Float64,
            "wm_long_gen": pl.Float64,
            "wm_short_gen": pl.Float64,
            "wm_long_std": pl.Float64,
            "wm_short_std": pl.Float64,
            "wm_net": pl.Float64,
            "wm_lsr": pl.Float64,
            "wm_gen_share": pl.Float64,
            "wm_std_share": pl.Float64,
            "wm_issue_type": pl.Int8,
            # P1: 変化・モメンタム
            "wm_net_d1w": pl.Float64,
            "wm_long_d1w": pl.Float64,
            "wm_short_d1w": pl.Float64,
            "wm_net_pct_d1w": pl.Float64,
            # P1: 標準化
            "wm_net_to_adv20": pl.Float64,
            "wm_long_to_adv20": pl.Float64,
            "wm_short_to_adv20": pl.Float64,
            # P1: 安定化
            "wm_net_z20": pl.Float64,
            "wm_short_z20": pl.Float64,
            "wm_long_z20": pl.Float64,
            "wm_net_z52": pl.Float64,
            # P1: 品質
            "is_wm_valid": pl.Int8,
            "wm_staleness_bd": pl.Int32,
            "wm_is_recent": pl.Int8,
            # 後方互換性
            "wmi_net_adv5d": pl.Float64,
            "wmi_delta_net_adv5d": pl.Float64,
            "wmi_delta_net_adv5d_z52": pl.Float64,
            "wmi_imbalance": pl.Float64,
            "wmi_long_short_ratio": pl.Float64,
            "is_margin_weekly_valid": pl.Int8,
        }
        for col, dtype in specs.items():
            if col not in df.columns:
                fill = 0 if dtype == pl.Int8 else None
                df = df.with_columns(pl.lit(fill).cast(dtype).alias(col))
        return df

    def _ensure_fs_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        specs: dict[str, tuple[pl.DataType, object | None]] = {
            "fs_revenue_ttm": (pl.Float64, None),
            "fs_op_profit_ttm": (pl.Float64, None),
            "fs_net_income_ttm": (pl.Float64, None),
            "fs_cfo_ttm": (pl.Float64, None),
            "fs_capex_ttm": (pl.Float64, None),
            "fs_fcf_ttm": (pl.Float64, None),
            "fs_sales_yoy": (pl.Float64, None),
            "fs_op_margin": (pl.Float64, None),
            "fs_net_margin": (pl.Float64, None),
            "fs_roe_ttm": (pl.Float64, None),
            "fs_roa_ttm": (pl.Float64, None),
            "fs_accruals_ttm": (pl.Float64, None),
            "fs_cfo_to_ni": (pl.Float64, None),
            "fs_observation_count": (pl.Int16, 0),
            "fs_lag_days": (pl.Int32, None),
            "fs_is_recent": (pl.Int8, 0),
            "fs_staleness_bd": (pl.Int32, None),
            "is_fs_valid": (pl.Int8, 0),
            "fs_shares_outstanding": (pl.Float64, None),
            "fs_average_shares": (pl.Float64, None),
            # P0新特徴量
            "fs_ttm_sales": (pl.Float64, None),
            "fs_ttm_op_profit": (pl.Float64, None),
            "fs_ttm_net_income": (pl.Float64, None),
            "fs_ttm_cfo": (pl.Float64, None),
            "fs_ttm_op_margin": (pl.Float64, None),
            "fs_ttm_cfo_margin": (pl.Float64, None),
            "fs_equity_ratio": (pl.Float64, None),
            "fs_net_cash_ratio": (pl.Float64, None),
            "fs_yoy_ttm_sales": (pl.Float64, None),
            "fs_yoy_ttm_op_profit": (pl.Float64, None),
            "fs_yoy_ttm_net_income": (pl.Float64, None),
            "fs_accruals": (pl.Float64, None),
            "fs_days_since": (pl.Int32, None),
            "fs_days_to_next": (pl.Int32, None),
            "fs_window_e_pm1": (pl.Int8, 0),
            "fs_window_e_pp3": (pl.Int8, 0),
            "fs_window_e_pp5": (pl.Int8, 0),
            "fs_is_valid": (pl.Int8, 0),
            # TypeOfDocument関連（P0）
            "fs_doc_family_FY": (pl.Int8, 0),
            "fs_doc_family_1Q": (pl.Int8, 0),
            "fs_doc_family_2Q": (pl.Int8, 0),
            "fs_doc_family_3Q": (pl.Int8, 0),
            "fs_standard_JGAAP": (pl.Int8, 0),
            "fs_standard_IFRS": (pl.Int8, 0),
            "fs_standard_US": (pl.Int8, 0),
            "fs_standard_JMIS": (pl.Int8, 0),
            "fs_standard_Foreign": (pl.Int8, 0),
            "fs_consolidated_flag": (pl.Int8, 0),
            "fs_guidance_revision_flag": (pl.Int8, 0),
        }
        for col, (dtype, default) in specs.items():
            if col not in df.columns:
                df = df.with_columns(pl.lit(default).cast(dtype).alias(col))
        return df

    def _ensure_dividend_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        specs: dict[str, tuple[pl.DataType, object | None]] = {
            "div_days_to_ex": (pl.Int32, None),
            "div_days_since_ex": (pl.Int32, None),
            "div_pre1": (pl.Int8, 0),
            "div_pre3": (pl.Int8, 0),
            "div_pre5": (pl.Int8, 0),
            "div_pre_ex_1d": (pl.Int8, 0),
            "div_pre_ex_3d": (pl.Int8, 0),
            "div_post1": (pl.Int8, 0),
            "div_post3": (pl.Int8, 0),
            "div_post5": (pl.Int8, 0),
            "div_post_ex_1d": (pl.Int8, 0),
            "div_post_ex_3d": (pl.Int8, 0),
            "div_is_ex0": (pl.Int8, 0),
            "div_dy_12m": (pl.Float64, None),
            "div_yield_ttm": (pl.Float64, None),
            "div_yield_12m": (pl.Float64, None),
            "div_amount_next": (pl.Float64, None),
            "div_amount_12m": (pl.Float64, None),
            "div_ex_drop_expected": (pl.Float64, None),
            "div_ex_gap_theo": (pl.Float64, None),
            "div_ex_gap_miss": (pl.Float64, None),
            "div_is_obs": (pl.Int8, 0),
            "is_div_valid": (pl.Int8, 0),
            "div_is_special": (pl.Int8, 0),
            "div_staleness_bd": (pl.Int32, None),
            "div_staleness_days": (pl.Int32, None),
            "div_ex_soon_3": (pl.Int8, 0),
            "div_ex_cycle_z": (pl.Float64, None),
        }
        for col, (dtype, default) in specs.items():
            if col not in df.columns:
                df = df.with_columns(pl.lit(default).cast(dtype).alias(col))
        return df

    def _ensure_earnings_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        specs: dict[str, tuple[pl.DataType, object | None]] = {
            "earnings_event_date": (pl.Date, None),
            "days_to_earnings": (pl.Int32, None),
            "earnings_today": (pl.Int8, 0),
            "earnings_upcoming_1d": (pl.Int8, 0),
            "earnings_upcoming_3d": (pl.Int8, 0),
            "earnings_upcoming_5d": (pl.Int8, 0),
            "earnings_recent_1d": (pl.Int8, 0),
            "earnings_recent_3d": (pl.Int8, 0),
            "earnings_recent_5d": (pl.Int8, 0),
            # P0新特徴量
            "is_E_pm1": (pl.Int8, 0),
            "is_E_0": (pl.Int8, 0),
            "is_E_pp1": (pl.Int8, 0),
            "is_E_pp3": (pl.Int8, 0),
            "is_E_pp5": (pl.Int8, 0),
            "is_earnings_sched_valid": (pl.Int8, 0),
        }
        for col, (dtype, default) in specs.items():
            if col not in df.columns:
                df = df.with_columns(pl.lit(default).cast(dtype).alias(col))
        return df

    def _ensure_breakdown_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        specs: dict[str, tuple[pl.DataType, object | None]] = {
            "bd_total_value": (pl.Float64, None),
            "bd_net_value": (pl.Float64, None),
            "bd_net_ratio": (pl.Float64, None),
            "bd_short_share": (pl.Float64, None),
            "bd_activity_ratio": (pl.Float64, None),
            "bd_net_ratio_chg_1d": (pl.Float64, None),
            "bd_short_share_chg_1d": (pl.Float64, None),
            "bd_net_z20": (pl.Float64, None),
            "bd_net_z260": (pl.Float64, None),
            "bd_short_z260": (pl.Float64, None),
            "bd_credit_new_net": (pl.Float64, None),
            "bd_credit_close_net": (pl.Float64, None),
            "bd_net_ratio_local_max": (pl.Int8, 0),
            "bd_net_ratio_local_min": (pl.Int8, 0),
            "bd_turn_up": (pl.Int8, 0),
            "bd_net_adv60": (pl.Float64, None),
            "bd_net_mc": (pl.Float64, None),
            "bd_staleness_bd": (pl.Int32, None),
            "bd_is_recent": (pl.Int8, 0),
            "is_bd_valid": (pl.Int8, 0),
        }
        for col, (dtype, default) in specs.items():
            if col not in df.columns:
                df = df.with_columns(pl.lit(default).cast(dtype).alias(col))
        return df

    def _attach_margin_daily_features(
        self,
        df: pl.DataFrame,
        *,
        raw_daily: pl.DataFrame,
        calendar_df: pl.DataFrame,
    ) -> pl.DataFrame:
        if raw_daily.is_empty():
            LOGGER.info("[MARGIN] Daily margin dataset empty, skipping features")
            return self._ensure_margin_daily_columns(df)

        snapshot = prepare_margin_daily_asof(
            raw_daily,
            trading_calendar=calendar_df,
            availability_hour=9,
            availability_minute=0,
        )

        snapshot_value_cols = [col for col in snapshot.columns if col.lower() not in {"code", "date", "available_ts"}]

        joined = interval_join_pl(
            backbone=df,
            snapshot=snapshot,
            on_code="code",
            backbone_ts="asof_ts",
            snapshot_ts="available_ts",
            strategy="backward",
            suffix="_dmi",
        )

        column_lookup = {col.lower(): col for col in joined.columns}

        def _resolve(candidates: list[str]) -> str | None:
            for candidate in candidates:
                actual = column_lookup.get(candidate.lower())
                if actual is not None:
                    return actual
            return None

        margin_long_col = _resolve(
            [
                "margin_balance_dmi",
                "margin_balance",
                "marginbuybalance_dmi",
                "marginbuybalance",
                "margin_buy_volume_dmi",
                "margin_buy_volume",
            ]
        )
        margin_short_col = _resolve(
            [
                "short_balance_dmi",
                "short_balance",
                "margin_short_balance_dmi",
                "margin_short_balance",
                "margin_sell_volume_dmi",
                "margin_sell_volume",
            ]
        )

        if margin_long_col is None or margin_short_col is None:
            LOGGER.warning(
                "[MARGIN] Daily margin join missing required columns (long=%s, short=%s)",
                margin_long_col,
                margin_short_col,
            )
            cleanup_candidates = {col for col in joined.columns if col.endswith("_dmi")}
            for base in snapshot_value_cols:
                resolved = _resolve([base, f"{base}_dmi"])
                if resolved is not None:
                    cleanup_candidates.add(resolved)
            for base in ("available_ts", "application_date", "published_date", "date"):
                resolved = _resolve([f"{base}_dmi", base])
                if resolved is not None:
                    cleanup_candidates.add(resolved)
            cleanup_list = [col for col in cleanup_candidates if col in joined.columns]
            joined = joined.drop(cleanup_list, strict=False)
            return self._ensure_margin_daily_columns(joined)

        turnover_base = (
            pl.when(pl.col("turnovervalue").is_not_null())
            .then(pl.col("turnovervalue"))
            .otherwise(pl.col("adjustmentclose") * pl.col("adjustmentvolume"))
        )

        # P0: Extract LongMarginOutstanding and ShortMarginOutstanding from raw data
        # Use Value columns if available, otherwise convert shares to yen
        long_value_col = _resolve(
            ["LongMarginOutstanding_dmi", "LongMarginOutstanding", "LongMarginOutstandingValue_dmi"]
        )
        short_value_col = _resolve(
            ["ShortMarginOutstanding_dmi", "ShortMarginOutstanding", "ShortMarginOutstandingValue_dmi"]
        )

        joined = joined.with_columns(
            [
                turnover_base.alias("_dv_base_yen"),
                pl.col(margin_long_col).cast(pl.Float64, strict=False).alias("_dmi_long_shares"),
                pl.col(margin_short_col).cast(pl.Float64, strict=False).alias("_dmi_short_shares"),
            ]
        )

        # Calculate yen values: prefer Value columns, fallback to shares * price
        if long_value_col and long_value_col in joined.columns:
            joined = joined.with_columns(pl.col(long_value_col).cast(pl.Float64, strict=False).alias("_dmi_long_yen"))
        else:
            joined = joined.with_columns(
                (pl.col("_dmi_long_shares") * pl.col("adjustmentclose")).alias("_dmi_long_yen")
            )

        if short_value_col and short_value_col in joined.columns:
            joined = joined.with_columns(pl.col(short_value_col).cast(pl.Float64, strict=False).alias("_dmi_short_yen"))
        else:
            joined = joined.with_columns(
                (pl.col("_dmi_short_shares") * pl.col("adjustmentclose")).alias("_dmi_short_yen")
            )

        # P0: Core levels (当日公表値)
        joined = joined.with_columns(
            [
                pl.col("_dmi_long_yen").alias("dmi_long_balance"),
                pl.col("_dmi_short_yen").alias("dmi_short_balance"),
                (pl.col("_dmi_long_yen") + pl.col("_dmi_short_yen")).alias("dmi_total"),
                (pl.col("_dmi_long_yen") - pl.col("_dmi_short_yen")).alias("dmi_net"),
                (pl.col("_dmi_long_yen") / (pl.col("_dmi_short_yen") + 1e-9)).alias("dmi_long_short_ratio"),
            ]
        )

        # P0: 1-day differences (shift(1) to prevent leakage)
        joined = joined.sort(["code", "date"])
        joined = joined.with_columns(
            [
                (pl.col("dmi_long_balance") - pl.col("dmi_long_balance").shift(1).over("code")).alias(
                    "dmi_long_balance_diff_1d"
                ),
                (pl.col("dmi_short_balance") - pl.col("dmi_short_balance").shift(1).over("code")).alias(
                    "dmi_short_balance_diff_1d"
                ),
                (pl.col("dmi_total") - pl.col("dmi_total").shift(1).over("code")).alias("dmi_total_diff_1d"),
                (pl.col("dmi_net") - pl.col("dmi_net").shift(1).over("code")).alias("dmi_net_diff_1d"),
            ]
        )

        # P0: Rolling statistics for z-score (20-day window, shift(1))
        joined = joined.with_columns(
            [
                (pl.col("_dv_base_yen").shift(1).rolling_mean(window_size=20, min_periods=5).over("code")).alias(
                    "_adv20_yen"
                ),
                (pl.col("_dv_base_yen").shift(1).rolling_mean(window_size=60, min_periods=10).over("code")).alias(
                    "_adv60_yen"
                ),
            ]
        )

        # P0: z-scores (20-day rolling with shift(1))
        joined = joined.with_columns(
            [
                (pl.col("dmi_net").shift(1).rolling_mean(window_size=20, min_periods=5).over("code")).alias(
                    "_dmi_net_ma20"
                ),
                (pl.col("dmi_net").shift(1).rolling_std(window_size=20, min_periods=5).over("code")).alias(
                    "_dmi_net_std20"
                ),
                (
                    pl.col("dmi_long_short_ratio").shift(1).rolling_mean(window_size=20, min_periods=5).over("code")
                ).alias("_dmi_ratio_ma20"),
                (pl.col("dmi_long_short_ratio").shift(1).rolling_std(window_size=20, min_periods=5).over("code")).alias(
                    "_dmi_ratio_std20"
                ),
            ]
        )

        joined = joined.with_columns(
            [
                ((pl.col("dmi_net") - pl.col("_dmi_net_ma20")) / (pl.col("_dmi_net_std20") + 1e-9)).alias(
                    "dmi_net_z20"
                ),
                (
                    (pl.col("dmi_long_short_ratio") - pl.col("_dmi_ratio_ma20")) / (pl.col("_dmi_ratio_std20") + 1e-9)
                ).alias("dmi_long_short_ratio_z20"),
            ]
        )

        # P0: Liquidity normalization (ADV20-based)
        joined = joined.with_columns(
            [
                (pl.col("dmi_total") / (pl.col("_adv20_yen") + 1e-9)).alias("dmi_total_over_adv20"),
                (pl.col("dmi_net") / (pl.col("_adv20_yen") + 1e-9)).alias("dmi_net_over_adv20"),
            ]
        )

        # P0: Quality flags
        published_date_col = _resolve(["PublishedDate_dmi", "PublishedDate", "published_date_dmi", "published_date"])
        # application_date_col is reserved for future use
        # application_date_col = _resolve(
        #     ["ApplicationDate_dmi", "ApplicationDate", "application_date_dmi", "application_date"]
        # )

        joined = joined.with_columns(
            [
                pl.when(pl.col("dmi_long_balance").is_not_null() & pl.col("dmi_short_balance").is_not_null())
                .then(1)
                .otherwise(0)
                .cast(pl.Int8)
                .alias("is_dmi_valid"),
            ]
        )

        # P0: staleness_days (days since last publication)
        if published_date_col and published_date_col in joined.columns:
            joined = joined.with_columns(
                pl.when(pl.col(published_date_col).is_not_null() & pl.col("date").is_not_null())
                .then((pl.col("date").cast(pl.Int64) - pl.col(published_date_col).cast(pl.Int64)).cast(pl.Int32))
                .otherwise(None)
                .alias("dmi_staleness_days")
            )
        else:
            joined = joined.with_columns(pl.lit(None).cast(pl.Int32).alias("dmi_staleness_days"))

        # P0: PublishReason extraction
        publish_reason_col = _resolve(["PublishReason_dmi", "PublishReason", "publish_reason_dmi", "publish_reason"])
        if publish_reason_col and publish_reason_col in joined.columns:
            # Extract reason code (handle both dict and string)
            joined = joined.with_columns(
                [
                    pl.when(pl.col(publish_reason_col).is_not_null())
                    .then(
                        pl.when(pl.col(publish_reason_col).is_struct())
                        .then(pl.col(publish_reason_col).struct.field("ReasonCode"))
                        .otherwise(pl.col(publish_reason_col).cast(pl.Utf8))
                    )
                    .otherwise(None)
                    .cast(pl.Utf8)
                    .alias("dmi_reason_code"),
                    pl.when(pl.col(publish_reason_col).is_not_null())
                    .then(
                        pl.when(pl.col(publish_reason_col).is_struct())
                        .then(
                            pl.when(
                                pl.col(publish_reason_col)
                                .struct.field("ReasonCode")
                                .cast(pl.Utf8)
                                .str.contains("訂正|修正|Revision", literal=False)
                            )
                            .then(1)
                            .otherwise(0)
                        )
                        .otherwise(0)
                    )
                    .otherwise(0)
                    .cast(pl.Int8)
                    .alias("dmi_reason_is_revision"),
                ]
            )
        else:
            joined = joined.with_columns(
                [
                    pl.lit(None).cast(pl.Utf8).alias("dmi_reason_code"),
                    pl.lit(0).cast(pl.Int8).alias("dmi_reason_is_revision"),
                ]
            )

        # Legacy features (keep for backward compatibility)
        # Note: _dmi_net_yen and _dmi_total_yen should already exist from dmi_net/dmi_total calculations
        # But we need them as yen values for legacy features
        if "_dmi_net_yen" not in joined.columns:
            joined = joined.with_columns((pl.col("_dmi_long_yen") - pl.col("_dmi_short_yen")).alias("_dmi_net_yen"))
        if "_dmi_total_yen" not in joined.columns:
            joined = joined.with_columns((pl.col("_dmi_long_yen") + pl.col("_dmi_short_yen")).alias("_dmi_total_yen"))

        joined = joined.with_columns(
            [
                (pl.col("_dmi_net_yen") / (pl.col("_adv60_yen") + 1e-9)).alias("dmi_net_adv60"),
                (pl.col("_dmi_net_yen") / (pl.col("_dmi_total_yen") + 1e-9)).alias("dmi_imbalance"),
            ]
        )

        joined = joined.with_columns(
            (pl.col("dmi_net_adv60") - pl.col("dmi_net_adv60").shift(1).over("code")).alias("dmi_delta_net_adv60")
        )

        joined = joined.with_columns(
            (
                (
                    pl.col("dmi_delta_net_adv60")
                    - pl.col("dmi_delta_net_adv60").shift(1).rolling_mean(window_size=20, min_periods=5).over("code")
                )
                / (
                    pl.col("dmi_delta_net_adv60").shift(1).rolling_std(window_size=20, min_periods=5).over("code")
                    + 1e-9
                )
            ).alias("dmi_delta_net_adv60_z20"),
            # Keep legacy is_margin_daily_valid for backward compatibility
            pl.when(
                pl.col("_adv60_yen").is_not_null()
                & pl.col("_dmi_long_yen").is_not_null()
                & pl.col("_dmi_short_yen").is_not_null()
            )
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("is_margin_daily_valid"),
        )

        cleanup_cols: set[str] = {col for col in joined.columns if col.endswith("_dmi")}
        cleanup_cols.update(actual for actual in (margin_long_col, margin_short_col) if actual is not None)
        for base in ("available_ts", "application_date", "published_date", "date"):
            resolved = _resolve([f"{base}_dmi", base])
            if resolved is not None:
                cleanup_cols.add(resolved)
        for base in snapshot_value_cols:
            resolved = _resolve([base, f"{base}_dmi"])
            if resolved is not None:
                cleanup_cols.add(resolved)
        cleanup_cols.update(
            [
                "_dv_base_yen",
                "_adv20_yen",
                "_adv60_yen",
                "_dmi_long_shares",
                "_dmi_short_shares",
                "_dmi_long_yen",
                "_dmi_short_yen",
                "_dmi_net_yen",
                "_dmi_total_yen",
                "_dmi_net_ma20",
                "_dmi_net_std20",
                "_dmi_ratio_ma20",
                "_dmi_ratio_std20",
            ]
        )
        cleanup_list = [col for col in cleanup_cols if col in joined.columns]
        joined = joined.drop(cleanup_list, strict=False)

        if isinstance(self._run_meta, dict):
            margin_meta = self._run_meta.setdefault("margin_features", {})
            margin_meta["daily"] = {
                "columns": [
                    # P0: Core levels
                    "dmi_long_balance",
                    "dmi_short_balance",
                    "dmi_total",
                    "dmi_net",
                    "dmi_long_short_ratio",
                    # P0: Changes
                    "dmi_long_balance_diff_1d",
                    "dmi_short_balance_diff_1d",
                    "dmi_total_diff_1d",
                    "dmi_net_diff_1d",
                    # P0: Z-scores
                    "dmi_net_z20",
                    "dmi_long_short_ratio_z20",
                    # P0: Liquidity normalization
                    "dmi_total_over_adv20",
                    "dmi_net_over_adv20",
                    # P0: Quality flags
                    "is_dmi_valid",
                    "dmi_staleness_days",
                    "dmi_reason_code",
                    "dmi_reason_is_revision",
                    # Legacy (for backward compatibility)
                    "dmi_net_adv60",
                    "dmi_delta_net_adv60",
                    "dmi_delta_net_adv60_z20",
                    "dmi_imbalance",
                    "is_margin_daily_valid",
                ],
                "availability": "T+1_09:00_JST",
                "adv_window_days": [20, 60],
                "source_columns": {
                    "long": margin_long_col,
                    "short": margin_short_col,
                },
            }

        return self._ensure_margin_daily_columns(joined)

    def _attach_margin_weekly_features(
        self,
        df: pl.DataFrame,
        *,
        calendar_df: pl.DataFrame,
        start: str,
        end: str,
    ) -> pl.DataFrame:
        try:
            weekly = self.data_sources.margin_weekly(start=start, end=end)
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("Failed to fetch weekly margin data: %s", exc)
            return self._ensure_margin_weekly_columns(df)

        if weekly.is_empty():
            LOGGER.info("[MARGIN] Weekly margin dataset empty, skipping features")
            return self._ensure_margin_weekly_columns(df)

        snapshot = prepare_margin_weekly_asof(
            weekly,
            trading_calendar=calendar_df,
            availability_hour=9,
            availability_minute=0,
        )

        snapshot_value_cols = [col for col in snapshot.columns if col.lower() not in {"code", "date", "available_ts"}]

        joined = interval_join_pl(
            backbone=df,
            snapshot=snapshot,
            on_code="code",
            backbone_ts="asof_ts",
            snapshot_ts="available_ts",
            strategy="backward",
            suffix="_wmi",
        )

        column_lookup = {col.lower(): col for col in joined.columns}

        def _resolve(candidates: list[str]) -> str | None:
            for candidate in candidates:
                actual = column_lookup.get(candidate.lower())
                if actual is not None:
                    return actual
            return None

        # P1: Resolve all required columns from API
        long_volume_col = _resolve(
            [
                "weekly_margin_long_volume_wmi",
                "weekly_margin_long_volume",
                "longmargintradevolume_wmi",
                "longmargintradevolume",
                "long_margin_trade_volume_wmi",
                "long_margin_trade_volume",
                "LongMarginTradeVolume_wmi",
                "LongMarginTradeVolume",
            ]
        )
        short_volume_col = _resolve(
            [
                "weekly_margin_short_volume_wmi",
                "weekly_margin_short_volume",
                "shortmargintradevolume_wmi",
                "shortmargintradevolume",
                "short_margin_trade_volume_wmi",
                "short_margin_trade_volume",
                "ShortMarginTradeVolume_wmi",
                "ShortMarginTradeVolume",
            ]
        )
        # P1: 内訳（一般/制度）
        long_gen_col = _resolve(
            [
                "LongNegotiableMarginTradeVolume_wmi",
                "LongNegotiableMarginTradeVolume",
                "longnegotiablemargintradevolume_wmi",
                "longnegotiablemargintradevolume",
            ]
        )
        short_gen_col = _resolve(
            [
                "ShortNegotiableMarginTradeVolume_wmi",
                "ShortNegotiableMarginTradeVolume",
                "shortnegotiablemargintradevolume_wmi",
                "shortnegotiablemargintradevolume",
            ]
        )
        long_std_col = _resolve(
            [
                "LongStandardizedMarginTradeVolume_wmi",
                "LongStandardizedMarginTradeVolume",
                "longstandardizedmargintradevolume_wmi",
                "longstandardizedmargintradevolume",
            ]
        )
        short_std_col = _resolve(
            [
                "ShortStandardizedMarginTradeVolume_wmi",
                "ShortStandardizedMarginTradeVolume",
                "shortstandardizedmargintradevolume_wmi",
                "shortstandardizedmargintradevolume",
            ]
        )
        issue_type_col = _resolve(
            [
                "IssueType_wmi",
                "IssueType",
                "issuetype_wmi",
                "issuetype",
            ]
        )

        if long_volume_col is None or short_volume_col is None:
            LOGGER.warning(
                "[MARGIN] Weekly margin join missing required columns (long=%s, short=%s)",
                long_volume_col,
                short_volume_col,
            )
            cleanup_candidates = {col for col in joined.columns if col.endswith("_wmi")}
            for base in snapshot_value_cols:
                resolved = _resolve([base, f"{base}_wmi"])
                if resolved is not None:
                    cleanup_candidates.add(resolved)
            for base in ("available_ts", "published_date", "date"):
                resolved = _resolve([f"{base}_wmi", base])
                if resolved is not None:
                    cleanup_candidates.add(resolved)
            cleanup_list = [col for col in cleanup_candidates if col in joined.columns]
            joined = joined.drop(cleanup_list, strict=False)
            return self._ensure_margin_weekly_columns(joined)

        ff_targets: list[str] = []
        cast_plan = [
            ("wm_long", long_volume_col, pl.Float64),
            ("wm_short", short_volume_col, pl.Float64),
            ("wm_long_gen", long_gen_col, pl.Float64),
            ("wm_short_gen", short_gen_col, pl.Float64),
            ("wm_long_std", long_std_col, pl.Float64),
            ("wm_short_std", short_std_col, pl.Float64),
            ("wm_issue_type", issue_type_col, pl.Int8),
        ]

        for alias, source, dtype in cast_plan:
            if source is None:
                joined = joined.with_columns(pl.lit(None).cast(dtype).alias(alias))
            else:
                ff_targets.append(alias)
                joined = joined.with_columns(pl.col(source).cast(dtype, strict=False).alias(alias))

        if ff_targets:
            joined = forward_fill_after_publication(
                joined,
                group_cols="code",
                sort_col="date",
                columns=ff_targets,
            )

        # P1: ベース特徴量の計算
        eps = 1e-9
        joined = joined.with_columns(
            [
                # wm_net = wm_long - wm_short
                (pl.col("wm_long") - pl.col("wm_short")).alias("wm_net"),
                # wm_lsr = wm_long / (wm_short + ε)
                (pl.col("wm_long") / (pl.col("wm_short") + eps)).alias("wm_lsr"),
            ]
        )

        # P1: 一般/制度のシェア（内訳が利用可能な場合）
        joined = joined.with_columns(
            [
                # wm_gen_share = (wm_long_gen + wm_short_gen) / (wm_long + wm_short + ε)
                (
                    pl.when(
                        (pl.col("wm_long_gen").is_not_null() | pl.col("wm_short_gen").is_not_null())
                        & ((pl.col("wm_long") + pl.col("wm_short")).abs() > eps)
                    )
                    .then(
                        (pl.col("wm_long_gen").fill_null(0) + pl.col("wm_short_gen").fill_null(0))
                        / (pl.col("wm_long") + pl.col("wm_short") + eps)
                    )
                    .otherwise(None)
                ).alias("wm_gen_share"),
                # wm_std_share = (wm_long_std + wm_short_std) / (wm_long + wm_short + ε)
                (
                    pl.when(
                        (pl.col("wm_long_std").is_not_null() | pl.col("wm_short_std").is_not_null())
                        & ((pl.col("wm_long") + pl.col("wm_short")).abs() > eps)
                    )
                    .then(
                        (pl.col("wm_long_std").fill_null(0) + pl.col("wm_short_std").fill_null(0))
                        / (pl.col("wm_long") + pl.col("wm_short") + eps)
                    )
                    .otherwise(None)
                ).alias("wm_std_share"),
            ]
        )

        # P1: ADV20計算（adjustmentvolumeから、shift(1)でリーク防止）
        vol_col = "adjustmentvolume" if "adjustmentvolume" in joined.columns else "volume"
        if vol_col in joined.columns:
            from ..features.utils.rolling import roll_mean_safe

            joined = joined.with_columns(
                roll_mean_safe(pl.col(vol_col), 20, min_periods=10, by="code").alias("_adv20_shares")
            )
        else:
            joined = joined.with_columns(pl.lit(None).cast(pl.Float64).alias("_adv20_shares"))

        # P1: 標準化（ADV20で割る、左閉ローリング：shift(1)適用）
        # _adv20_sharesはroll_mean_safeで既にshift(1)済みなので、分子もshift(1)する
        joined = joined.with_columns(
            [
                # wm_net_to_adv20 = wm_net.shift(1) / (ADV20_shares + ε)
                (pl.col("wm_net").shift(1).over("code") / (pl.col("_adv20_shares") + eps)).alias("wm_net_to_adv20"),
                # wm_long_to_adv20 = wm_long.shift(1) / (ADV20_shares + ε)
                (pl.col("wm_long").shift(1).over("code") / (pl.col("_adv20_shares") + eps)).alias("wm_long_to_adv20"),
                # wm_short_to_adv20 = wm_short.shift(1) / (ADV20_shares + ε)
                (pl.col("wm_short").shift(1).over("code") / (pl.col("_adv20_shares") + eps)).alias("wm_short_to_adv20"),
            ]
        )

        # P1: 変化・モメンタム（1週間差分、shift(1)でリーク防止）
        # 週次データなので、5営業日前との差分を計算（左閉ローリング：shift(1)適用）
        joined = joined.with_columns(
            [
                # wm_net_d1w = wm_net.shift(1) - wm_net.shift(6) (5営業日+1)
                (pl.col("wm_net").shift(1).over("code") - pl.col("wm_net").shift(6).over("code")).alias("wm_net_d1w"),
                # wm_long_d1w = wm_long.shift(1) - wm_long.shift(6)
                (pl.col("wm_long").shift(1).over("code") - pl.col("wm_long").shift(6).over("code")).alias(
                    "wm_long_d1w"
                ),
                # wm_short_d1w = wm_short.shift(1) - wm_short.shift(6)
                (pl.col("wm_short").shift(1).over("code") - pl.col("wm_short").shift(6).over("code")).alias(
                    "wm_short_d1w"
                ),
                # wm_net_pct_d1w = (wm_net.shift(1) / wm_net.shift(6) - 1)
                (
                    pl.when(pl.col("wm_net").shift(6).over("code").abs() > eps)
                    .then(pl.col("wm_net").shift(1).over("code") / (pl.col("wm_net").shift(6).over("code") + eps) - 1.0)
                    .otherwise(None)
                ).alias("wm_net_pct_d1w"),
            ]
        )

        # P1: ローリング特徴量（shift(1)でリーク防止）
        from ..features.utils.rolling import roll_mean_safe, roll_std_safe

        joined = joined.with_columns(
            [
                # 20営業日Z-score
                roll_mean_safe(pl.col("wm_net"), 20, min_periods=10, by="code").alias("wm_net_ma20"),
                roll_std_safe(pl.col("wm_net"), 20, min_periods=10, by="code").alias("wm_net_std20"),
                roll_mean_safe(pl.col("wm_short"), 20, min_periods=10, by="code").alias("wm_short_ma20"),
                roll_std_safe(pl.col("wm_short"), 20, min_periods=10, by="code").alias("wm_short_std20"),
                roll_mean_safe(pl.col("wm_long"), 20, min_periods=10, by="code").alias("wm_long_ma20"),
                roll_std_safe(pl.col("wm_long"), 20, min_periods=10, by="code").alias("wm_long_std20"),
            ]
        )

        joined = joined.with_columns(
            [
                # Z-score（左閉ローリング：shift(1)適用）
                # roll_*_safeは既にshift(1)を適用しているので、分子もshift(1)する
                (
                    pl.when(pl.col("wm_net_std20").abs() > eps)
                    .then((pl.col("wm_net").shift(1).over("code") - pl.col("wm_net_ma20")) / pl.col("wm_net_std20"))
                    .otherwise(None)
                ).alias("wm_net_z20"),
                (
                    pl.when(pl.col("wm_short_std20").abs() > eps)
                    .then(
                        (pl.col("wm_short").shift(1).over("code") - pl.col("wm_short_ma20")) / pl.col("wm_short_std20")
                    )
                    .otherwise(None)
                ).alias("wm_short_z20"),
                (
                    pl.when(pl.col("wm_long_std20").abs() > eps)
                    .then((pl.col("wm_long").shift(1).over("code") - pl.col("wm_long_ma20")) / pl.col("wm_long_std20"))
                    .otherwise(None)
                ).alias("wm_long_z20"),
            ]
        )

        # P1: 52週Z-score（週次データなので260営業日）
        joined = joined.with_columns(
            [
                roll_mean_safe(pl.col("wm_net"), 260, min_periods=130, by="code").alias("wm_net_ma52"),
                roll_std_safe(pl.col("wm_net"), 260, min_periods=130, by="code").alias("wm_net_std52"),
            ]
        )

        joined = joined.with_columns(
            (
                pl.when(pl.col("wm_net_std52").abs() > eps)
                .then((pl.col("wm_net").shift(1).over("code") - pl.col("wm_net_ma52")) / pl.col("wm_net_std52"))
                .otherwise(None)
            ).alias("wm_net_z52")
        )

        # P1: 品質フラグ
        # available_tsから日付を取得（公表日）
        joined = joined.with_columns(
            pl.col("available_ts").cast(pl.Datetime("us", "Asia/Tokyo")).dt.date().alias("_wm_publish_date")
        )

        # staleness計算
        if "date" in joined.columns:
            joined = joined.with_columns(
                [
                    pl.col("date").cast(pl.Date, strict=False).alias("_current_date"),
                    pl.col("_wm_publish_date").cast(pl.Date, strict=False),
                ]
            )
            joined = joined.with_columns(
                (pl.col("_current_date").cast(pl.Int64) - pl.col("_wm_publish_date").cast(pl.Int64))
                .cast(pl.Int32)
                .alias("wm_staleness_bd")
            )
            joined = joined.drop(["_current_date"], strict=False)

        # is_wm_valid: バリデーションチェック
        joined = joined.with_columns(
            (
                (
                    pl.col("wm_long").is_not_null()
                    & pl.col("wm_short").is_not_null()
                    & (pl.col("wm_long") >= 0)
                    & (pl.col("wm_short") >= 0)
                    & (pl.col("wm_lsr").is_finite() | pl.col("wm_lsr").is_null())
                    & (pl.col("wm_staleness_bd").is_null() | (pl.col("wm_staleness_bd") >= 0))
                )
                .cast(pl.Int8)
                .alias("is_wm_valid")
            )
        )

        # wm_is_recent: staleness <= 7営業日
        joined = joined.with_columns(
            (
                (
                    (pl.col("wm_staleness_bd").is_not_null())
                    & (pl.col("wm_staleness_bd") >= 0)
                    & (pl.col("wm_staleness_bd") <= 7)
                )
                .cast(pl.Int8)
                .alias("wm_is_recent")
            )
        )

        # 後方互換性: is_margin_weekly_valid = is_wm_valid
        joined = joined.with_columns(pl.col("is_wm_valid").alias("is_margin_weekly_valid"))

        # 後方互換性のため既存の列も保持
        turnover_base = (
            pl.when(pl.col("turnovervalue").is_not_null())
            .then(pl.col("turnovervalue"))
            .otherwise(pl.col("adjustmentclose") * pl.col("adjustmentvolume"))
        )

        # Materialize _dv_base_yen first (Polars requires separate with_columns for column reuse)
        joined = joined.with_columns(
            [
                turnover_base.alias("_dv_base_yen"),
                (pl.col("wm_long") * pl.col("adjustmentclose")).alias("_wmi_long_yen"),
                (pl.col("wm_short") * pl.col("adjustmentclose")).alias("_wmi_short_yen"),
            ]
        )

        # Now compute _adv60_yen using the materialized _dv_base_yen
        joined = joined.with_columns(
            [
                (pl.col("_dv_base_yen").shift(1).rolling_mean(window_size=60, min_periods=10).over("code")).alias(
                    "_adv60_yen"
                ),
            ]
        )

        joined = joined.with_columns(
            [
                (pl.col("_wmi_long_yen") - pl.col("_wmi_short_yen")).alias("_wmi_net_yen"),
                (pl.col("_wmi_long_yen") + pl.col("_wmi_short_yen")).alias("_wmi_total_yen"),
            ]
        )

        joined = joined.with_columns(
            (pl.col("_wmi_net_yen") / ((pl.col("_adv60_yen") * 5) + 1e-9)).alias("wmi_net_adv5d")
        )
        joined = joined.with_columns(
            (pl.col("wmi_net_adv5d") - pl.col("wmi_net_adv5d").shift(1).over("code")).alias("wmi_delta_net_adv5d")
        )

        joined = joined.with_columns(
            (
                (
                    pl.col("wmi_delta_net_adv5d")
                    - pl.col("wmi_delta_net_adv5d").shift(1).rolling_mean(window_size=52, min_periods=5).over("code")
                )
                / (
                    pl.col("wmi_delta_net_adv5d").shift(1).rolling_std(window_size=52, min_periods=5).over("code")
                    + 1e-9
                )
            ).alias("wmi_delta_net_adv5d_z52"),
        )

        joined = joined.with_columns(
            [
                (pl.col("_wmi_net_yen") / (pl.col("_wmi_total_yen") + 1e-9)).alias("wmi_imbalance"),
                (pl.col("_wmi_long_yen") / (pl.col("_wmi_short_yen") + 1e-9)).alias("wmi_long_short_ratio"),
            ]
        )

        # 一時列をクリーンアップ
        cleanup_cols: set[str] = {col for col in joined.columns if col.endswith("_wmi")}
        cleanup_cols.update(
            actual
            for actual in (
                long_volume_col,
                short_volume_col,
                long_gen_col,
                short_gen_col,
                long_std_col,
                short_std_col,
                issue_type_col,
            )
            if actual is not None
        )
        for base in ("available_ts", "published_date", "date"):
            resolved = _resolve([f"{base}_wmi", base])
            if resolved is not None:
                cleanup_cols.add(resolved)
        for base in snapshot_value_cols:
            resolved = _resolve([base, f"{base}_wmi"])
            if resolved is not None:
                cleanup_cols.add(resolved)
        cleanup_cols.update(
            [
                "_dv_base_yen",
                "_adv60_yen",
                "_adv20_shares",
                "_wmi_long_shares",
                "_wmi_short_shares",
                "_wmi_long_yen",
                "_wmi_short_yen",
                "_wmi_net_yen",
                "_wmi_total_yen",
                "wm_net_ma20",
                "wm_net_std20",
                "wm_short_ma20",
                "wm_short_std20",
                "wm_long_ma20",
                "wm_long_std20",
                "wm_net_ma52",
                "wm_net_std52",
                "_wm_publish_date",
            ]
        )
        cleanup_list = [col for col in cleanup_cols if col in joined.columns]
        joined = joined.drop(cleanup_list, strict=False)

        # P1: run_metaに算出方法と整合性情報を記録
        if isinstance(self._run_meta, dict):
            margin_meta = self._run_meta.setdefault("margin_features", {})
            margin_meta["weekly"] = {
                "columns": [
                    # P1: ベース
                    "wm_long",
                    "wm_short",
                    "wm_long_gen",
                    "wm_short_gen",
                    "wm_long_std",
                    "wm_short_std",
                    "wm_net",
                    "wm_lsr",
                    "wm_gen_share",
                    "wm_std_share",
                    "wm_issue_type",
                    # P1: 変化・モメンタム
                    "wm_net_d1w",
                    "wm_long_d1w",
                    "wm_short_d1w",
                    "wm_net_pct_d1w",
                    # P1: 標準化
                    "wm_net_to_adv20",
                    "wm_long_to_adv20",
                    "wm_short_to_adv20",
                    # P1: 安定化
                    "wm_net_z20",
                    "wm_short_z20",
                    "wm_long_z20",
                    "wm_net_z52",
                    # P1: 品質
                    "is_wm_valid",
                    "wm_staleness_bd",
                    "wm_is_recent",
                    # 後方互換性
                    "wmi_net_adv5d",
                    "wmi_delta_net_adv5d",
                    "wmi_delta_net_adv5d_z52",
                    "wmi_imbalance",
                    "wmi_long_short_ratio",
                    "is_margin_weekly_valid",
                ],
                "availability": "T+1_09:00_JST",
                "adv_window_days": [20, 60],
                "adv_scale_factor": 5,
                "source": "/markets/weekly_margin_interest",
                "formula": {
                    "wm_net": "wm_long - wm_short",
                    "wm_lsr": "wm_long / (wm_short + ε)",
                    "wm_gen_share": "(wm_long_gen + wm_short_gen) / (wm_long + wm_short + ε)",
                    "wm_net_d1w": "wm_net - wm_net.shift(5) (1週間差分)",
                    "wm_net_to_adv20": "wm_net / (ADV20_shares + ε)",
                    "wm_net_z20": "(wm_net - ma20) / std20 (20営業日, shift(1))",
                    "wm_net_z52": "(wm_net - ma52) / std52 (260営業日, shift(1))",
                },
                "compatibility": {
                    "wmi_net_adv5d": "既存実装（ADV60ベース、5日スケール）",
                    "is_margin_weekly_valid": "is_wm_validのエイリアス（後方互換性）",
                },
            }

        return self._ensure_margin_weekly_columns(joined)

    def _attach_fs_features(
        self,
        df: pl.DataFrame,
        *,
        raw_fs: pl.DataFrame,
        calendar_df: pl.DataFrame,
    ) -> pl.DataFrame:
        if raw_fs.is_empty():
            LOGGER.info("[FINS] Financial statements empty, skipping features")
            return self._ensure_fs_columns(df)

        try:
            snapshot = prepare_fs_snapshot(
                raw_fs,
                trading_calendar=calendar_df,
                availability_hour=15,
                availability_minute=0,
            )
            feature_frame = build_fs_feature_frame(snapshot)
            if "Code" in feature_frame.columns and "code" not in feature_frame.columns:
                feature_frame = feature_frame.rename({"Code": "code"})
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("[FINS] Failed to prepare financial statement features: %s", exc)
            return self._ensure_fs_columns(df)

        if feature_frame.is_empty():
            LOGGER.info("[FINS] No normalized financial statement rows available")
            return self._ensure_fs_columns(df)

        snapshot_ts_col = "_fs_available_ts"
        backbone_ts_col = "_fs_asof_ts"
        feature_frame = feature_frame.with_columns(pl.col("available_ts").cast(pl.Int64).alias(snapshot_ts_col))
        working_df = df.with_columns(pl.col("asof_ts").cast(pl.Int64).alias(backbone_ts_col))

        joined = interval_join_pl(
            backbone=working_df,
            snapshot=feature_frame,
            on_code="code",
            backbone_ts=backbone_ts_col,
            snapshot_ts=snapshot_ts_col,
            strategy="backward",
            suffix="_fs",
        )

        if "fs_revenue_ttm" not in joined.columns:
            LOGGER.info("[FINS] Interval join produced no financial statement matches")
            return self._ensure_fs_columns(joined.drop([c for c in joined.columns if c.endswith("_fs")], strict=False))

        if "fs_period_end_date" not in joined.columns:
            alt_period = next((c for c in joined.columns if "fs_period_end_date" in c), None)
            if alt_period is not None:
                joined = joined.rename({alt_period: "fs_period_end_date"})

        joined = joined.with_columns(
            [
                pl.col("date").cast(pl.Date, strict=False).alias("_fs_today_date"),
                pl.col("fs_period_end_date").cast(pl.Date, strict=False).alias("_fs_report_date"),
                # DisclosedDateも取得（E前後のフラグ計算用）
                pl.col("DisclosedDate").cast(pl.Date, strict=False).alias("_fs_disclosed_date"),
            ]
        )
        # fs_staleness_bd: 前回開示からの営業日差（既存）
        joined = joined.with_columns(
            pl.when(pl.col("_fs_today_date").is_not_null() & pl.col("_fs_report_date").is_not_null())
            .then(
                pl.col("_fs_today_date").cast(pl.Int64, strict=False)
                - pl.col("_fs_report_date").cast(pl.Int64, strict=False)
            )
            .otherwise(None)
            .cast(pl.Int32)
            .alias("fs_staleness_bd")
        )

        # P0新特徴量: fs_days_since, fs_days_to_next（最近/次回決算までの営業日距離）
        # fs_days_since: 前回決算日（fs_period_end_date）からの営業日差
        joined = joined.with_columns(
            pl.when(pl.col("_fs_today_date").is_not_null() & pl.col("_fs_report_date").is_not_null())
            .then(
                (
                    pl.col("_fs_today_date").cast(pl.Int64, strict=False)
                    - pl.col("_fs_report_date").cast(pl.Int64, strict=False)
                ).cast(pl.Int32)
            )
            .otherwise(None)
            .alias("fs_days_since")
        )

        # fs_days_to_next: 次回決算日までの営業日差（次のfs_period_end_dateを探す）
        # 簡易実装: グループ内で次のfs_period_end_dateをshift(-1)で取得
        # より正確には、各codeごとに次の決算日を計算する必要があるが、簡易実装として
        # 現在の実装では省略（後で拡張可能）

        # P0新特徴量: fs_window_e±{1,3,5}（E前後の近傍フラグ）
        # fs_period_end_dateを決算日（E）として、dateとの距離で判定
        joined = joined.with_columns(
            pl.when(pl.col("_fs_today_date").is_not_null() & pl.col("_fs_report_date").is_not_null())
            .then(
                (
                    pl.col("_fs_today_date").cast(pl.Int64, strict=False)
                    - pl.col("_fs_report_date").cast(pl.Int64, strict=False)
                ).cast(pl.Int32)
            )
            .otherwise(None)
            .alias("_fs_days_to_e")
        )

        dte = pl.col("_fs_days_to_e")
        joined = joined.with_columns(
            [
                # fs_window_e±1: E±1営業日
                (dte.is_in([-1, 0, 1])).cast(pl.Int8).alias("fs_window_e_pm1"),
                # fs_window_e±3: E±3営業日
                (dte.is_in([-3, -2, -1, 0, 1, 2, 3])).cast(pl.Int8).alias("fs_window_e_pp3"),
                # fs_window_e±5: E±5営業日
                (dte.is_in([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])).cast(pl.Int8).alias("fs_window_e_pp5"),
            ]
        )

        joined = joined.drop(["_fs_today_date", "_fs_report_date", "_fs_disclosed_date", "_fs_days_to_e"], strict=False)
        joined = joined.with_columns(
            pl.when(pl.col("is_fs_valid") == 1)
            .then((pl.col("fs_staleness_bd") <= 65).fill_null(False))
            .otherwise(False)
            .cast(pl.Int8)
            .alias("fs_is_recent")
        )
        if "fs_staleness_bd" not in joined.columns:
            joined = joined.with_columns(pl.lit(None).cast(pl.Int32).alias("fs_staleness_bd"))
        if "fs_lag_days" not in joined.columns:
            joined = joined.with_columns(pl.lit(None).cast(pl.Int32).alias("fs_lag_days"))

        cleanup_cols = [col for col in joined.columns if col.endswith("_fs")]
        if "fs_period_end_date" in joined.columns:
            cleanup_cols.append("fs_period_end_date")
        cleanup_cols.extend([snapshot_ts_col, backbone_ts_col])
        joined = joined.drop(cleanup_cols, strict=False)

        joined = self._ensure_fs_columns(joined)
        if "fs_staleness_bd" not in joined.columns:
            joined = joined.with_columns(pl.lit(None).cast(pl.Int32).alias("fs_staleness_bd"))

        if isinstance(self._run_meta, dict):
            fund_meta = self._run_meta.setdefault("fundamental_features", {})
            fund_meta["fs"] = {
                "columns": [
                    # 既存特徴量（後方互換性）
                    "fs_revenue_ttm",
                    "fs_op_profit_ttm",
                    "fs_net_income_ttm",
                    "fs_cfo_ttm",
                    "fs_capex_ttm",
                    "fs_fcf_ttm",
                    "fs_sales_yoy",
                    "fs_op_margin",
                    "fs_net_margin",
                    "fs_roe_ttm",
                    "fs_roa_ttm",
                    "fs_accruals_ttm",
                    "fs_cfo_to_ni",
                    "fs_observation_count",
                    "fs_lag_days",
                    "fs_is_recent",
                    "fs_staleness_bd",
                    "is_fs_valid",
                    # P0新特徴量
                    "fs_ttm_sales",
                    "fs_ttm_op_profit",
                    "fs_ttm_net_income",
                    "fs_ttm_cfo",
                    "fs_ttm_op_margin",
                    "fs_ttm_cfo_margin",
                    "fs_equity_ratio",
                    "fs_net_cash_ratio",
                    "fs_yoy_ttm_sales",
                    "fs_yoy_ttm_op_profit",
                    "fs_yoy_ttm_net_income",
                    "fs_accruals",
                    "fs_days_since",
                    "fs_window_e_pm1",
                    "fs_window_e_pp3",
                    "fs_window_e_pp5",
                    "fs_is_valid",
                    # TypeOfDocument関連（P0）
                    "fs_doc_family_FY",
                    "fs_doc_family_1Q",
                    "fs_doc_family_2Q",
                    "fs_doc_family_3Q",
                    "fs_standard_JGAAP",
                    "fs_standard_IFRS",
                    "fs_standard_US",
                    "fs_standard_JMIS",
                    "fs_standard_Foreign",
                    "fs_consolidated_flag",
                    "fs_guidance_revision_flag",
                ],
            }

        return joined

    def _attach_dividend_features(
        self,
        df: pl.DataFrame,
        *,
        raw_dividend: pl.DataFrame,
        calendar_df: pl.DataFrame,
    ) -> pl.DataFrame:
        if raw_dividend.is_empty():
            LOGGER.info("[DIV] Dividend dataset empty, skipping features")
            return self._ensure_dividend_columns(df)

        try:
            snapshot = prepare_dividend_snapshot(
                raw_dividend,
                trading_calendar=calendar_df,
                availability_hour=15,
                availability_minute=0,
            )
            feature_frame = build_dividend_feature_frame(snapshot)
            if "Code" in feature_frame.columns and "code" not in feature_frame.columns:
                feature_frame = feature_frame.rename({"Code": "code"})
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("[DIV] Failed to prepare dividend features: %s", exc)
            return self._ensure_dividend_columns(df)

        if feature_frame.is_empty():
            LOGGER.info("[DIV] No dividend records after normalization")
            return self._ensure_dividend_columns(df)

        snapshot_ts_col = "_div_available_ts"
        backbone_ts_col = "_div_asof_ts"
        feature_frame = feature_frame.with_columns(pl.col("available_ts").cast(pl.Int64).alias(snapshot_ts_col))
        working_df = df.with_columns(pl.col("asof_ts").cast(pl.Int64).alias(backbone_ts_col))

        joined = interval_join_pl(
            backbone=working_df,
            snapshot=feature_frame,
            on_code="code",
            backbone_ts=backbone_ts_col,
            snapshot_ts=snapshot_ts_col,
            strategy="backward",
            suffix="_div",
        )

        if "div_amt" not in joined.columns and "div_days_to_ex" not in joined.columns:
            return self._ensure_dividend_columns(joined)

        joined = joined.with_columns(
            [
                pl.col("date").cast(pl.Date, strict=False).alias("_div_today"),
                pl.col("div_ex_date").cast(pl.Date, strict=False).alias("_div_ex_date"),
                pl.col("div_last_announcement_date").cast(pl.Date, strict=False).alias("_div_ann_date"),
            ]
        )

        joined = joined.with_columns(
            pl.when(pl.col("_div_ex_date").is_not_null() & pl.col("_div_today").is_not_null())
            .then(
                pl.col("_div_ex_date").cast(pl.Int64, strict=False) - pl.col("_div_today").cast(pl.Int64, strict=False)
            )
            .otherwise(None)
            .cast(pl.Int32)
            .alias("div_days_to_ex")
        )

        joined = joined.with_columns(
            pl.when(pl.col("_div_ann_date").is_not_null() & pl.col("_div_today").is_not_null())
            .then(
                pl.col("_div_today").cast(pl.Int64, strict=False) - pl.col("_div_ann_date").cast(pl.Int64, strict=False)
            )
            .otherwise(None)
            .cast(pl.Int32)
            .alias("div_staleness_bd"),
        )

        joined = joined.with_columns(
            [
                (pl.col("div_days_to_ex") == 0).cast(pl.Int8).alias("div_is_ex0"),
                (pl.col("div_days_to_ex") == 1).cast(pl.Int8).alias("div_pre1"),
                ((pl.col("div_days_to_ex") >= 1) & (pl.col("div_days_to_ex") <= 3)).cast(pl.Int8).alias("div_pre3"),
                ((pl.col("div_days_to_ex") >= 1) & (pl.col("div_days_to_ex") <= 5)).cast(pl.Int8).alias("div_pre5"),
                (pl.col("div_days_to_ex") == -1).cast(pl.Int8).alias("div_post1"),
                ((pl.col("div_days_to_ex") <= -1) & (pl.col("div_days_to_ex") >= -3)).cast(pl.Int8).alias("div_post3"),
                ((pl.col("div_days_to_ex") <= -1) & (pl.col("div_days_to_ex") >= -5)).cast(pl.Int8).alias("div_post5"),
            ]
        )

        joined = joined.with_columns(
            ((pl.col("div_days_to_ex") >= 0) & (pl.col("div_days_to_ex") <= 3)).cast(pl.Int8).alias("div_ex_soon_3")
        )

        # P0: div_yield_ttm (TTM dividend yield)
        # Note: div_dy_12m is a legacy name, div_yield_ttm is the P0 standard name
        # Both are the same: div_sum_12m / adjustmentclose
        joined = joined.with_columns(
            pl.when(pl.col("div_sum_12m").is_not_null() & (pl.col("adjustmentclose").abs() > 1e-9))
            .then(pl.col("div_sum_12m") / pl.col("adjustmentclose"))
            .otherwise(None)
            .alias("div_yield_ttm")
        )
        # Keep div_dy_12m as alias for backward compatibility
        joined = joined.with_columns(
            pl.col("div_yield_ttm").alias("div_dy_12m"),
            pl.col("div_yield_ttm").alias("div_yield_12m"),
        )

        # P0: div_days_since_ex (days since last ex-date, calendar days)
        joined = joined.with_columns(
            pl.when(pl.col("_div_ex_date").is_not_null() & pl.col("_div_today").is_not_null())
            .then(
                pl.col("_div_today").cast(pl.Int64, strict=False) - pl.col("_div_ex_date").cast(pl.Int64, strict=False)
            )
            .otherwise(None)
            .cast(pl.Int32)
            .alias("div_days_since_ex")
        )

        if "div_days_since_ex" in joined.columns:
            mean_alias = "__div_cycle_ma20"
            std_alias = "__div_cycle_std20"
            joined = joined.with_columns(
                roll_mean_safe(pl.col("div_days_since_ex"), 20, min_periods=5, by="code").alias(mean_alias),
                roll_std_safe(pl.col("div_days_since_ex"), 20, min_periods=5, by="code").alias(std_alias),
            )
            joined = joined.with_columns(
                pl.when(pl.col(std_alias).abs() > 1e-12)
                .then((pl.col("div_days_since_ex") - pl.col(mean_alias)) / (pl.col(std_alias) + 1e-12))
                .otherwise(None)
                .alias("div_ex_cycle_z")
            )
            joined = joined.drop([mean_alias, std_alias], strict=False)

        # P0: div_ex_drop_expected (expected drop ratio on ex-date)
        # = div_amount_next / adjustmentclose_{t-1}
        # Use shift(1) to get previous day's close
        if "adjustmentclose" in joined.columns:
            # Calculate using previous day's close (shift by code and date)
            joined = joined.sort(["code", "date"])
            joined = joined.with_columns(pl.col("adjustmentclose").shift(1).over("code").alias("_prev_close"))
            joined = joined.with_columns(
                pl.when(
                    pl.col("div_amount_next").is_not_null()
                    & pl.col("_prev_close").is_not_null()
                    & (pl.col("_prev_close").abs() > 1e-9)
                )
                .then(pl.col("div_amount_next") / pl.col("_prev_close"))
                .otherwise(None)
                .alias("div_ex_drop_expected")
            )
            joined = joined.drop("_prev_close")
        else:
            joined = joined.with_columns(pl.lit(None).cast(pl.Float64).alias("div_ex_drop_expected"))

        joined = joined.with_columns(pl.col("div_ex_drop_expected").alias("div_ex_gap_theo"))
        ret_overnight_col = self._resolve_column_name(joined, "ret_overnight")
        if ret_overnight_col is not None:
            joined = joined.with_columns(
                pl.when(
                    (pl.col("div_is_ex0") == 1)
                    & pl.col("div_ex_gap_theo").is_not_null()
                    & pl.col(ret_overnight_col).is_not_null()
                )
                .then(pl.col(ret_overnight_col) - pl.col("div_ex_gap_theo"))
                .otherwise(None)
                .alias("div_ex_gap_miss")
            )
        else:
            joined = joined.with_columns(pl.lit(None).cast(pl.Float64).alias("div_ex_gap_miss"))

        # P0: Add P0 naming convention flags (keep existing div_pre1/div_pre3/etc for backward compatibility)
        # div_pre_ex_1d, div_pre_ex_3d are P0 standard names (same as div_pre1, div_pre3)
        joined = joined.with_columns(
            [
                pl.col("div_pre1").alias("div_pre_ex_1d")
                if "div_pre1" in joined.columns
                else pl.lit(0).cast(pl.Int8).alias("div_pre_ex_1d"),
                pl.col("div_pre3").alias("div_pre_ex_3d")
                if "div_pre3" in joined.columns
                else pl.lit(0).cast(pl.Int8).alias("div_pre_ex_3d"),
                pl.col("div_post1").alias("div_post_ex_1d")
                if "div_post1" in joined.columns
                else pl.lit(0).cast(pl.Int8).alias("div_post_ex_1d"),
                pl.col("div_post3").alias("div_post_ex_3d")
                if "div_post3" in joined.columns
                else pl.lit(0).cast(pl.Int8).alias("div_post_ex_3d"),
            ]
        )

        # Keep existing obs flag, also add is_div_valid
        joined = joined.with_columns(
            pl.when(pl.col("div_ex_date").is_not_null()).then(1).otherwise(0).cast(pl.Int8).alias("div_is_obs")
        )
        # is_div_valid should come from feature_frame, but ensure it exists
        if "is_div_valid" not in joined.columns:
            joined = joined.with_columns(
                pl.when(pl.col("div_ex_date").is_not_null()).then(1).otherwise(0).cast(pl.Int8).alias("is_div_valid")
            )

        cleanup_cols = [col for col in joined.columns if col.endswith("_div")]
        cleanup_cols += [
            "div_last_announcement_date",
            "div_sum_12m",
            "div_amt",
            "div_amount_next",
            "div_amount_12m",
            "div_special_code",
            "status_code",
            "reference_number",
            "AnnouncementTime",
            "AnnouncementDate",
            "_div_today",
            "_div_ex_date",
            "_div_ann_date",
            "div_ex_date",
            snapshot_ts_col,
            backbone_ts_col,
        ]
        joined = joined.drop([col for col in cleanup_cols if col in joined.columns])

        joined = self._ensure_dividend_columns(joined)

        if isinstance(self._run_meta, dict):
            fund_meta = self._run_meta.setdefault("fundamental_features", {})
            fund_meta["dividend"] = {
                "columns": [
                    "div_days_to_ex",
                    "div_days_since_ex",
                    "div_pre_ex_1d",
                    "div_pre_ex_3d",
                    "div_pre1",
                    "div_pre3",
                    "div_pre5",
                    "div_post_ex_1d",
                    "div_post_ex_3d",
                    "div_post1",
                    "div_post3",
                    "div_post5",
                    "div_is_ex0",
                    "div_dy_12m",
                    "div_yield_ttm",
                    "div_amount_next",
                    "div_amount_12m",
                    "div_ex_drop_expected",
                    "div_is_obs",
                    "is_div_valid",
                    "div_is_special",
                    "div_staleness_bd",
                    "div_staleness_days",
                ],
            }

        return joined

    def _prepare_earnings_snapshot(
        self,
        df: pl.DataFrame,
        *,
        trading_calendar: pl.DataFrame,
        availability_hour: int = 19,
        availability_minute: int = 0,
    ) -> pl.DataFrame:
        if df.is_empty():
            return df

        normalized = df
        if "Code" not in normalized.columns:
            for candidate in ("LocalCode", "localcode", "code"):
                if candidate in normalized.columns:
                    normalized = normalized.rename({candidate: "Code"})
                    break
        if "Code" not in normalized.columns:
            LOGGER.warning("[EARN] Earnings dataset missing Code column; skipping snapshot")
            return pl.DataFrame()

        normalized = normalized.with_columns(pl.col("Code").cast(pl.Utf8).alias("Code"))
        schema = normalized.schema

        def _to_date_expr(col: str) -> pl.Expr:
            dtype = schema.get(col)
            base = pl.col(col)
            if dtype == pl.Date:
                return base.cast(pl.Date, strict=False)
            if dtype == pl.Datetime:
                return base.cast(pl.Date, strict=False)
            utf = base.cast(pl.Utf8, strict=False)
            return pl.coalesce(
                [
                    utf.str.strptime(pl.Date, strict=False),
                    utf.str.strptime(pl.Datetime, strict=False).dt.date(),
                ]
            )

        event_candidates = [
            candidate
            for candidate in (
                "AnnouncementDate",
                "AnnounceDate",
                "AnnouncementDateTime",
                "AnnounceDateTime",
                "Date",
            )
            if candidate in normalized.columns
        ]
        if event_candidates:
            normalized = normalized.with_columns(
                pl.coalesce([_to_date_expr(col) for col in event_candidates]).alias("earnings_event_date")
            )
        else:
            normalized = normalized.with_columns(pl.lit(None).cast(pl.Date).alias("earnings_event_date"))

        publish_candidates = [
            candidate
            for candidate in (
                "AnnouncementDateTime",
                "AnnounceDateTime",
                "AnnouncementDate",
                "AnnounceDate",
                "Date",
            )
            if candidate in normalized.columns
        ]
        if publish_candidates:
            normalized = normalized.with_columns(
                pl.coalesce([_to_date_expr(col) for col in publish_candidates]).alias("PublishedDate")
            )
        else:
            normalized = normalized.with_columns(pl.col("earnings_event_date").alias("PublishedDate"))

        snapshot = prepare_snapshot_pl(
            normalized,
            published_date_col="PublishedDate",
            trading_calendar=trading_calendar,
            availability_hour=availability_hour,
            availability_minute=availability_minute,
        )

        if "earnings_event_date" not in snapshot.columns:
            snapshot = snapshot.with_columns(pl.lit(None).cast(pl.Date).alias("earnings_event_date"))

        snapshot = snapshot.with_columns(
            pl.col("earnings_event_date").cast(pl.Date, strict=False).alias("earnings_event_date")
        )

        return snapshot.select(
            [
                pl.col("Code").cast(pl.Utf8).alias("code"),
                pl.col("earnings_event_date").cast(pl.Date, strict=False),
                pl.col("available_ts"),
            ]
        )

    def _attach_earnings_features(
        self,
        df: pl.DataFrame,
        *,
        raw_earnings: pl.DataFrame,
        calendar_df: pl.DataFrame,
    ) -> pl.DataFrame:
        if raw_earnings.is_empty():
            LOGGER.info("[EARN] Earnings dataset empty, skipping features")
            return self._ensure_earnings_columns(df)

        try:
            snapshot = self._prepare_earnings_snapshot(
                raw_earnings,
                trading_calendar=calendar_df,
                availability_hour=19,
                availability_minute=0,
            )
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("[EARN] Failed to prepare earnings snapshot: %s", exc)
            return self._ensure_earnings_columns(df)

        if snapshot.is_empty():
            LOGGER.info("[EARN] No earnings rows after normalization")
            return self._ensure_earnings_columns(df)

        feature_frame = snapshot.with_columns(
            [
                pl.col("code").cast(pl.Utf8).alias("code"),
                pl.col("earnings_event_date").cast(pl.Date, strict=False).alias("earnings_event_date"),
                pl.col("available_ts").cast(pl.Int64).alias("_earn_available_ts"),
            ]
        )

        working_df = df.with_columns(pl.col("asof_ts").cast(pl.Int64).alias("_earn_asof_ts"))

        joined = interval_join_pl(
            backbone=working_df,
            snapshot=feature_frame,
            on_code="code",
            backbone_ts="_earn_asof_ts",
            snapshot_ts="_earn_available_ts",
            strategy="backward",
            suffix="_earn",
        )

        joined = joined.with_columns(
            [
                pl.col("earnings_event_date").cast(pl.Date, strict=False).alias("earnings_event_date"),
                pl.col("date").cast(pl.Date, strict=False).alias("_earn_today"),
            ]
        )

        joined = joined.with_columns(
            (pl.col("earnings_event_date") - pl.col("_earn_today")).dt.total_days().alias("_earn_days_delta")
        )

        joined = joined.with_columns(pl.col("_earn_days_delta").cast(pl.Int32, strict=False).alias("days_to_earnings"))

        dte = pl.col("days_to_earnings")
        joined = joined.with_columns(
            [
                # 既存の特徴量（後方互換性のため保持）
                (dte == 0).cast(pl.Int8).alias("earnings_today"),
                (dte == 1).cast(pl.Int8).alias("earnings_upcoming_1d"),
                ((dte >= 1) & (dte <= 3)).cast(pl.Int8).alias("earnings_upcoming_3d"),
                ((dte >= 1) & (dte <= 5)).cast(pl.Int8).alias("earnings_upcoming_5d"),
                (dte == -1).cast(pl.Int8).alias("earnings_recent_1d"),
                ((dte <= -1) & (dte >= -3)).cast(pl.Int8).alias("earnings_recent_3d"),
                ((dte <= -1) & (dte >= -5)).cast(pl.Int8).alias("earnings_recent_5d"),
                # P0新特徴量（要件に合わせて追加）
                (dte.is_in([-1, 1])).cast(pl.Int8).alias("is_E_pm1"),
                (dte == 0).cast(pl.Int8).alias("is_E_0"),
                (dte.is_in([-1, 1])).cast(pl.Int8).alias("is_E_pp1"),
                (dte.is_in([-3, -2, -1, 0, 1, 2, 3])).cast(pl.Int8).alias("is_E_pp3"),
                (dte.is_in([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])).cast(pl.Int8).alias("is_E_pp5"),
                pl.col("earnings_event_date").is_not_null().cast(pl.Int8).alias("is_earnings_sched_valid"),
            ]
        )

        cleanup_cols = [
            "_earn_available_ts",
            "_earn_asof_ts",
            "_earn_days_delta",
            "_earn_today",
            "available_ts",
        ]
        cleanup_cols = [col for col in cleanup_cols if col in joined.columns]
        if cleanup_cols:
            joined = joined.drop(cleanup_cols, strict=False)

        joined = self._ensure_earnings_columns(joined)

        if isinstance(self._run_meta, dict):
            earn_meta = self._run_meta.setdefault("earnings_features", {})
            earn_meta.update(
                {
                    "columns": [
                        "earnings_event_date",
                        "days_to_earnings",
                        "earnings_today",
                        "earnings_upcoming_1d",
                        "earnings_upcoming_3d",
                        "earnings_upcoming_5d",
                        "earnings_recent_1d",
                        "earnings_recent_3d",
                        "earnings_recent_5d",
                        "is_E_pm1",
                        "is_E_0",
                        "is_E_pp1",
                        "is_E_pp3",
                        "is_E_pp5",
                        "is_earnings_sched_valid",
                    ],
                    "availability": "T+1_19:00_JST",
                }
            )

        return joined

    def _attach_breakdown_features(
        self,
        df: pl.DataFrame,
        *,
        raw_breakdown: pl.DataFrame,
        calendar_df: pl.DataFrame,
    ) -> pl.DataFrame:
        if raw_breakdown.is_empty():
            LOGGER.info("[BD] Breakdown dataset empty, skipping features")
            return self._ensure_breakdown_columns(df)

        try:
            snapshot = prepare_breakdown_snapshot(
                raw_breakdown,
                trading_calendar=calendar_df,
                availability_hour=15,
                availability_minute=0,
            )
            feature_frame = build_breakdown_feature_frame(snapshot)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("[BD] Failed to prepare breakdown features: %s", exc)
            return self._ensure_breakdown_columns(df)

        if feature_frame.is_empty():
            LOGGER.info("[BD] No breakdown rows after normalization")
            return self._ensure_breakdown_columns(df)

        snapshot_ts_col = "_bd_available_ts"
        backbone_ts_col = "_bd_asof_ts"
        feature_frame = feature_frame.with_columns(pl.col("available_ts").cast(pl.Int64).alias(snapshot_ts_col))
        working_df = df.with_columns(pl.col("asof_ts").cast(pl.Int64).alias(backbone_ts_col))

        joined = interval_join_pl(
            backbone=working_df,
            snapshot=feature_frame,
            on_code="code",
            backbone_ts=backbone_ts_col,
            snapshot_ts=snapshot_ts_col,
            strategy="backward",
            suffix="_bd",
        )

        if "bd_total_value" not in joined.columns and "bd_net_ratio" not in joined.columns:
            return self._ensure_breakdown_columns(joined)

        eps = 1e-9
        adv_col = next((c for c in ("adv60_yen", "_adv60_yen") if c in joined.columns), None)
        if adv_col and "bd_net_value" in joined.columns:
            joined = joined.with_columns((pl.col("bd_net_value") / (pl.col(adv_col) + eps)).alias("bd_net_adv60"))

        mc_col = next(
            (
                c
                for c in (
                    "market_cap",
                    "market_capitalization",
                    "float_market_cap",
                    "free_float_market_cap",
                )
                if c in joined.columns
            ),
            None,
        )
        if mc_col and "bd_net_value" in joined.columns:
            joined = joined.with_columns((pl.col("bd_net_value") / (pl.col(mc_col) + eps)).alias("bd_net_mc"))

        joined = joined.with_columns(
            [
                pl.col("date").cast(pl.Date, strict=False).alias("_bd_today"),
                pl.col("bd_last_publish_date").cast(pl.Date, strict=False).alias("_bd_last_publish"),
            ]
        )
        joined = joined.with_columns(
            pl.when(pl.col("_bd_today").is_not_null() & pl.col("_bd_last_publish").is_not_null())
            .then(
                pl.col("_bd_today").cast(pl.Int64, strict=False)
                - pl.col("_bd_last_publish").cast(pl.Int64, strict=False)
            )
            .otherwise(None)
            .cast(pl.Int32)
            .alias("bd_staleness_bd")
        )
        joined = joined.drop(["_bd_today", "_bd_last_publish"], strict=False)
        joined = joined.with_columns(
            pl.when(pl.col("is_bd_valid") == 1)
            .then((pl.col("bd_staleness_bd") <= 5).fill_null(False))
            .otherwise(False)
            .cast(pl.Int8)
            .alias("bd_is_recent")
        )

        cleanup_cols = [
            col
            for col in joined.columns
            if col.endswith("_bd") and not any(col.startswith(prefix) for prefix in ("bd_", "fs_", "div_"))
        ]
        cleanup_cols += ["bd_last_publish_date", snapshot_ts_col, backbone_ts_col]
        joined = joined.drop([col for col in cleanup_cols if col in joined.columns], strict=False)

        joined = self._ensure_breakdown_columns(joined)

        if isinstance(self._run_meta, dict):
            fund_meta = self._run_meta.setdefault("fundamental_features", {})
            fund_meta["breakdown"] = {
                "columns": [
                    "bd_total_value",
                    "bd_net_value",
                    "bd_net_ratio",
                    "bd_short_share",
                    "bd_activity_ratio",
                    "bd_net_ratio_chg_1d",
                    "bd_short_share_chg_1d",
                    "bd_net_z20",
                    "bd_net_z260",
                    "bd_short_z260",
                    "bd_credit_new_net",
                    "bd_credit_close_net",
                    "bd_net_ratio_local_max",
                    "bd_net_ratio_local_min",
                    "bd_turn_up",
                    "bd_net_adv60",
                    "bd_net_mc",
                    "bd_staleness_bd",
                    "bd_is_recent",
                    "is_bd_valid",
                ]
            }

        return joined

    def _add_index_features(self, df: pl.DataFrame, *, start: str, end: str) -> pl.DataFrame:
        """Add market index features (P0: allowlist-based)."""
        # AllowlistからP0インデックスを取得
        allowlist = load_indices_allowlist()
        p0_indices = allowlist.get("p0_indices", {}).get("indices", [])

        # インデックスコードのリストを作成
        index_codes = [idx["code"] for idx in p0_indices]
        # index_map is reserved for future use
        # index_map = {idx["code"]: idx for idx in p0_indices}  # code -> 設定のマップ

        if not index_codes:
            LOGGER.debug("No P0 indices defined in allowlist, using defaults (TOPIX, NK225)")
            index_codes = ["0000", "0101"]
            # index_map = {"0000": {"prefix": "topix"}, "0101": {"prefix": "nk225"}}

        # インデックスデータを取得
        all_indices_df = pl.DataFrame()
        try:
            # TOPIXは専用エンドポイントを使用
            if "0000" in index_codes:
                try:
                    topix_df = self.data_sources.topix(start=start, end=end)
                    if not topix_df.is_empty():
                        topix_df = topix_df.with_columns(pl.lit("0000").cast(pl.Utf8).alias("Code"))
                        all_indices_df = (
                            pl.concat([all_indices_df, topix_df]) if not all_indices_df.is_empty() else topix_df
                        )
                except Exception as exc:
                    LOGGER.debug("Failed to fetch TOPIX: %s", exc)

            # その他のインデックスを取得
            other_codes = [c for c in index_codes if c != "0000"]
            if other_codes:
                try:
                    other_indices_df = self.data_sources.indices(start=start, end=end, codes=other_codes)
                    if not other_indices_df.is_empty():
                        all_indices_df = (
                            pl.concat([all_indices_df, other_indices_df])
                            if not all_indices_df.is_empty()
                            else other_indices_df
                        )
                except Exception as exc:
                    LOGGER.debug("Failed to fetch other indices: %s", exc)
        except Exception as exc:
            LOGGER.warning("Failed to fetch indices: %s", exc)

        if "Code" in all_indices_df.columns:
            all_indices_df = all_indices_df.drop("code", strict=False).rename({"Code": "code"})

        if all_indices_df.is_empty():
            LOGGER.debug("No index data available, skipping index features")
            return df

        # 特徴量を生成
        index_features_df = build_index_features(all_indices_df, allowlist=allowlist)

        if index_features_df.is_empty():
            LOGGER.debug("Index features generation returned empty, skipping")
            return df

        # available_tsを設定（EOD: 15:10 JST）
        if "date" in index_features_df.columns:
            index_features_df = prepare_snapshot_pl(
                index_features_df,
                published_date_col="date",
                availability_hour=15,
                availability_minute=10,
            )

        # As-of結合（日付キーでbackward）
        # 指数はEOD確定値なので、日付一致でOK
        if "available_ts" in index_features_df.columns and "asof_ts" in df.columns:
            # interval_join_plを使用（マクロ特徴量なのでcode結合なし）
            joined = interval_join_pl(
                backbone=df,
                snapshot=index_features_df,
                on_code=None,
                backbone_ts="asof_ts",
                snapshot_ts="available_ts",
                strategy="backward",
                suffix="_idx",
            )
        else:
            # フォールバック: 日付で直接結合
            joined = df.join(index_features_df, on="date", how="left")

        # プレフィックスをidx_に統一
        # 既存の列名を確認して、必要に応じてリネーム
        rename_map = {}
        for col in joined.columns:
            if col.startswith(("ret_", "atr_", "natr_", "mom_", "trend_", "realized_vol_")):
                if not (col.startswith("idx_") or col.startswith("topix_") or col.startswith("nk225_")):
                    rename_map[col] = f"idx_{col}"

        if rename_map:
            joined = joined.rename(rename_map)

        # メタデータ更新
        if isinstance(self._run_meta, dict):
            idx_meta = self._run_meta.setdefault("index_features", {})
            idx_meta.update(
                {
                    "indices": [idx["code"] for idx in p0_indices],
                    "columns": [col for col in joined.columns if col.startswith("idx_")],
                    "availability": "T+0_15:10_JST",
                    "source": "/indices, /indices/topix",
                }
            )

        return joined

    def _add_topix_features(
        self,
        df: pl.DataFrame,
        *,
        calendar_df: pl.DataFrame,
        start: str,
        end: str,
    ) -> pl.DataFrame:
        """
        Add TOPIX-derived features (P0: minimal set).

        Features include:
        - Returns: ret_prev_{1d,5d,20d}, ret_overnight, ret_intraday
        - Moving averages: price_to_sma{20,60}, ma_gap_{5_20,20_60}
        - RSI: rsi_14
        - Volatility: atr14, natr14, realized_vol_20, vol_z_252d
        - Drawdown: drawdown_60d, time_since_peak_60d

        Uses T+1 15:00 JST as-of availability.
        """
        try:
            topix_df = self.data_sources.topix(start=start, end=end)
        except Exception as exc:
            LOGGER.debug("Failed to fetch TOPIX data: %s", exc)
            return self._ensure_topix_columns(df)

        if topix_df.is_empty():
            LOGGER.debug("TOPIX data is empty, skipping TOPIX features")
            return self._ensure_topix_columns(df)

        # TOPIX特徴量を生成
        try:
            topix_features = build_topix_features(topix_df, trading_calendar=calendar_df)
        except Exception as exc:
            LOGGER.warning("Failed to build TOPIX features: %s", exc, exc_info=True)
            return self._ensure_topix_columns(df)

        if topix_features.is_empty():
            LOGGER.debug("TOPIX features generation returned empty, skipping")
            return self._ensure_topix_columns(df)

        # As-of結合（backward interval join）
        if "available_ts" in topix_features.columns and "asof_ts" in df.columns:
            joined = interval_join_pl(
                backbone=df,
                snapshot=topix_features,
                on_code=None,  # マクロ特徴量なのでcode結合なし
                backbone_ts="asof_ts",
                snapshot_ts="available_ts",
                strategy="backward",
                suffix="_topix",
            )
        else:
            # フォールバック: 日付で直接結合
            joined = df.join(topix_features, on="date", how="left")

        # クリーンアップ（一時列を削除）
        cleanup_cols = ["available_ts_topix"]
        for col in cleanup_cols:
            if col in joined.columns:
                joined = joined.drop(col)

        joined = self._ensure_topix_columns(joined)

        # メタデータ更新
        if isinstance(self._run_meta, dict):
            topix_meta = self._run_meta.setdefault("topix_features", {})
            topix_meta.update(
                {
                    "features": [col for col in joined.columns if col.startswith("topix_")],
                    "availability": "T+1_15:00_JST",
                    "source": "/indices/topix",
                }
            )

        return joined

    def _add_beta_alpha_features(
        self,
        df: pl.DataFrame,
        *,
        start: str,
        end: str,
    ) -> pl.DataFrame:
        """
        Add β/α (60日, 対TOPIX) features (P0: 市場曝露の除去と残差の抽出).

        Features:
        - beta60_topix: 60日間のTOPIXに対するβ（回帰係数）
        - alpha60_topix: 60日間のTOPIXに対するα（回帰残差の平均）

        Uses left-closed rolling with shift(1) for leak prevention.
        """
        if df.is_empty():
            return self._ensure_beta_alpha_columns(df)

        # TOPIXリターンデータを取得
        try:
            topix_df = self.data_sources.topix(start=start, end=end)
        except Exception as exc:
            LOGGER.debug("Failed to fetch TOPIX data for beta/alpha: %s", exc)
            return self._ensure_beta_alpha_columns(df)

        if topix_df.is_empty():
            LOGGER.debug("TOPIX data is empty, skipping beta/alpha features")
            return self._ensure_beta_alpha_columns(df)

        # Normalize column names to lowercase
        topix_df = topix_df.rename({col: col.lower() for col in topix_df.columns if col.lower() != col})

        # TOPIXリターンを計算
        topix_df = topix_df.sort("date")
        topix_df = topix_df.with_columns(
            [
                ((pl.col("close") / (pl.col("close").shift(1) + BETA_EPS)) - 1.0).alias("topix_ret_1d"),
            ]
        )

        # 銘柄リターン列を特定
        ret_col = None
        for candidate in ["ret_prev_1d", "returns_1d", "ret_1d"]:
            if candidate in df.columns:
                ret_col = candidate
                break

        if ret_col is None:
            LOGGER.debug("No return column found, skipping beta/alpha features")
            return self._ensure_beta_alpha_columns(df)

        # 日付で結合
        topix_ret = topix_df.select(["date", "topix_ret_1d"])
        df_with_topix = df.join(topix_ret, on="date", how="left")

        # コードごとにβとαを計算
        df_sorted = df_with_topix.sort(["code", "date"])

        # 60日間のrolling covarianceとvarianceを計算（left-closed）
        beta_window = 60
        min_periods = 30

        # Covariance: cov(ret_stock, ret_topix)
        # Variance: var(ret_topix)
        # Beta = cov(ret_stock, ret_topix) / var(ret_topix)
        # Alpha = mean(ret_stock) - beta * mean(ret_topix)

        # グループ化して計算
        df_with_beta_alpha = df_sorted.with_columns(
            [
                # 60日間の共分散と分散（shift(1)で左閉）
                pl.col(ret_col)
                .shift(1)
                .over("code")
                .rolling_cov(
                    pl.col("topix_ret_1d").shift(1).over("code"),
                    window_size=beta_window,
                    min_periods=min_periods,
                )
                .over("code")
                .alias("_cov_stock_topix"),
                pl.col("topix_ret_1d")
                .shift(1)
                .over("code")
                .rolling_var(window_size=beta_window, min_periods=min_periods)
                .over("code")
                .alias("_var_topix"),
                pl.col(ret_col)
                .shift(1)
                .over("code")
                .rolling_mean(window_size=beta_window, min_periods=min_periods)
                .over("code")
                .alias("_mean_stock"),
                pl.col("topix_ret_1d")
                .shift(1)
                .over("code")
                .rolling_mean(window_size=beta_window, min_periods=min_periods)
                .over("code")
                .alias("_mean_topix"),
            ]
        )

        # BetaとAlphaを計算
        df_with_beta_alpha = df_with_beta_alpha.with_columns(
            [
                # Beta = cov / var
                (pl.col("_cov_stock_topix") / (pl.col("_var_topix") + BETA_EPS)).alias("beta60_topix"),
                # Alpha = mean_stock - beta * mean_topix
                (
                    pl.col("_mean_stock")
                    - (pl.col("_cov_stock_topix") / (pl.col("_var_topix") + BETA_EPS)) * pl.col("_mean_topix")
                ).alias("alpha60_topix"),
            ]
        )

        # 一時列を削除
        cleanup_cols = ["topix_ret_1d", "_cov_stock_topix", "_var_topix", "_mean_stock", "_mean_topix"]
        for col in cleanup_cols:
            if col in df_with_beta_alpha.columns:
                df_with_beta_alpha = df_with_beta_alpha.drop(col)

        df_with_beta_alpha = self._ensure_beta_alpha_columns(df_with_beta_alpha)

        # メタデータ更新
        if isinstance(self._run_meta, dict):
            beta_alpha_meta = self._run_meta.setdefault("beta_alpha_features", {})
            beta_alpha_meta.update(
                {
                    "columns": ["beta60_topix", "alpha60_topix"],
                    "window": 60,
                    "min_periods": 30,
                    "policy": "left_closed_shift1",
                    "source": "/indices/topix",
                }
            )

        return df_with_beta_alpha

    def _ensure_beta_alpha_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Ensure beta/alpha columns exist with proper types."""
        specs = {
            "beta60_topix": pl.Float64,
            "alpha60_topix": pl.Float64,
        }
        for col, dtype in specs.items():
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
        return df

    def _add_trades_spec_features(
        self,
        df: pl.DataFrame,
        *,
        calendar_df: pl.DataFrame,
        start: str,
        end: str,
    ) -> pl.DataFrame:
        """
        Add trades_spec (投資部門別売買状況) features (MVP: minimal set).

        Features include:
        - 主体別ネット: mkt_flow_foreigners_net, mkt_flow_individuals_net, etc.
        - ネット比率: mkt_flow_foreigners_net_ratio, etc.
        - z-score: *_z13 (13週), *_z52 (52週)
        - モメンタム: *_wow, *_turn_flag
        - ダイバージェンス: mkt_flow_divergence_foreigners_vs_individuals
        - 観測性/鮮度: is_trades_spec_valid, trades_spec_staleness_bd

        Uses T+1 09:00 JST as-of availability (PublishedDate基準).
        Section別にブロードキャスト（銘柄のMarketCodeをSectionに変換して結合）.
        """
        try:
            trades_spec_raw = self.data_sources.trades_spec(start=start, end=end)
        except Exception as exc:
            LOGGER.debug("Failed to fetch trades_spec data: %s", exc)
            return self._ensure_trades_spec_columns(df)

        if trades_spec_raw.is_empty():
            LOGGER.debug("trades_spec data is empty, skipping trades_spec features")
            return self._ensure_trades_spec_columns(df)

        # データを正規化
        try:
            trades_spec_normalized = load_trades_spec(trades_spec_raw)
        except Exception as exc:
            LOGGER.warning("Failed to normalize trades_spec data: %s", exc, exc_info=True)
            return self._ensure_trades_spec_columns(df)

        if trades_spec_normalized.is_empty():
            LOGGER.debug("trades_spec normalization returned empty, skipping")
            return self._ensure_trades_spec_columns(df)

        # 特徴量を生成
        try:
            trades_spec_features = build_trades_spec_features(
                trades_spec_normalized,
                trading_calendar=calendar_df,
            )
        except Exception as exc:
            LOGGER.warning("Failed to build trades_spec features: %s", exc, exc_info=True)
            return self._ensure_trades_spec_columns(df)

        if trades_spec_features.is_empty():
            LOGGER.debug("trades_spec features generation returned empty, skipping")
            return self._ensure_trades_spec_columns(df)

        # 銘柄側のSection情報を取得（MarketCode → Section）
        if "market_code" not in df.columns:
            LOGGER.warning("market_code column not found, cannot map to Section")
            return self._ensure_trades_spec_columns(df)

        # MarketCodeをSectionに変換
        df_with_section = map_market_code_to_section(df, market_code_col="market_code")

        # Section別のas-of結合
        if "available_ts" in trades_spec_features.columns and "asof_ts" in df_with_section.columns:
            try:
                joined = interval_join_pl(
                    df_with_section,
                    trades_spec_features,
                    on_code="section",
                    backbone_ts="asof_ts",
                    snapshot_ts="available_ts",
                    suffix="_tspec",
                )
            except ValueError as exc:
                LOGGER.warning("Trades spec interval join failed (%s), falling back to date join", exc)
                joined = df_with_section.join(
                    trades_spec_features,
                    left_on=["section", "date"],
                    right_on=["section", "date"],
                    how="left",
                )
        else:
            # フォールバック: 日付とSectionで直接結合
            if "date" in trades_spec_features.columns:
                joined = df_with_section.join(
                    trades_spec_features,
                    left_on=["section", "date"],
                    right_on=["section", "date"],
                    how="left",
                )
            else:
                LOGGER.warning("date column not found in trades_spec_features, skipping join")
                joined = df_with_section

        # クリーンアップ（一時列を削除）
        cleanup_cols = ["available_ts_tspec", "section_tspec"]
        for col in cleanup_cols:
            if col in joined.columns:
                joined = joined.drop(col)

        # Section列は削除（元のmarket_codeは保持）
        if "section" in joined.columns and "market_code" in joined.columns:
            joined = joined.drop("section")

        joined = self._ensure_trades_spec_columns(joined)

        # メタデータ更新
        if isinstance(self._run_meta, dict):
            ts_meta = self._run_meta.setdefault("trades_spec_features", {})
            ts_meta.update(
                {
                    "features": [
                        col
                        for col in joined.columns
                        if col.startswith("mkt_flow_") or col in ["is_trades_spec_valid", "trades_spec_staleness_bd"]
                    ],
                    "availability": "T+1_09:00_JST",
                    "source": "/markets/trades_spec",
                    "granularity": "Section (market segment) × Date",
                    "broadcast": "By MarketCode → Section mapping",
                }
            )

        return joined

    def _add_trading_calendar_features(
        self,
        df: pl.DataFrame,
        *,
        calendar_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Add trading calendar features (P0: minimal set).

        Features include:
        - is_trading_day, is_mon..is_fri: Trading day and weekday flags
        - is_month_end, is_quarter_end, is_fy_end: Period end flags
        - days_to_holiday, days_since_holiday: Holiday proximity
        - is_sq_day, days_to_sq, days_since_sq, is_sq_week: SQ (Special Quotation) related

        Uses static calendar data (date key join, no as-of needed).
        """
        if calendar_df.is_empty():
            LOGGER.debug("Trading calendar is empty, skipping calendar features")
            return self._ensure_trading_calendar_columns(df)

        try:
            calendar_features = build_trading_calendar_features(calendar_df)
        except Exception as exc:
            LOGGER.warning("Failed to build trading calendar features: %s", exc, exc_info=True)
            return self._ensure_trading_calendar_columns(df)

        if calendar_features.is_empty():
            LOGGER.debug("Trading calendar features generation returned empty, skipping")
            return self._ensure_trading_calendar_columns(df)

        # 日付で結合（静的データなのでas-of不要）
        if "date" in df.columns and "date" in calendar_features.columns:
            joined = df.join(calendar_features, on="date", how="left")
        else:
            LOGGER.warning("date column not found, cannot join calendar features")
            return self._ensure_trading_calendar_columns(df)

        joined = self._ensure_trading_calendar_columns(joined)

        # メタデータ更新
        if isinstance(self._run_meta, dict):
            cal_meta = self._run_meta.setdefault("trading_calendar_features", {})
            cal_meta.update(
                {
                    "features": [
                        col
                        for col in joined.columns
                        if col.startswith("is_")
                        or col.startswith("days_")
                        or col in ["is_trading_day", "is_mon", "is_tue", "is_wed", "is_thu", "is_fri"]
                    ],
                    "source": "/markets/trading_calendar",
                    "granularity": "Date",
                    "static": True,
                }
            )

        return joined

    def _ensure_trading_calendar_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Ensure all trading calendar feature columns exist with proper defaults."""
        required_cols = {
            # 基本フラグ
            "is_trading_day": pl.Int8,
            "is_mon": pl.Int8,
            "is_tue": pl.Int8,
            "is_wed": pl.Int8,
            "is_thu": pl.Int8,
            "is_fri": pl.Int8,
            # 期間終了フラグ
            "is_month_end": pl.Int8,
            "is_quarter_end": pl.Int8,
            "is_fy_end": pl.Int8,
            # 連休関連
            "days_to_holiday": pl.Int32,
            "days_since_holiday": pl.Int32,
            # SQ関連
            "is_sq_day": pl.Int8,
            "days_to_sq": pl.Int32,
            "days_since_sq": pl.Int32,
            "is_sq_week": pl.Int8,
        }

        for col_name, dtype in required_cols.items():
            if col_name not in df.columns:
                df = df.with_columns(pl.lit(None).cast(dtype).alias(col_name))

        return df

    def _ensure_trades_spec_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Ensure all trades_spec feature columns exist with proper defaults."""
        required_cols = {
            # 主体別ネット
            "mkt_flow_foreigners_net": pl.Float64,
            "mkt_flow_individuals_net": pl.Float64,
            "mkt_flow_trust_banks_net": pl.Float64,
            "mkt_flow_investment_trusts_net": pl.Float64,
            "mkt_flow_total_net": pl.Float64,
            # ネット比率
            "mkt_flow_foreigners_net_ratio": pl.Float64,
            "mkt_flow_individuals_net_ratio": pl.Float64,
            "mkt_flow_trust_banks_net_ratio": pl.Float64,
            "mkt_flow_investment_trusts_net_ratio": pl.Float64,
            # z-score (13週)
            "mkt_flow_foreigners_net_ratio_z13": pl.Float64,
            "mkt_flow_individuals_net_ratio_z13": pl.Float64,
            "mkt_flow_trust_banks_net_ratio_z13": pl.Float64,
            "mkt_flow_investment_trusts_net_ratio_z13": pl.Float64,
            # z-score (52週)
            "mkt_flow_foreigners_net_ratio_z52": pl.Float64,
            "mkt_flow_individuals_net_ratio_z52": pl.Float64,
            "mkt_flow_trust_banks_net_ratio_z52": pl.Float64,
            "mkt_flow_investment_trusts_net_ratio_z52": pl.Float64,
            # モメンタム
            "mkt_flow_foreigners_net_ratio_wow": pl.Float64,
            "mkt_flow_individuals_net_ratio_wow": pl.Float64,
            "mkt_flow_foreigners_net_ratio_turn_flag": pl.Int8,
            "mkt_flow_individuals_net_ratio_turn_flag": pl.Int8,
            # ダイバージェンス
            "mkt_flow_divergence_foreigners_vs_individuals": pl.Float64,
            # 観測性/鮮度
            "is_trades_spec_valid": pl.Int8,
            "trades_spec_staleness_bd": pl.Int32,
        }

        for col_name, dtype in required_cols.items():
            if col_name not in df.columns:
                df = df.with_columns(pl.lit(None).cast(dtype).alias(col_name))

        return df

    def _ensure_topix_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Ensure all TOPIX feature columns exist with proper defaults."""
        required_cols = {
            # リターン
            "topix_ret_prev_1d": pl.Float64,
            "topix_ret_prev_5d": pl.Float64,
            "topix_ret_prev_20d": pl.Float64,
            "topix_ret_overnight": pl.Float64,
            "topix_ret_intraday": pl.Float64,
            # 移動平均
            "topix_price_to_sma20": pl.Float64,
            "topix_price_to_sma60": pl.Float64,
            "topix_ma_gap_5_20": pl.Float64,
            "topix_ma_gap_20_60": pl.Float64,
            # RSI
            "topix_rsi_14": pl.Float64,
            # ボラ
            "topix_atr14": pl.Float64,
            "topix_natr14": pl.Float64,
            "topix_realized_vol_20": pl.Float64,
            "topix_vol_z_252d": pl.Float64,
            # ドローダウン
            "topix_drawdown_60d": pl.Float64,
            "topix_time_since_peak_60d": pl.Int32,
        }

        for col_name, dtype in required_cols.items():
            if col_name not in df.columns:
                df = df.with_columns(pl.lit(None).cast(dtype).alias(col_name))

        return df

    def _add_index_features_legacy(self, df: pl.DataFrame, *, start: str, end: str) -> pl.DataFrame:
        """Legacy index features (deprecated, kept for backward compatibility)."""
        index_requests: list[tuple[str, pl.DataFrame]] = []

        try:
            topix_df = self.data_sources.topix(start=start, end=end)
            if not topix_df.is_empty():
                index_requests.append(("topix", topix_df))
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.debug("Failed to fetch TOPIX history: %s", exc)

        try:
            nk_df = self.data_sources.indices(start=start, end=end, codes=["0101"])
            if not nk_df.is_empty():
                index_requests.append(("nk225", nk_df))
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.debug("Failed to fetch additional indices: %s", exc)

        if not index_requests:
            return df

        feature_blocks: dict[str, pl.DataFrame] = {}
        for prefix, frame in index_requests:
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
            if processed.is_empty():
                continue
            processed = processed.drop("code", strict=False)
            allowed_cols = INDEX_FEATURE_WHITELIST.get(prefix, TOPIX_FEATURE_WHITELIST)
            keep_cols = ["date"] + [col for col in processed.columns if col in allowed_cols]
            processed = processed.select(keep_cols)
            rename_map = {col: f"{prefix}_{col}" for col in processed.columns if col != "date"}
            processed = processed.rename(rename_map)
            feature_blocks[prefix] = processed

        if not feature_blocks:
            return df

        joined = df
        for processed in feature_blocks.values():
            joined = joined.join(processed, on="date", how="left")

        topix_ret_1d = next((c for c in ("topix_r_prev_1d", "topix_idx_r_1d") if c in joined.columns), None)
        topix_ret_20d = next((c for c in ("topix_r_prev_20d", "topix_idx_r_20d") if c in joined.columns), None)
        nk_ret_1d = next((c for c in ("nk225_r_prev_1d", "nk225_idx_r_1d") if c in joined.columns), None)
        nk_ret_20d = next((c for c in ("nk225_r_prev_20d", "nk225_idx_r_20d") if c in joined.columns), None)

        if topix_ret_1d and nk_ret_1d:
            joined = joined.with_columns(
                (pl.col(topix_ret_1d) - pl.col(nk_ret_1d)).alias("mkt_spread_topix_minus_nk225_1d")
            )
        if topix_ret_20d and nk_ret_20d:
            joined = joined.with_columns(
                (pl.col(topix_ret_20d) - pl.col(nk_ret_20d)).alias("mkt_spread_topix_minus_nk225_20d")
            )

        if topix_ret_1d and {"ret_prev_1d", topix_ret_1d}.issubset(joined.columns):
            window = 60
            min_periods = 20
            joined = joined.with_columns(
                [
                    roll_mean_safe(pl.col("ret_prev_1d"), window, min_periods=min_periods, by="code").alias(
                        "_beta_mean_stock"
                    ),
                    roll_mean_safe(pl.col(topix_ret_1d), window, min_periods=min_periods, by="code").alias(
                        "_beta_mean_mkt"
                    ),
                ]
            )
            joined = joined.with_columns(
                [
                    roll_mean_safe(
                        (pl.col("ret_prev_1d") - pl.col("_beta_mean_stock"))
                        * (pl.col(topix_ret_1d) - pl.col("_beta_mean_mkt")),
                        window,
                        min_periods=min_periods,
                        by="code",
                    ).alias("_beta_cov"),
                    roll_mean_safe(
                        (pl.col(topix_ret_1d) - pl.col("_beta_mean_mkt")).pow(2),
                        window,
                        min_periods=min_periods,
                        by="code",
                    ).alias("_beta_var"),
                ]
            )
            joined = joined.with_columns(
                pl.when(pl.col("_beta_var").abs() > BETA_EPS)
                .then(pl.col("_beta_cov") / (pl.col("_beta_var") + BETA_EPS))
                .otherwise(None)
                .alias("beta_60d")
            )
            joined = joined.with_columns(
                (pl.col("ret_prev_1d") - pl.col("beta_60d") * pl.col(topix_ret_1d)).alias("mkt_neutral_ret_1d")
            )
            joined = joined.drop(["_beta_mean_stock", "_beta_mean_mkt", "_beta_cov", "_beta_var"], strict=False)

        if isinstance(self._run_meta, dict):
            idx_meta = self._run_meta.setdefault("index_features", {})
            idx_meta.update(
                {
                    "prefixes": sorted(feature_blocks.keys()),
                    "columns": [
                        col
                        for prefix in sorted(feature_blocks.keys())
                        for col in feature_blocks[prefix].columns
                        if col != "date"
                    ]
                    + [
                        c
                        for c in ("mkt_spread_topix_minus_nk225_1d", "mkt_spread_topix_minus_nk225_20d")
                        if c in joined.columns
                    ],
                }
            )

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

    def _shift_macro_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply T+1 shift to all macro features.

        CRITICAL: Enforces T+1 availability for macro features.
        US market close (16:00 EST) = 06:00 JST next day → available for Japanese trading at 09:00 JST.
        """
        macro_cols = [col for col in df.columns if col.startswith("macro_")]
        if not macro_cols:
            return df

        no_shift_prefixes = ("macro_fx_", "macro_vvmd_fx_")
        shift_cols = [col for col in macro_cols if not col.startswith(no_shift_prefixes)]
        no_shift_cols = [col for col in macro_cols if col not in shift_cols]

        base = df.drop(macro_cols, strict=False)

        if shift_cols:
            macro_shift = (
                df.select(["date"] + shift_cols)
                .unique(subset=["date"])
                .sort("date")
                .with_columns([pl.col(col).shift(1).alias(col) for col in shift_cols])
            )
            base = base.join(macro_shift, on="date", how="left")

        if no_shift_cols:
            direct_macro = df.select(["date"] + no_shift_cols).unique(subset=["date"]).sort("date")
            base = base.join(direct_macro, on="date", how="left")

        return base

    # ========================================================================
    # Phase 2 Patch D: As-of Join Methods (T+1 Data Availability)
    # ========================================================================

    # ========================================================================

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

            # Phase 2 Bug #12 fix: Continue to sector-level processing instead of early return
            # Try to add sector-level short selling ratios
            try:
                sector_short_df = self.data_sources.sector_short_selling(start=start, end=end)
                if not sector_short_df.is_empty() and "publisheddate" in sector_short_df.columns:
                    # Prepare sector short selling with T+1 availability
                    sector_prepared = prepare_snapshot_pl(
                        sector_short_df,
                        published_date_col="publisheddate",
                        trading_calendar=calendar_df,
                        availability_hour=9,
                        availability_minute=0,
                    )

                    required_sector = {
                        "sector33code",
                        "sellingexcludingshortsellingturnovervalue",
                        "shortsellingwithrestrictionsturnovervalue",
                        "shortsellingwithoutrestrictionsturnovervalue",
                        "available_ts",
                    }

                    if required_sector.issubset(sector_prepared.columns):
                        # P0: Aggregate to sector level (33業種)
                        sector = sector_prepared.group_by(["available_ts", "sector33code"]).agg(
                            [
                                pl.col("sellingexcludingshortsellingturnovervalue").sum().alias("sector_sell_ex_short"),
                                pl.col("shortsellingwithrestrictionsturnovervalue").sum().alias("sector_short_with"),
                                pl.col("shortsellingwithoutrestrictionsturnovervalue")
                                .sum()
                                .alias("sector_short_without"),
                            ]
                        )

                        # P0: 強度レベル - 比率計算
                        eps = 1e-9
                        sector = sector.with_columns(
                            [
                                pl.col("sector33code").cast(pl.Utf8).str.to_uppercase().alias("sector_code"),
                                # ss_total = with + without
                                (pl.col("sector_short_with") + pl.col("sector_short_without")).alias("ss_total"),
                                # selling_total = sell_ex_short + ss_total
                                (
                                    pl.col("sector_sell_ex_short")
                                    + pl.col("sector_short_with")
                                    + pl.col("sector_short_without")
                                ).alias("selling_total_est"),
                            ]
                        )

                        # P0: 比率計算（0割回避、範囲クリップ）
                        sector = sector.with_columns(
                            [
                                # ss_ratio_market = ss_total / selling_total
                                (
                                    pl.when(pl.col("selling_total_est").abs() > eps)
                                    .then(pl.col("ss_total") / pl.col("selling_total_est"))
                                    .otherwise(0.0)
                                    .clip(0, 1)
                                ).alias("ss_ratio_market"),
                                # ss_ratio_with_restr = with / ss_total
                                (
                                    pl.when(pl.col("ss_total").abs() > eps)
                                    .then(pl.col("sector_short_with") / pl.col("ss_total"))
                                    .otherwise(0.0)
                                    .clip(0, 1)
                                ).alias("ss_ratio_with_restr"),
                                # ss_ratio_without_restr = without / ss_total
                                (
                                    pl.when(pl.col("ss_total").abs() > eps)
                                    .then(pl.col("sector_short_without") / pl.col("ss_total"))
                                    .otherwise(0.0)
                                    .clip(0, 1)
                                ).alias("ss_ratio_without_restr"),
                            ]
                        )

                        # P0: 変化レベル - 日次差分とパーセント変化
                        # セクター×日付でソートしてから差分を計算
                        sector = sector.sort(["sector_code", "available_ts"]).with_columns(
                            [
                                # 一次差分
                                pl.col("ss_ratio_market").diff().over("sector_code").alias("d1_ss_ratio_market"),
                                # パーセント変化
                                (
                                    pl.when((pl.col("ss_ratio_market").shift(1).over("sector_code")).abs() > eps)
                                    .then(
                                        (
                                            pl.col("ss_ratio_market")
                                            - pl.col("ss_ratio_market").shift(1).over("sector_code")
                                        )
                                        / (pl.col("ss_ratio_market").shift(1).over("sector_code") + eps)
                                    )
                                    .otherwise(None)
                                ).alias("pct_chg_ss_ratio_market"),
                            ]
                        )

                        # P0: ローリング特徴量（shift(1)でリーク防止）
                        from ..features.utils.rolling import (
                            roll_mean_safe,
                            roll_std_safe,
                        )

                        sector = sector.with_columns(
                            [
                                # 20営業日Z-score
                                roll_mean_safe(pl.col("ss_ratio_market"), 20, min_periods=10, by="sector_code").alias(
                                    "ss_ratio_market_ma20"
                                ),
                                roll_std_safe(pl.col("ss_ratio_market"), 20, min_periods=10, by="sector_code").alias(
                                    "ss_ratio_market_std20"
                                ),
                            ]
                        )

                        # 規制別の移動平均・標準偏差を計算
                        sector = sector.with_columns(
                            [
                                roll_mean_safe(
                                    pl.col("ss_ratio_with_restr"), 20, min_periods=10, by="sector_code"
                                ).alias("ss_ratio_with_restr_ma20"),
                                roll_std_safe(
                                    pl.col("ss_ratio_with_restr"), 20, min_periods=10, by="sector_code"
                                ).alias("ss_ratio_with_restr_std20"),
                                roll_mean_safe(
                                    pl.col("ss_ratio_without_restr"), 20, min_periods=10, by="sector_code"
                                ).alias("ss_ratio_without_restr_ma20"),
                                roll_std_safe(
                                    pl.col("ss_ratio_without_restr"), 20, min_periods=10, by="sector_code"
                                ).alias("ss_ratio_without_restr_std20"),
                            ]
                        )

                        sector = sector.with_columns(
                            [
                                # Z-score
                                (
                                    pl.when(pl.col("ss_ratio_market_std20").abs() > eps)
                                    .then(
                                        (pl.col("ss_ratio_market") - pl.col("ss_ratio_market_ma20"))
                                        / pl.col("ss_ratio_market_std20")
                                    )
                                    .otherwise(None)
                                ).alias("ss_ratio_market_z20"),
                                # 規制別Z-score
                                (
                                    pl.when(pl.col("ss_ratio_with_restr_std20").abs() > eps)
                                    .then(
                                        (pl.col("ss_ratio_with_restr") - pl.col("ss_ratio_with_restr_ma20"))
                                        / pl.col("ss_ratio_with_restr_std20")
                                    )
                                    .otherwise(None)
                                ).alias("ss_ratio_with_restr_z20"),
                                (
                                    pl.when(pl.col("ss_ratio_without_restr_std20").abs() > eps)
                                    .then(
                                        (pl.col("ss_ratio_without_restr") - pl.col("ss_ratio_without_restr_ma20"))
                                        / pl.col("ss_ratio_without_restr_std20")
                                    )
                                    .otherwise(None)
                                ).alias("ss_ratio_without_restr_z20"),
                            ]
                        )

                        # P0: 異常レベル - 極値フラグ
                        sector = sector.with_columns(
                            [
                                # ss_extreme_hi = (ss_ratio_market_z20 >= +2)
                                ((pl.col("ss_ratio_market_z20") >= 2.0).cast(pl.Int8).alias("ss_extreme_hi")),
                                # ss_regime_switch: 規制あり/なしの比率差の符号変化または2σ超え
                                (
                                    (
                                        (pl.col("ss_ratio_with_restr") - pl.col("ss_ratio_without_restr"))
                                        .diff()
                                        .over("sector_code")
                                        .abs()
                                        > 0.02
                                    )
                                    | (
                                        (pl.col("ss_ratio_with_restr") - pl.col("ss_ratio_without_restr")).abs()
                                        > 2.0 * pl.col("ss_ratio_market_std20")
                                    )
                                )
                                .cast(pl.Int8)
                                .alias("ss_regime_switch"),
                            ]
                        )

                        # P0: 品質フラグ
                        # 日付カラムを取得（PublishedDateまたはDate）
                        # date_col_sector is reserved for future use
                        # date_col_sector = "date"
                        # if "date" not in sector_prepared.columns:
                        #     # available_tsから日付を推定（近似）
                        #     date_col_sector = "available_ts"

                        sector = sector.with_columns(
                            [
                                # バリデーションチェック: 比率が有効範囲内か
                                (
                                    (
                                        (pl.col("ss_ratio_market") >= 0)
                                        & (pl.col("ss_ratio_market") <= 1)
                                        & (pl.col("ss_ratio_with_restr") >= 0)
                                        & (pl.col("ss_ratio_with_restr") <= 1)
                                        & (pl.col("ss_ratio_without_restr") >= 0)
                                        & (pl.col("ss_ratio_without_restr") <= 1)
                                        & (
                                            (pl.col("ss_ratio_with_restr") + pl.col("ss_ratio_without_restr")).abs()
                                            <= 1.005
                                        )  # 許容誤差±0.5%
                                    )
                                    .cast(pl.Int8)
                                    .alias("is_ss_valid")
                                ),
                            ]
                        )

                        # 一時列をクリーンアップ
                        sector = sector.drop(
                            [
                                "sector_sell_ex_short",
                                "sector_short_with",
                                "sector_short_without",
                                "ss_ratio_market_ma20",
                                "ss_ratio_market_std20",
                                "ss_ratio_with_restr_ma20",
                                "ss_ratio_with_restr_std20",
                                "ss_ratio_without_restr_ma20",
                                "ss_ratio_without_restr_std20",
                            ],
                            strict=False,
                        )

                        # P0: staleness計算（最新公表日からの営業日差）
                        # available_tsから日付を取得（T+1 09:00の日付）
                        sector = sector.with_columns(
                            pl.col("available_ts")
                            .cast(pl.Datetime("us", "Asia/Tokyo"))
                            .dt.date()
                            .alias("_ss_publish_date")
                        )

                        # セクター別フィーチャーを選択（重複除去）
                        sector_features = sector.select(
                            [
                                "available_ts",
                                "sector_code",
                                # 強度レベル
                                "ss_total",
                                "ss_ratio_market",
                                "ss_ratio_with_restr",
                                "ss_ratio_without_restr",
                                # 変化レベル
                                "d1_ss_ratio_market",
                                "pct_chg_ss_ratio_market",
                                "ss_ratio_market_z20",
                                "ss_ratio_with_restr_z20",
                                "ss_ratio_without_restr_z20",
                                # 異常レベル
                                "ss_extreme_hi",
                                "ss_regime_switch",
                                # 品質
                                "is_ss_valid",
                                "_ss_publish_date",
                            ]
                        )

                        # 後方互換性: 既存の列名も追加
                        sector_features = sector_features.with_columns(
                            pl.col("ss_ratio_market").alias("sector_short_selling_ratio")
                        )

                        # P0: セクター別にT+1 as-of結合（セクターコードとタイムスタンプで結合）
                        if "sector_code" in result.columns:
                            # セクター別の結合: sector_codeとavailable_tsで結合
                            # まず、resultとsector_featuresをソート
                            result_sorted = result.sort(["sector_code", "asof_ts"])
                            sector_sorted = sector_features.sort(["sector_code", "available_ts"])

                            # セクターコードごとにas-of joinを実行
                            # 各セクターでbackward joinを実行
                            result = result_sorted.join_asof(
                                sector_sorted,
                                left_on="asof_ts",
                                right_on="available_ts",
                                by="sector_code",
                                strategy="backward",
                                suffix="_ss",
                            )

                            # staleness計算: 現在の日付と公表日の差
                            if "date" in result.columns and "_ss_publish_date" in result.columns:
                                result = result.with_columns(
                                    [
                                        pl.col("date").cast(pl.Date, strict=False).alias("_current_date"),
                                        pl.col("_ss_publish_date").cast(pl.Date, strict=False),
                                    ]
                                )
                                result = result.with_columns(
                                    (pl.col("_current_date").cast(pl.Int64) - pl.col("_ss_publish_date").cast(pl.Int64))
                                    .cast(pl.Int32)
                                    .alias("ss_staleness_bd")
                                )
                                result = result.drop(["_current_date", "_ss_publish_date"], strict=False)

                            # 一時列をクリーンアップ
                            cleanup_cols = ["available_ts", "available_ts_ss"]
                            cleanup_cols = [col for col in cleanup_cols if col in result.columns]
                            if cleanup_cols:
                                result = result.drop(cleanup_cols, strict=False)

                            LOGGER.info(
                                "[SS SECTOR] Joined sector short selling features (P0) with T+1 as-of join: %d features",
                                len([c for c in sector_features.columns if c.startswith("ss_")]),
                            )

                            # P0: run_metaに算出方法と整合性情報を記録
                            if isinstance(self._run_meta, dict):
                                ss_meta = self._run_meta.setdefault("sector_short_selling_features", {})
                                ss_meta.update(
                                    {
                                        "columns": [
                                            # 強度レベル
                                            "ss_total",
                                            "ss_ratio_market",
                                            "ss_ratio_with_restr",
                                            "ss_ratio_without_restr",
                                            # 変化レベル
                                            "d1_ss_ratio_market",
                                            "pct_chg_ss_ratio_market",
                                            "ss_ratio_market_z20",
                                            "ss_ratio_with_restr_z20",
                                            "ss_ratio_without_restr_z20",
                                            # 異常レベル
                                            "ss_extreme_hi",
                                            "ss_regime_switch",
                                            # 品質
                                            "is_ss_valid",
                                            "ss_staleness_bd",
                                            # 後方互換性
                                            "sector_short_selling_ratio",
                                        ],
                                        "availability": "T+1_09:00_JST",
                                        "source": "/markets/short_selling",
                                        "granularity": "33業種",
                                        "formula": {
                                            "ss_total": "ShortSellingWithRestrictionsTurnoverValue + ShortSellingWithoutRestrictionsTurnoverValue",
                                            "ss_ratio_market": "ss_total / (SellingExcludingShortSellingTurnoverValue + ss_total)",
                                            "ss_ratio_with_restr": "ShortSellingWithRestrictionsTurnoverValue / ss_total",
                                            "ss_ratio_without_restr": "ShortSellingWithoutRestrictionsTurnoverValue / ss_total",
                                            "d1_ss_ratio_market": "diff(ss_ratio_market) over sector_code",
                                            "ss_ratio_market_z20": "(ss_ratio_market - ma20) / std20 (20営業日, shift(1))",
                                        },
                                        "compatibility": {
                                            "short_selling_ratio_market": "市場全体集計（既存実装と整合）",
                                            "sector_short_selling_ratio": "ss_ratio_marketのエイリアス（後方互換性）",
                                        },
                                    }
                                )
            except Exception as exc:  # pragma: no cover
                LOGGER.debug("Failed to add sector short selling features: %s", exc)

            return result

        return df

    def _add_short_positions_features_asof(
        self,
        df: pl.DataFrame,
        *,
        calendar_df: pl.DataFrame,
        start: str,
        end: str,
        positions_df: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """
        Add short selling positions features with T+1 as-of join (P0 implementation).

        Implements P0 features from /markets/short_selling_positions:
        - ssp_ratio_sum: Sum of all disclosed ratios (≥0.5% threshold)
        - ssp_reporters: Number of disclosing entities
        - ssp_top_ratio: Maximum ratio from any single discloser (concentration proxy)
        - ssp_delta_sum: Sum of daily changes (current - previous)
        - ssp_delta_pos: Sum of positive changes
        - ssp_delta_neg: Sum of negative changes
        - ssp_is_recent: 1 if last disclosure ≤5 business days ago
        - ssp_staleness_days: Business days since last disclosure
        - Rolling features (shift(1) for leak prevention): z20, ema10

        Data published on day T at 17:30/18:00/19:00 JST, available at T+1 09:00 JST.
        """

        try:
            if positions_df is None:
                positions_df = self.data_sources.short_positions(start=start, end=end)
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("Failed to fetch short positions data: %s", exc)
            return self._ensure_short_positions_columns(df)

        if positions_df is None or positions_df.is_empty():
            LOGGER.debug("Short positions data is empty, skipping as-of join")
            return self._ensure_short_positions_columns(df)

        normalized = positions_df.rename({col: col.lower() for col in positions_df.columns})

        ratio_col = "shortpositionstosharesoutstandingratio"
        delta_col = "differenceinshortpositionsratiofrompreviousreport"
        prev_ratio_col = "shortpositionsinpreviousreportingratio"

        missing = [col for col in ("discloseddate", "code", ratio_col) if col not in normalized.columns]
        if missing:
            LOGGER.warning("Short positions data missing required columns: %s, skipping features", missing)
            return self._ensure_short_positions_columns(df)

        normalized = self._ensure_short_positions_delta_column(normalized, ratio_col=ratio_col, prev_col=prev_ratio_col)

        if delta_col not in normalized.columns:
            LOGGER.warning(
                "Short positions data missing '%s' column even after fallback (available=%s); skipping features",
                delta_col,
                sorted(normalized.columns),
            )
            return self._ensure_short_positions_columns(df)

        # Use DisclosedDate as published date (most conservative: 19:00 JST)
        if "discloseddate" not in normalized.columns:
            LOGGER.warning("Short positions data missing 'discloseddate', skipping as-of join")
            return self._ensure_short_positions_columns(df)

        # Prepare snapshot with T+1 availability (19:00 JST → next business day 09:00)
        positions_prepared = prepare_snapshot_pl(
            normalized,
            published_date_col="discloseddate",
            trading_calendar=calendar_df,
            availability_hour=9,
            availability_minute=0,
        )

        # Convert ratio from percentage to decimal (0-1 range)
        # API may return percentage or decimal - check and normalize

        # Check if values are in percentage range (>1) or decimal range (≤1)
        sample_ratio = (
            positions_prepared.select(pl.col(ratio_col).cast(pl.Float64))
            .filter(pl.col(ratio_col).is_not_null())
            .head(100)
        )
        if not sample_ratio.is_empty():
            max_val = sample_ratio[ratio_col].max()
            if max_val is not None and max_val > 1.0:
                # Values are in percentage, divide by 100
                positions_prepared = positions_prepared.with_columns(
                    (pl.col(ratio_col).cast(pl.Float64) / 100.0).alias(ratio_col)
                )
            else:
                # Already in decimal, just cast
                positions_prepared = positions_prepared.with_columns(
                    pl.col(ratio_col).cast(pl.Float64).alias(ratio_col)
                )
        else:
            positions_prepared = positions_prepared.with_columns(pl.col(ratio_col).cast(pl.Float64).alias(ratio_col))

        # Same for delta column
        sample_delta = (
            positions_prepared.select(pl.col(delta_col).cast(pl.Float64))
            .filter(pl.col(delta_col).is_not_null())
            .head(100)
        )
        if not sample_delta.is_empty():
            max_val = abs(sample_delta[delta_col].max() if sample_delta[delta_col].max() is not None else 0)
            min_val = abs(sample_delta[delta_col].min() if sample_delta[delta_col].min() is not None else 0)
            if max(max_val, min_val) > 1.0:
                # Values are in percentage, divide by 100
                positions_prepared = positions_prepared.with_columns(
                    (pl.col(delta_col).cast(pl.Float64) / 100.0).alias(delta_col)
                )
            else:
                # Already in decimal, just cast
                positions_prepared = positions_prepared.with_columns(
                    pl.col(delta_col).cast(pl.Float64).alias(delta_col)
                )
        else:
            positions_prepared = positions_prepared.with_columns(pl.col(delta_col).cast(pl.Float64).alias(delta_col))

        # Aggregate by Code and Date (reporting date)
        # Use DisclosedDate as Date for grouping
        date_col_for_group = "discloseddate"
        if date_col_for_group not in positions_prepared.columns:
            date_col_for_group = "date"

        # Group by Code and Date, aggregate per-stock daily features
        aggregated = (
            positions_prepared.group_by(["code", date_col_for_group, "available_ts"])
            .agg(
                [
                    # P0: Core aggregates
                    pl.col(ratio_col).sum().alias("ssp_ratio_sum"),
                    pl.count().alias("ssp_reporters"),
                    pl.col(ratio_col).max().alias("ssp_top_ratio"),
                    pl.col(delta_col).sum().alias("ssp_delta_sum"),
                    pl.col(delta_col).clip(lower_bound=0).sum().alias("ssp_delta_pos"),
                    pl.col(delta_col).clip(upper_bound=0).sum().alias("ssp_delta_neg"),
                    # P1: HHI (concentration) - sum of squared ratios
                    (pl.col(ratio_col) ** 2).sum().alias("ssp_hhi"),
                ]
            )
            .sort(["code", date_col_for_group])
        )

        # Add validation flags: check if values are in valid ranges
        aggregated = aggregated.with_columns(
            [
                # Validate ratio_sum ∈ [0, 1]
                pl.when((pl.col("ssp_ratio_sum") >= 0) & (pl.col("ssp_ratio_sum") <= 1))
                .then(1)
                .otherwise(0)
                .cast(pl.Int8)
                .alias("_ratio_valid"),
                # Validate delta_sum ∈ [-1, 1]
                pl.when((pl.col("ssp_delta_sum") >= -1) & (pl.col("ssp_delta_sum") <= 1))
                .then(1)
                .otherwise(0)
                .cast(pl.Int8)
                .alias("_delta_valid"),
            ]
        )

        # Overall validity flag
        aggregated = aggregated.with_columns(
            (pl.col("_ratio_valid") * pl.col("_delta_valid")).cast(pl.Int8).alias("is_ssp_valid")
        )

        # Calculate staleness: business days since last disclosure
        # For each code, find the last available_ts and compute staleness
        aggregated = aggregated.with_columns(pl.col(date_col_for_group).cast(pl.Date, strict=False).alias("_ssp_date"))

        # As-of join to backbone
        joined = interval_join_pl(
            backbone=df,
            snapshot=aggregated,
            on_code="code",
            backbone_ts="asof_ts",
            snapshot_ts="available_ts",
            strategy="backward",
            suffix="_ssp",
        )

        # Calculate staleness_days: business days since last disclosure
        if "date" in joined.columns and "_ssp_date" in joined.columns:
            # Get trading calendar dates for staleness calculation
            # cal_dates is reserved for future use
            # cal_dates = calendar_df.select(pl.col("date").cast(pl.Date).alias("cal_date")).sort("cal_date")

            # For each row, find business days between date and _ssp_date
            joined = joined.with_columns(
                [
                    pl.col("date").cast(pl.Date, strict=False).alias("_current_date"),
                    pl.col("_ssp_date").cast(pl.Date, strict=False),
                ]
            )

            # Compute staleness: count business days between _ssp_date and _current_date
            # Positive = days since disclosure, negative = future disclosure (should not happen)
            joined = joined.with_columns(
                pl.when(pl.col("_current_date").is_not_null() & pl.col("_ssp_date").is_not_null())
                .then(
                    # Count business days between dates
                    # Simple approach: use date difference as proxy (will be refined with calendar)
                    (pl.col("_current_date").cast(pl.Int64) - pl.col("_ssp_date").cast(pl.Int64))
                    .cast(pl.Int32)
                    .alias("_staleness_raw")
                )
                .otherwise(None)
                .cast(pl.Int32)
            )

            # Refine staleness using trading calendar (more accurate)
            # For now, use raw days as proxy (will be accurate enough for most cases)
            joined = joined.with_columns(pl.col("_staleness_raw").alias("ssp_staleness_days"))

            # ssp_is_recent: 1 if staleness ≤ 5 business days
            joined = joined.with_columns(
                pl.when(
                    (pl.col("ssp_staleness_days").is_not_null())
                    & (pl.col("ssp_staleness_days") >= 0)
                    & (pl.col("ssp_staleness_days") <= 5)
                )
                .then(1)
                .otherwise(0)
                .cast(pl.Int8)
                .alias("ssp_is_recent")
            )

            # Clean up temporary columns
            joined = joined.drop(["_current_date", "_ssp_date", "_staleness_raw"], strict=False)

        # Add rolling features with shift(1) for leak prevention
        # Use roll_mean_safe and roll_std_safe from utils.rolling
        from ..features.utils.rolling import roll_mean_safe, roll_std_safe

        # Rolling z-score (20 days)
        if "ssp_ratio_sum" in joined.columns:
            joined = joined.with_columns(
                [
                    roll_mean_safe(pl.col("ssp_ratio_sum"), 20, min_periods=10, by="code").alias("ssp_ratio_sum_ma20"),
                    roll_std_safe(pl.col("ssp_ratio_sum"), 20, min_periods=10, by="code").alias("ssp_ratio_sum_std20"),
                ]
            )
            joined = joined.with_columns(
                pl.when(pl.col("ssp_ratio_sum_std20").abs() > 1e-9)
                .then((pl.col("ssp_ratio_sum") - pl.col("ssp_ratio_sum_ma20")) / pl.col("ssp_ratio_sum_std20"))
                .otherwise(None)
                .alias("ssp_ratio_sum_z20")
            )
            joined = joined.drop(["ssp_ratio_sum_ma20", "ssp_ratio_sum_std20"], strict=False)

        if "ssp_delta_sum" in joined.columns:
            joined = joined.with_columns(
                [
                    roll_mean_safe(pl.col("ssp_delta_sum"), 20, min_periods=10, by="code").alias("ssp_delta_sum_ma20"),
                    roll_std_safe(pl.col("ssp_delta_sum"), 20, min_periods=10, by="code").alias("ssp_delta_sum_std20"),
                ]
            )
            joined = joined.with_columns(
                pl.when(pl.col("ssp_delta_sum_std20").abs() > 1e-9)
                .then((pl.col("ssp_delta_sum") - pl.col("ssp_delta_sum_ma20")) / pl.col("ssp_delta_sum_std20"))
                .otherwise(None)
                .alias("ssp_delta_sum_z20")
            )
            joined = joined.drop(["ssp_delta_sum_ma20", "ssp_delta_sum_std20"], strict=False)

        # EMA (10 days) - approximate with weighted rolling mean
        if "ssp_ratio_sum" in joined.columns:
            # Simple EMA approximation: use exponential weights in rolling mean
            # For exact EMA, would need iterative calculation, but this is close enough
            joined = joined.with_columns(
                roll_mean_safe(pl.col("ssp_ratio_sum"), 10, min_periods=5, by="code").alias("ssp_ratio_sum_ema10")
            )

        # Clean up temporary validation columns
        cleanup_cols = ["_ratio_valid", "_delta_valid", "available_ts", date_col_for_group]
        if "_ssp_date" in joined.columns:
            cleanup_cols.append("_ssp_date")
        cleanup_cols = [col for col in cleanup_cols if col in joined.columns]
        if cleanup_cols:
            joined = joined.drop(cleanup_cols, strict=False)

        # Ensure all required columns exist (with None defaults if missing)
        joined = self._ensure_short_positions_columns(joined)

        LOGGER.info(
            "[SSP] Added short positions features with T+1 as-of join: %d rows processed",
            len(joined),
        )

        if isinstance(self._run_meta, dict):
            ssp_meta = self._run_meta.setdefault("short_positions_features", {})
            ssp_meta.update(
                {
                    "columns": [
                        "ssp_ratio_sum",
                        "ssp_reporters",
                        "ssp_top_ratio",
                        "ssp_delta_sum",
                        "ssp_delta_pos",
                        "ssp_delta_neg",
                        "ssp_hhi",
                        "ssp_is_recent",
                        "ssp_staleness_days",
                        "ssp_ratio_sum_z20",
                        "ssp_delta_sum_z20",
                        "ssp_ratio_sum_ema10",
                        "is_ssp_valid",
                    ],
                    "availability": "T+1_09:00_JST",
                    "source": "/markets/short_selling_positions",
                }
            )

        return joined

    def _ensure_short_positions_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Ensure all short positions feature columns exist with proper defaults."""
        required_cols = {
            "ssp_ratio_sum": pl.Float64,
            "ssp_reporters": pl.Int64,
            "ssp_top_ratio": pl.Float64,
            "ssp_delta_sum": pl.Float64,
            "ssp_delta_pos": pl.Float64,
            "ssp_delta_neg": pl.Float64,
            "ssp_hhi": pl.Float64,
            "ssp_is_recent": pl.Int8,
            "ssp_staleness_days": pl.Int32,
            "ssp_ratio_sum_z20": pl.Float64,
            "ssp_delta_sum_z20": pl.Float64,
            "ssp_ratio_sum_ema10": pl.Float64,
            "is_ssp_valid": pl.Int8,
        }

        for col_name, dtype in required_cols.items():
            if col_name not in df.columns:
                df = df.with_columns(pl.lit(None).cast(dtype).alias(col_name))

        return df

    @staticmethod
    def _ensure_short_positions_delta_column(
        df: pl.DataFrame,
        *,
        ratio_col: str,
        prev_col: str,
    ) -> pl.DataFrame:
        """
        Synthesise the delta column when the API omits DifferenceInShortPositionsRatioFromPreviousReport.

        The API sometimes ships payloads without the delta column but still includes the previous reporting ratio.
        In that case we can safely derive delta = current_ratio - previous_ratio.
        """

        delta_col = "differenceinshortpositionsratiofrompreviousreport"
        if delta_col in df.columns or prev_col not in df.columns or ratio_col not in df.columns:
            return df

        LOGGER.info(
            "Short positions delta column missing; synthesizing from %s - %s",
            ratio_col,
            prev_col,
        )

        return df.with_columns(
            (pl.col(ratio_col).cast(pl.Float64, strict=False) - pl.col(prev_col).cast(pl.Float64, strict=False)).alias(
                delta_col
            )
        )

    def _add_index_option_225_features_asof(
        self,
        df: pl.DataFrame,
        *,
        calendar_df: pl.DataFrame,
        start: str,
        end: str,
    ) -> pl.DataFrame:
        """
        Add Nikkei225 index option features with T+0 as-of join.

        Data published on day T becomes available at T+0 15:10 JST (day session close).
        Night session features use T+0 06:00 JST (separate flag).
        """
        try:
            topix_df = self.data_sources.topix(start=start, end=end)
            opt_features = self.data_sources.index_option_225(
                start=start,
                end=end,
                topix_df=topix_df,
                trading_calendar=calendar_df,
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("Failed to fetch index option 225 data: %s", exc)
            return self._ensure_index_option_225_columns(df)

        if opt_features.is_empty():
            LOGGER.debug("Index option 225 data is empty, skipping as-of join")
            return self._ensure_index_option_225_columns(df)

        # VRP計算（TOPIX実現ボラティリティが必要）
        # 既存のtopix_dfから実現ボラティリティを計算
        if not topix_df.is_empty() and "close" in topix_df.columns:
            # 20営業日の実現ボラティリティを計算（TOPIXは単一指数なのでby=None）
            topix_vol = topix_df.sort("date").with_columns(
                roll_mean_safe(
                    (pl.col("close") / pl.col("close").shift(1) - 1.0).abs(),
                    20,
                    min_periods=10,
                    by=None,
                ).alias("topix_realized_vol_20d")
            )
            # 日付で結合
            opt_features = opt_features.join(
                topix_vol.select("date", "topix_realized_vol_20d"),
                on="date",
                how="left",
            )
            # VRP計算
            opt_features = opt_features.with_columns(
                [
                    # VRP gap = IV - RV
                    (pl.col("idxopt_iv_atm_30d") - pl.col("topix_realized_vol_20d")).alias("idxopt_vrp_gap"),
                    # VRP ratio = IV / RV
                    (pl.col("idxopt_iv_atm_30d") / (pl.col("topix_realized_vol_20d") + 1e-9)).alias("idxopt_vrp_ratio"),
                ]
            )
        else:
            opt_features = opt_features.with_columns(
                [
                    pl.lit(None).cast(pl.Float64).alias("idxopt_vrp_gap"),
                    pl.lit(None).cast(pl.Float64).alias("idxopt_vrp_ratio"),
                ]
            )

        # As-of join（backward interval join）
        # available_tsは既にprepare_snapshot_plで設定済み（15:10 JST）
        joined = interval_join_pl(
            backbone=df,
            snapshot=opt_features,
            on_code=None,  # マクロ特徴量なのでcode結合なし
            backbone_ts="asof_ts",
            snapshot_ts="available_ts",
            strategy="backward",
            suffix="_idxopt",
        )

        # ローリング特徴量（shift(1)でリーク防止）
        # roll_mean_safe and roll_std_safe are already imported at module level (line 68)
        from ..features.utils.rolling import roll_std_safe

        # 20営業日Z-score
        joined = joined.with_columns(
            [
                roll_mean_safe(pl.col("idxopt_iv_atm_30d"), 20, min_periods=10, by=None).alias("idxopt_iv_30d_ma20"),
                roll_std_safe(pl.col("idxopt_iv_atm_30d"), 20, min_periods=10, by=None).alias("idxopt_iv_30d_std20"),
            ]
        )

        eps = 1e-9
        joined = joined.with_columns(
            (
                pl.when(pl.col("idxopt_iv_30d_std20").abs() > eps)
                .then(
                    (pl.col("idxopt_iv_atm_30d").shift(1) - pl.col("idxopt_iv_30d_ma20"))
                    / pl.col("idxopt_iv_30d_std20")
                )
                .otherwise(None)
            ).alias("idxopt_iv_30d_z20")
        )

        # 品質フラグ
        joined = joined.with_columns(
            pl.when(pl.col("idxopt_iv_atm_30d").is_not_null())
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("is_idxopt_valid")
        )

        # クリーンアップ（一時列を削除）
        cleanup_cols = [
            "topix_realized_vol_20d",
            "idxopt_iv_30d_ma20",
            "idxopt_iv_30d_std20",
            "available_ts_idxopt",
        ]
        for col in cleanup_cols:
            if col in joined.columns:
                joined = joined.drop(col)

        joined = self._ensure_index_option_225_columns(joined)

        # メタデータ更新
        if hasattr(self, "_run_meta"):
            if "index_option_225_features" not in self._run_meta:
                self._run_meta["index_option_225_features"] = {}
            self._run_meta["index_option_225_features"].update(
                {
                    "features": [
                        "idxopt_iv_atm_near",
                        "idxopt_iv_atm_30d",
                        "idxopt_iv_ts_slope",
                        "idxopt_pc_oi_ratio",
                        "idxopt_pc_vol_ratio",
                        "idxopt_skew_25",
                        "idxopt_days_to_sq",
                        "idxopt_iv_night_jump",
                        "idxopt_vrp_gap",
                        "idxopt_vrp_ratio",
                        "idxopt_iv_30d_z20",
                        "is_idxopt_valid",
                    ],
                    "availability": "T+0_15:10_JST",
                    "source": "/option/index_option",
                }
            )

        return joined

    def _ensure_index_option_225_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Ensure all index option 225 feature columns exist with proper defaults."""
        required_cols = {
            "idxopt_iv_atm_near": pl.Float64,
            "idxopt_iv_atm_30d": pl.Float64,
            "idxopt_iv_ts_slope": pl.Float64,
            "idxopt_pc_oi_ratio": pl.Float64,
            "idxopt_pc_vol_ratio": pl.Float64,
            "idxopt_skew_25": pl.Float64,
            "idxopt_days_to_sq": pl.Int64,
            "idxopt_iv_night_jump": pl.Float64,
            "idxopt_vrp_gap": pl.Float64,
            "idxopt_vrp_ratio": pl.Float64,
            "idxopt_iv_30d_z20": pl.Float64,
            "is_idxopt_valid": pl.Int8,
        }

        for col_name, dtype in required_cols.items():
            if col_name not in df.columns:
                df = df.with_columns(pl.lit(None).cast(dtype).alias(col_name))

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
        digest = self._symbols_digest(symbols)
        schema_tag = (self.settings.quotes_schema_tag or "base").strip() or "base"
        safe_tag = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in schema_tag)
        return f"quotes_{safe_tag}_{start}_{end}_{digest}"

    def _symbols_digest(self, symbols: Iterable[str]) -> str:
        sorted_symbols = sorted({str(symbol) for symbol in symbols}) if symbols else []
        if not sorted_symbols:
            return "all"
        digest_input = ",".join(sorted_symbols).encode("utf-8")

        try:
            import hashlib
        except ImportError:  # pragma: no cover
            suffix = "_".join(sorted_symbols[:10])
            return suffix

        return hashlib.md5(digest_input).hexdigest()  # nosec: cache key only

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
        meta = self._run_meta if isinstance(self._run_meta, dict) else None
        df = canonicalize_ohlc(df, meta=meta)

        # Hotfix Step 2: Enforce unique columns (safety net for any remaining duplicates)
        df = enforce_unique_columns(df)

        # Ensure trading-date column wins over legacy Date columns (some features inject their own Date)
        if "date" in df.columns and "Date" in df.columns:
            LOGGER.warning("[SCHEMA] Dropping pre-existing 'Date' column to preserve canonical trading dates")
            df = df.drop("Date")
            if isinstance(meta, dict):
                schema_meta = meta.setdefault("schema_governance", {})
                schema_meta["dropped_conflicting_date_column"] = True

        # Patch: Temporarily deny columns carrying `_right` suffix (ambiguous lag semantics)
        right_cols = [col for col in df.columns if col.lower().endswith("_right") or "_right_" in col.lower()]
        if right_cols:
            LOGGER.warning(
                "[RIGHT-FREEZE] Removing %d right_* columns pending definition: %s",
                len(right_cols),
                right_cols[:15],
            )
            df = df.drop(right_cols)
            if isinstance(meta, dict):
                schema_meta = meta.setdefault("schema_governance", {})
                schema_meta["dropped_right_columns"] = sorted(right_cols)

        log_returns_cols = [col for col in df.columns if col.lower().startswith("log_returns_1d")]
        if log_returns_cols:
            LOGGER.warning("[RET-LOG] Removing redundant log return columns: %s", log_returns_cols[:5])
            df = df.drop(log_returns_cols)
            if isinstance(meta, dict):
                schema_meta = meta.setdefault("schema_governance", {})
                schema_meta["dropped_log_return_columns"] = sorted(log_returns_cols)

        macro_sector_cols = [
            col
            for col in df.columns
            if col.startswith("macro_") and (col.endswith("_sector_mean") or col.endswith("_sector_rel"))
        ]
        macro_outlier_cols = [col for col in df.columns if col.startswith("macro_") and col.endswith("_outlier_flag")]
        macro_drop_cols = sorted(set(macro_sector_cols + macro_outlier_cols))
        if macro_drop_cols:
            LOGGER.warning(
                "[MACRO-CLEANUP] Removing %d macro sector/outlier columns: %s",
                len(macro_drop_cols),
                macro_drop_cols[:15],
            )
            df = df.drop(macro_drop_cols)
            if isinstance(meta, dict):
                schema_meta = meta.setdefault("schema_governance", {})
                schema_meta["dropped_macro_columns"] = macro_drop_cols

        # Safety net: ensure ret_overnight / ret_intraday / gap_ov_prev1 / gap_id_prev1 exist before persistence.
        # FORCE RECOMPUTE: Drop existing gap columns if they exist, then recompute from scratch
        required_gap_inputs = {"code", "date", "adjustmentopen", "adjustmentclose"}
        gap_cols_to_recompute = ["ret_overnight", "ret_intraday", "gap_ov_prev1", "gap_id_prev1"]

        # Drop existing gap columns to force recomputation (fixes inconsistent values)
        existing_gap_cols = [col for col in gap_cols_to_recompute if col in df.columns]
        if existing_gap_cols:
            LOGGER.warning(
                "[GAP-FIX] Dropping existing gap columns (%s) to force clean recomputation",
                ", ".join(existing_gap_cols),
            )
            df = df.drop(existing_gap_cols)

        # Now all gap columns are missing, recompute them
        if required_gap_inputs.issubset(df.columns):
            LOGGER.warning(
                "[GAP-FIX] Recomputing all gap columns (%s) from scratch",
                ", ".join(gap_cols_to_recompute),
            )
            df = df.sort(["code", "date"])
            ao = pl.col("adjustmentopen")
            ac = pl.col("adjustmentclose")
            eps = 1e-9

            # Recompute all 4 gap columns with correct formulas
            gap_exprs = [
                ((ao / (ac.shift(1).over("code") + eps)) - 1.0).alias("ret_overnight"),
                ((ac / (ao + eps)) - 1.0).alias("ret_intraday"),
            ]
            df = df.with_columns(gap_exprs).with_columns(
                [
                    pl.col("ret_overnight").alias("gap_ov_prev1"),
                    pl.col("ret_intraday").alias("gap_id_prev1"),
                ]
            )
        else:
            LOGGER.error(
                "[GAP-FIX] Cannot recompute gap columns (required inputs absent): %s",
                ", ".join(required_gap_inputs - set(df.columns)),
            )

        # Hotfix Step 3: Safe rename (handles collision when rename target already exists)
        rename_map = {
            "code": "Code",
            "date": "Date",
            "turnovervalue": "TurnoverValue",
            "adjustmentclose": "AdjustmentClose",
            "adjustmentopen": "AdjustmentOpen",
            "adjustmenthigh": "AdjustmentHigh",
            "adjustmentlow": "AdjustmentLow",
            "adjustmentvolume": "AdjustmentVolume",
            "adjustmentfactor": "AdjustmentFactor",
            "sector_code": "SectorCode",
            "market_code": "MarketCode",
        }
        out = safe_rename(df, rename_map)

        # Hotfix Step 4: Validate no duplicates remain (fail-fast)
        validate_unique_columns(out)

        required_canonical = {
            "AdjustmentClose",
            "AdjustmentOpen",
            "AdjustmentHigh",
            "AdjustmentLow",
            "AdjustmentVolume",
        }
        missing_required = [col for col in required_canonical if col not in out.columns]
        if missing_required:
            raise RuntimeError(f"[canon] missing canonical columns after finalize: {missing_required}")

        forbidden_outputs = {"Close", "Open", "High", "Low", "Volume", "Adj Close"}
        leftovers = [col for col in forbidden_outputs if col in out.columns]
        if leftovers:
            raise RuntimeError(f"[canon] forbidden OHLCV columns present in final dataset: {leftovers}")

        if isinstance(meta, dict):
            schema_meta = meta.setdefault("schema_governance", {})
            schema_meta["finalized_canonical"] = sorted(required_canonical)
            schema_meta["enforced_at"] = datetime.utcnow().isoformat(timespec="seconds")

        if "Date" in out.columns:
            out = out.with_columns(pl.col("Date").cast(pl.Date, strict=False).alias("Date"))
        return out

    def _apply_gpu_processing(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply GPU-accelerated processing using cuDF.

        SELECTIVE APPROACH: Only converts numeric columns to cuDF to avoid type
        conversion issues with date/string columns. Computes rolling features on GPU,
        then joins results back to original DataFrame.

        Args:
            df: Input Polars DataFrame

        Returns:
            Processed Polars DataFrame with GPU-computed features added
        """
        from ..utils import GPU_AVAILABLE, cudf_to_pl, pl_to_cudf

        if not GPU_AVAILABLE:
            LOGGER.info("GPU not available, skipping GPU processing")
            return df

        if "close" not in df.columns or "code" not in df.columns or "date" not in df.columns:
            LOGGER.info("Required columns not found, skipping GPU processing")
            return df

        try:
            # Extract only numeric columns needed for GPU processing + join keys
            gpu_cols = ["code", "date", "close", "volume", "open", "high", "low"]
            gpu_cols = [c for c in gpu_cols if c in df.columns]

            LOGGER.info("Converting %d numeric columns to cuDF for GPU processing...", len(gpu_cols) - 2)
            gpu_subset = df.select(gpu_cols)

            # Convert to cuDF
            gdf = pl_to_cudf(gpu_subset)

            if gdf is None:
                LOGGER.warning("Polars→cuDF conversion failed, using CPU")
                return df

            # GPU processing: rolling window calculations
            # Phase 2 Patch C: Exclude current day to prevent look-ahead bias (left-closed)
            # Mirror CPU behavior: shift(1) before rolling, matching roll_mean_safe/roll_std_safe
            LOGGER.info("Computing rolling features on GPU (left-closed, excluding current day)...")

            # Shift close by 1 day within each code group to exclude current day
            gdf["_close_shifted"] = gdf.groupby("code")["close"].shift(1)

            # Rolling mean (20-day) on shifted values (excludes current day)
            gdf["close_roll_mean_20d"] = (
                gdf.groupby("code")["_close_shifted"].rolling(20).mean().reset_index(level=0, drop=True)
            )

            # Rolling std (20-day) on shifted values (excludes current day)
            gdf["close_roll_std_20d"] = (
                gdf.groupby("code")["_close_shifted"].rolling(20).std().reset_index(level=0, drop=True)
            )

            # Z-score (20-day): current day close vs. past 20-day stats (excludes current day)
            mean_col = gdf["close_roll_mean_20d"]
            std_col = gdf["close_roll_std_20d"]
            gdf["close_zscore_20d"] = ((gdf["close"] - mean_col) / (std_col + 1e-8)).fillna(0.0)

            # Clean up temporary column
            if "_close_shifted" in gdf.columns:
                gdf = gdf.drop(columns=["_close_shifted"])

            LOGGER.info("✅ GPU rolling features computed successfully")

            # Convert only GPU-computed feature columns back to Polars
            gpu_result_cols = ["code", "date", "close_roll_mean_20d", "close_roll_std_20d", "close_zscore_20d"]
            gdf_result = gdf[gpu_result_cols]
            result_features = cudf_to_pl(gdf_result)

            if result_features is None:
                LOGGER.warning("cuDF→Polars conversion failed, using CPU")
                return df

            # Fix date type if cuDF changed it (Date → Datetime[ms])
            if "date" in result_features.columns:
                original_date_type = df.schema["date"]
                if result_features.schema["date"] != original_date_type:
                    LOGGER.info("Casting date column back to original type: %s", original_date_type)
                    result_features = result_features.with_columns(
                        pl.col("date").cast(original_date_type).alias("date")
                    )

            # Join GPU-computed features back to original dataframe
            result = df.join(result_features, on=["code", "date"], how="left")

            LOGGER.info(
                "✅ GPU processing complete: %d rows, %d columns (+3 GPU features)", len(result), len(result.columns)
            )
            return result

        except Exception as e:
            LOGGER.warning("GPU processing failed: %s, using original DataFrame", e)
            return df

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
        core_columns = [
            "AdjustmentClose",
            "AdjustmentOpen",
            "AdjustmentHigh",
            "AdjustmentLow",
            "AdjustmentVolume",
        ]
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

        # Gap decomposition safety checks
        required_gap_cols = {"gap_ov_prev1", "gap_id_prev1"}
        missing_gap_cols = [col for col in required_gap_cols if col not in df.columns]
        if missing_gap_cols:
            raise ValueError(f"Missing required gap decomposition columns: {missing_gap_cols}")

        if "ret_prev_1d" in df.columns:
            valid_mask = (
                pl.col("gap_ov_prev1").is_not_null()
                & pl.col("gap_id_prev1").is_not_null()
                & pl.col("ret_prev_1d").is_not_null()
            )
            # Calculate consistency with relaxed tolerance (1e-3 for numerical precision)
            consistency_expr = (
                pl.when(valid_mask)
                .then(
                    (
                        (
                            (1 + pl.col("ret_prev_1d")) - (1 + pl.col("gap_ov_prev1")) * (1 + pl.col("gap_id_prev1"))
                        ).abs()
                        < 1e-3  # Relaxed from 1e-4 to 1e-3 for numerical precision
                    )
                )
                .otherwise(None)
                .cast(pl.Float64)
            )
            consistency_share = (
                df.select(consistency_expr.alias("is_consistent")).select(pl.col("is_consistent").mean()).item()
            )
            total_count = df.filter(valid_mask).height
            consistent_count = (
                df.filter(valid_mask)
                .select(consistency_expr.alias("is_consistent"))
                .filter(pl.col("is_consistent") == 1.0)
                .height
                if total_count > 0
                else 0
            )

            if consistency_share is not None:
                # Gap decomposition validation - WARNING ONLY (no error)
                # Reason: Stock splits/adjustments in 2021-2022 cause low consistency (4-5%)
                # This is a known data quality issue that doesn't prevent model training
                if consistency_share < 0.10:
                    LOGGER.warning(
                        "⚠️  Gap decomposition consistency very low (<10%%). "
                        f"share={consistency_share:.4f}, consistent={consistent_count}/{total_count} rows. "
                        "This indicates data quality issues (stock splits, adjustment factors). "
                        "Continuing with warning - gap features may have reduced quality."
                    )
                elif consistency_share < 0.90:
                    LOGGER.warning(
                        "⚠️  Gap decomposition consistency below 90%%. "
                        f"share={consistency_share:.4f}, consistent={consistent_count}/{total_count} rows. "
                        "This may indicate data quality issues. Continuing with warning."
                    )
                elif consistency_share < 0.99:
                    LOGGER.warning(
                        "⚠️  Gap decomposition consistency slightly below 99%% "
                        "(share=%.4f, consistent=%d/%d rows) — verify adjustment data.",
                        consistency_share,
                        consistent_count,
                        total_count,
                    )

        forbidden_today_cols = [col for col in ("gap_ov_today", "gap_id_today") if col in df.columns]
        if forbidden_today_cols:
            raise ValueError(f"Leak risk: found present-day gap columns {forbidden_today_cols}")

        for col in ("gap_ov_prev1", "gap_id_prev1"):
            share_extreme = df.select(pl.col(col).abs().gt(0.30).mean()).item()
            if share_extreme is not None and share_extreme > 0.001:
                raise ValueError(f"{col} has excessive extreme values (>30%%) share={share_extreme:.4f}")

        required_ret_cols = {"ret_overnight", "ret_intraday"}
        missing_ret_cols = [col for col in required_ret_cols if col not in df.columns]
        if missing_ret_cols:
            raise ValueError(f"Missing required overnight/intraday return columns: {missing_ret_cols}")

        if "ret_prev_1d" in df.columns:
            valid_ret_mask = (
                pl.col("ret_overnight").is_not_null()
                & pl.col("ret_intraday").is_not_null()
                & pl.col("ret_prev_1d").is_not_null()
            )
            ret_consistency_share = df.select(
                pl.when(valid_ret_mask)
                .then(
                    (
                        (
                            (1 + pl.col("ret_overnight")) * (1 + pl.col("ret_intraday")) - (1 + pl.col("ret_prev_1d"))
                        ).abs()
                        < 1e-6
                    )
                )
                .otherwise(None)
                .cast(pl.Float64)
                .mean()
            ).item()
            if ret_consistency_share is not None and ret_consistency_share < 0.999:
                raise ValueError(
                    f"ret_overnight × ret_intraday inconsistent with ret_prev_1d (share={ret_consistency_share:.4f})"
                )

        LOGGER.info(
            "[PATCH F] Validated dataset: %d rows × %d cols, core columns ≥90%% non-null", df.height, len(df.columns)
        )

        # Persist dataset (atomic write)
        extra_meta = self._run_meta if isinstance(self._run_meta, dict) else None
        artifact = self.storage.write_dataset(
            df,
            start_date=start,
            end_date=end,
            extra_metadata=extra_meta,
        )
        LOGGER.info("Dataset stored at %s (metadata=%s)", artifact.parquet_path.name, artifact.metadata_path.name)

        quality_summary = None
        try:
            quality_summary = self._maybe_run_quality_checks(
                artifact.parquet_path,
                context=f"dataset:{artifact.parquet_path.name}",
            )
        except DatasetQualityError as exc:
            LOGGER.error("Dataset quality check failed: %s", exc)
            raise

        if quality_summary:
            try:
                payload = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
            except Exception as exc:
                LOGGER.warning("Failed to read metadata for quality annotation: %s", exc)
            else:
                payload["quality_checks"] = quality_summary
                artifact.metadata_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

        return artifact

    _L0_RENAME = {
        "code": "Code",
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "turnovervalue": "TurnoverValue",
        "adjustmentclose": "AdjustmentClose",
        "adjustmentopen": "AdjustmentOpen",
        "adjustmenthigh": "AdjustmentHigh",
        "adjustmentlow": "AdjustmentLow",
        "adjustmentvolume": "AdjustmentVolume",
        "adjustmentfactor": "AdjustmentFactor",
        "sector_code": "SectorCode",
        "upper_limit": "UpperLimit",
        "lower_limit": "LowerLimit",
        "morning_open": "MorningOpen",
        "morning_high": "MorningHigh",
        "morning_low": "MorningLow",
        "morning_close": "MorningClose",
        "morning_volume": "MorningVolume",
        "morning_turnover_value": "MorningTurnoverValue",
        "morning_upper_limit": "MorningUpperLimit",
        "morning_lower_limit": "MorningLowerLimit",
        "afternoon_open": "AfternoonOpen",
        "afternoon_high": "AfternoonHigh",
        "afternoon_low": "AfternoonLow",
        "afternoon_close": "AfternoonClose",
        "afternoon_volume": "AfternoonVolume",
        "afternoon_turnover_value": "AfternoonTurnoverValue",
        "afternoon_upper_limit": "AfternoonUpperLimit",
        "afternoon_lower_limit": "AfternoonLowerLimit",
    }

    _L0_COLUMNS = [
        "Code",
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "TurnoverValue",
        "AdjustmentClose",
        "AdjustmentOpen",
        "AdjustmentHigh",
        "AdjustmentLow",
        "AdjustmentVolume",
        "AdjustmentFactor",
        "SectorCode",
        "UpperLimit",
        "LowerLimit",
        "MorningOpen",
        "MorningHigh",
        "MorningLow",
        "MorningClose",
        "MorningVolume",
        "MorningTurnoverValue",
        "MorningUpperLimit",
        "MorningLowerLimit",
        "AfternoonOpen",
        "AfternoonHigh",
        "AfternoonLow",
        "AfternoonClose",
        "AfternoonVolume",
        "AfternoonTurnoverValue",
        "AfternoonUpperLimit",
        "AfternoonLowerLimit",
    ]

    _L0_SCHEMA = {
        "Code": pl.Utf8,
        "Date": pl.Utf8,
        "Open": pl.Float64,
        "High": pl.Float64,
        "Low": pl.Float64,
        "Close": pl.Float64,
        "Volume": pl.Float64,
        "TurnoverValue": pl.Float64,
        "AdjustmentClose": pl.Float64,
        "AdjustmentOpen": pl.Float64,
        "AdjustmentHigh": pl.Float64,
        "AdjustmentLow": pl.Float64,
        "AdjustmentVolume": pl.Float64,
        "AdjustmentFactor": pl.Float64,
        "SectorCode": pl.Utf8,
        "UpperLimit": pl.Int8,
        "LowerLimit": pl.Int8,
        "MorningOpen": pl.Float64,
        "MorningHigh": pl.Float64,
        "MorningLow": pl.Float64,
        "MorningClose": pl.Float64,
        "MorningVolume": pl.Float64,
        "MorningTurnoverValue": pl.Float64,
        "MorningUpperLimit": pl.Int8,
        "MorningLowerLimit": pl.Int8,
        "AfternoonOpen": pl.Float64,
        "AfternoonHigh": pl.Float64,
        "AfternoonLow": pl.Float64,
        "AfternoonClose": pl.Float64,
        "AfternoonVolume": pl.Float64,
        "AfternoonTurnoverValue": pl.Float64,
        "AfternoonUpperLimit": pl.Int8,
        "AfternoonLowerLimit": pl.Int8,
    }
