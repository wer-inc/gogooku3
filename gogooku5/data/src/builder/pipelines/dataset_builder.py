"""High-level orchestration for dataset creation."""
from __future__ import annotations

from dataclasses import dataclass, field
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
from ..utils import (
    CacheManager,
    DatasetArtifact,
    StorageClient,
    business_date_range,
    configure_logger,
)

LOGGER = configure_logger("builder.pipeline")


@dataclass
class DatasetBuilder:
    """Coordinate data fetching, feature engineering, and parquet export."""

    settings: DatasetBuilderSettings = field(default_factory=get_settings)
    fetcher: JQuantsFetcher = field(default_factory=JQuantsFetcher)
    cache: CacheManager = field(default_factory=CacheManager)
    storage: StorageClient = field(default_factory=StorageClient)
    data_sources: DataSourceManager = field(default_factory=DataSourceManager)
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

        LOGGER.info("Starting dataset build from %s to %s", start, end)
        listed_manager = ListedManager(fetcher=self.fetcher)
        listed = listed_manager.refresh() if refresh_listed else listed_manager.listed()
        if not listed:
            listed = listed_manager.refresh()
        decider = AxisDecider.from_listed_symbols(listed)
        symbols = decider.choose_symbols()

        listed_df = self._prepare_listed_dataframe(listed)
        if symbols:
            listed_df = listed_df.filter(pl.col("code").is_in(symbols))
        else:
            LOGGER.warning("No listed symbols resolved; dataset will be empty for %s-%s", start, end)

        calendar_df = self._business_calendar(start=start, end=end)

        cache_key = f"quotes_{start}_{end}"
        quotes_df = self.cache.load_dataframe(cache_key)
        if quotes_df is None:
            LOGGER.debug("Cache miss for key %s; fetching quotes", cache_key)
            quotes_payload = self._fetch_quotes(symbols, start=start, end=end)
            quotes_df = self._format_quotes(quotes_payload)
            self.cache.save_dataframe(cache_key, quotes_df)
            cache_index = self.cache.load_index()
            cache_index[cache_key] = {"start": start, "end": end, "rows": quotes_df.height}
            self.cache.save_index(cache_index)
        else:
            LOGGER.info("Cache hit for key %s", cache_key)

        aligned_quotes = self._align_quotes_with_calendar(quotes_df, calendar_df, listed_df)

        margin_df = self._fetch_margin_data(start=start, end=end)
        combined_df = self._join_margin_data(aligned_quotes, margin_df)
        combined_df = self.sector_features.add_features(combined_df)
        combined_df = self.peer_features.add_features(combined_df)

        flow_df = self.data_sources.trades_spec(start=start, end=end)
        combined_df = self.flow_features.add_features(combined_df, flow_df)

        vix_features = self.data_sources.macro_vix(start=start, end=end)
        combined_df = self.macro_features.add_vix(combined_df, vix_features)

        combined_df = self.volatility_features.add_features(combined_df)
        combined_df = self.graph_features.add_features(combined_df)
        combined_df = self.advanced_features.add_features(combined_df)
        combined_df = self.technical_features.add_features(combined_df)
        index_enriched = self.index_features.build_features(combined_df)
        enriched_df = self.quality_features.generate_quality_features(index_enriched)
        finalized = self._finalize_for_output(enriched_df)
        artifact = self._persist_dataset(finalized, start=start, end=end)
        self.storage.ensure_remote_symlink(target=str(artifact.latest_symlink))
        return artifact.latest_symlink

    def _fetch_quotes(self, codes: Iterable[str], *, start: str, end: str) -> List[dict[str, str]]:
        codes_list = list(codes)
        if not codes_list:
            LOGGER.warning("No symbols provided for quote fetch between %s and %s", start, end)
            return []
        fetcher = QuotesFetcher(client=self.fetcher)
        return fetcher.fetch_batch(codes=codes_list, start=start, end=end)

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

        numeric_cols = [col for col in ["close", "open", "high", "low", "volume"] if col in df.columns]
        if numeric_cols:
            df = df.with_columns([pl.col(col).cast(pl.Float64, strict=False).alias(col) for col in numeric_cols])

        date_dtype = df.schema.get("date")
        if isinstance(date_dtype, PlDatetimeType):
            df = df.with_columns(pl.col("date").dt.date().alias("date"))
        elif isinstance(date_dtype, PlDateType):
            pass
        else:
            df = df.with_columns(pl.col("date").str.strptime(pl.Date, strict=False).alias("date"))

        columns = ["code", "sector_code", "date", "close", "open", "high", "low", "volume"]
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
        rename_map = {col: col.lower() for col in df.columns}
        df = df.rename(rename_map)
        if "marketcode" in df.columns and "market_code" not in df.columns:
            df = df.rename({"marketcode": "market_code"})
        if "section" in df.columns and "sector_code" not in df.columns:
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
            days = business_date_range(start, end)
            df = pl.DataFrame({"date": days})
            return df.with_columns(pl.col("date").str.strptime(pl.Date, strict=False))

        calendar_df, _ = self.cache.get_or_fetch_dataframe(
            cache_key,
            _fetch,
            ttl_days=self.settings.calendar_cache_ttl_days,
        )
        return calendar_df.select("date").unique().sort("date")

    def _align_quotes_with_calendar(
        self,
        quotes: pl.DataFrame,
        calendar: pl.DataFrame,
        listed: pl.DataFrame,
    ) -> pl.DataFrame:
        if listed.is_empty():
            if quotes.is_empty():
                listed = pl.DataFrame(
                    {
                        "code": pl.Series([], dtype=pl.Utf8),
                        "sector_code": pl.Series([], dtype=pl.Utf8),
                        "market_code": pl.Series([], dtype=pl.Utf8),
                    }
                )
            else:
                listed = (
                    quotes.select("code")
                    .unique()
                    .with_columns(
                        pl.lit("UNKNOWN").alias("sector_code"),
                        pl.lit(None).cast(pl.Utf8).alias("market_code"),
                    )
                )

        base = listed.rename({"sector_code": "sector_code_listed"})
        base = base.with_columns(
            pl.col("sector_code_listed").fill_null("UNKNOWN"),
            pl.col("market_code").cast(pl.Utf8, strict=False),
        )

        if not quotes.is_empty():
            quote_dates = quotes.select("date").unique()
            calendar = (
                pl.concat([calendar, quote_dates], how="diagonal_relaxed")
                .unique(subset=["date"])
                .sort("date")
            )

        grid = base.join(calendar, how="cross")

        if quotes.is_empty():
            aligned = grid
        else:
            aligned = grid.join(quotes, on=["code", "date"], how="left")

        if "sector_code" in aligned.columns:
            aligned = aligned.with_columns(
                pl.when(pl.col("sector_code").is_null() | (pl.col("sector_code") == ""))
                .then(pl.col("sector_code_listed"))
                .otherwise(pl.col("sector_code"))
                .alias("sector_code")
            )
        else:
            aligned = aligned.with_columns(pl.col("sector_code_listed").alias("sector_code"))

        if "sector_code_listed" in aligned.columns:
            aligned = aligned.drop("sector_code_listed")

        return aligned.sort(["code", "date"])

    def _fetch_margin_data(self, *, start: str, end: str) -> pl.DataFrame:
        df = self.data_sources.margin_daily(start=start, end=end)
        return self.margin_features.build_features(df)

    def _join_margin_data(self, quotes: pl.DataFrame, margin: pl.DataFrame) -> pl.DataFrame:
        if margin.is_empty():
            return quotes
        return quotes.join(margin, on=["code", "date"], how="left")

    def _finalize_for_output(self, df: pl.DataFrame) -> pl.DataFrame:
        """Normalize schema before persistence."""

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
        present_map = {k: v for k, v in rename_map.items() if k in df.columns}
        out = df.rename(present_map) if present_map else df
        if "Date" in out.columns:
            out = out.with_columns(pl.col("Date").cast(pl.Date, strict=False).alias("Date"))
        return out

    def _persist_dataset(self, df: pl.DataFrame, *, start: str, end: str) -> DatasetArtifact:
        LOGGER.info("Persisting dataset artifacts (rows=%d, cols=%d)", df.height, len(df.columns))
        artifact = self.storage.write_dataset(df, start_date=start, end_date=end)
        LOGGER.info("Dataset stored at %s (metadata=%s)", artifact.parquet_path.name, artifact.metadata_path.name)
        return artifact
