"""High-level orchestration for dataset creation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

import polars as pl
from polars.datatypes import Date as PlDateType
from polars.datatypes import Datetime as PlDatetimeType
from requests import HTTPError

from ..api import AxisDecider, JQuantsFetcher, ListedManager, QuotesFetcher
from ..config import DatasetBuilderSettings, get_settings
from ..features.core.advanced import AdvancedFeatureEngineer
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
from ..utils import CacheManager, StorageClient, configure_logger, date_range

LOGGER = configure_logger("builder.pipeline")


@dataclass
class DatasetBuilder:
    """Coordinate data fetching, feature engineering, and parquet export."""

    settings: DatasetBuilderSettings = field(default_factory=get_settings)
    fetcher: JQuantsFetcher = field(default_factory=JQuantsFetcher)
    cache: CacheManager = field(default_factory=CacheManager)
    storage: StorageClient = field(default_factory=StorageClient)
    quality_features: QualityFinancialFeaturesGeneratorPolars = field(
        default_factory=QualityFinancialFeaturesGeneratorPolars
    )
    index_features: IndexFeatureEngineer = field(default_factory=IndexFeatureEngineer)
    margin_features: MarginDailyFeatureEngineer = field(default_factory=MarginDailyFeatureEngineer)
    sector_features: SectorAggregationFeatures = field(default_factory=SectorAggregationFeatures)
    peer_features: PeerFeatureEngineer = field(default_factory=PeerFeatureEngineer)
    volatility_features: AdvancedVolatilityFeatures = field(default_factory=AdvancedVolatilityFeatures)
    graph_features: GraphFeatureEngineer = field(default_factory=GraphFeatureEngineer)
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

        cache_key = f"quotes_{start}_{end}"
        df = self.cache.load_dataframe(cache_key)
        if df is None:
            LOGGER.debug("Cache miss for key %s; fetching quotes", cache_key)
            quotes = self._fetch_quotes(decider.choose_symbols(limit=50), start=start, end=end)
            df = self._format_quotes(quotes)
            self.cache.save_dataframe(cache_key, df)
            cache_index = self.cache.load_index()
            cache_index[cache_key] = {"start": start, "end": end, "rows": df.height}
            self.cache.save_index(cache_index)
        else:
            LOGGER.info("Cache hit for key %s", cache_key)

        margin_df = self._fetch_margin_data(start=start, end=end)
        combined_df = self._join_margin_data(df, margin_df)
        combined_df = self.sector_features.add_features(combined_df)
        combined_df = self.peer_features.add_features(combined_df)

        combined_df = self.volatility_features.add_features(combined_df)
        combined_df = self.graph_features.add_features(combined_df)
        combined_df = self.advanced_features.add_features(combined_df)
        combined_df = self.technical_features.add_features(combined_df)
        index_enriched = self.index_features.build_features(combined_df)
        enriched_df = self.quality_features.generate_quality_features(index_enriched)
        output_path = self._write_dataset(enriched_df)
        self.storage.ensure_remote_symlink(target=str(output_path))
        return output_path

    def _fetch_quotes(self, codes: Iterable[str], *, start: str, end: str) -> List[dict[str, str]]:
        fetcher = QuotesFetcher(client=self.fetcher)
        return fetcher.fetch_batch(codes=codes, start=start, end=end)

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

    def _fetch_margin_data(self, *, start: str, end: str) -> pl.DataFrame:
        dates = date_range(start, end)
        try:
            records = self.fetcher.fetch_margin_daily_window(dates=dates)
        except HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            if status == 403:
                LOGGER.warning("Margin API access forbidden (status=403); continuing without margin data.")
                records = []
            else:
                raise
        if not records:
            LOGGER.info("No margin data returned for %s-%s", start, end)
            return pl.DataFrame(
                {
                    "code": [],
                    "date": [],
                    "margin_balance": [],
                    "short_balance": [],
                }
            )
        df = pl.DataFrame(records)
        rename_map = {col: col.lower() for col in df.columns}
        df = df.rename(rename_map)

        date_dtype = df.schema.get("date")
        if isinstance(date_dtype, PlDatetimeType):
            df = df.with_columns(pl.col("date").dt.date().alias("date"))
        elif isinstance(date_dtype, PlDateType):
            pass
        else:
            df = df.with_columns(pl.col("date").str.strptime(pl.Date, strict=False).alias("date"))

        numeric_cols = [col for col in ["margin_balance", "short_balance"] if col in df.columns]
        if numeric_cols:
            df = df.with_columns([pl.col(col).cast(pl.Float64, strict=False).alias(col) for col in numeric_cols])
        df = self.margin_features.build_features(df)
        return df

    def _join_margin_data(self, quotes: pl.DataFrame, margin: pl.DataFrame) -> pl.DataFrame:
        if margin.is_empty():
            return quotes
        return quotes.join(margin, on=["code", "date"], how="left")

    def _write_dataset(self, df: pl.DataFrame) -> Path:
        output_dir = self.settings.data_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        version_path = output_dir / "ml_dataset_latest.parquet"
        df.write_parquet(version_path)
        LOGGER.info("Wrote dataset with %d rows to %s", df.height, version_path)
        return version_path
