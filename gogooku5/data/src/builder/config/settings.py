"""Configuration objects for the dataset builder."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Final

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatasetBuilderSettings(BaseSettings):
    """Central configuration container for dataset generation."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    jquants_auth_email: str = Field(..., env="JQUANTS_AUTH_EMAIL")
    jquants_auth_password: str = Field(..., env="JQUANTS_AUTH_PASSWORD")
    jquants_plan_tier: str = Field("standard", env="JQUANTS_PLAN_TIER")

    data_output_dir: Path = Field(default_factory=lambda: Path("output"), env="DATA_OUTPUT_DIR")
    data_cache_dir: Path = Field(default_factory=lambda: Path("output/cache"), env="DATA_CACHE_DIR")

    latest_dataset_symlink: str = Field("ml_dataset_latest.parquet", env="LATEST_DATASET_SYMLINK")
    latest_metadata_symlink: str = Field("ml_dataset_latest_metadata.json", env="LATEST_METADATA_SYMLINK")
    latest_feature_index_symlink: str = Field("feature_index_latest.json", env="LATEST_FEATURE_INDEX_SYMLINK")
    dataset_tag: str = Field("full", env="DATASET_TAG")
    dataset_retention_keep: int = Field(3, ge=1, env="DATASET_RETENTION_KEEP")
    dataset_parquet_compression: str = Field("zstd", env="DATASET_PARQUET_COMPRESSION")
    cache_ttl_days_default: int = Field(1, ge=0, env="CACHE_TTL_DAYS_DEFAULT")
    margin_daily_cache_ttl_days: int = Field(1, ge=0, env="CACHE_TTL_DAYS_MARGIN_DAILY")
    margin_weekly_cache_ttl_days: int = Field(7, ge=0, env="CACHE_TTL_DAYS_MARGIN_WEEKLY")
    topix_cache_ttl_days: int = Field(3, ge=0, env="CACHE_TTL_DAYS_TOPIX")
    trades_spec_cache_ttl_days: int = Field(7, ge=0, env="CACHE_TTL_DAYS_TRADES_SPEC")
    short_selling_cache_ttl_days: int = Field(2, ge=0, env="CACHE_TTL_DAYS_SHORT")
    sector_short_cache_ttl_days: int = Field(2, ge=0, env="CACHE_TTL_DAYS_SECTOR_SHORT")
    macro_cache_ttl_days: int = Field(14, ge=0, env="CACHE_TTL_DAYS_MACRO")
    calendar_cache_ttl_days: int = Field(7, ge=0, env="CACHE_TTL_DAYS_CALENDAR")

    # Arrow IPC cache optimization settings
    cache_format: str = Field("auto", env="CACHE_FORMAT")
    """Cache file format: 'parquet' (compatible), 'ipc' (fast), 'auto' (IPC with Parquet fallback)."""

    dataset_output_format: str = Field("parquet", env="DATASET_OUTPUT_FORMAT")
    """Final dataset format: 'parquet' (archival), 'ipc' (fast reads), 'dual' (both formats)."""

    enable_lazy_scans: bool = Field(True, env="ENABLE_LAZY_SCANS")
    """Enable Polars lazy scan optimizations (predicate pushdown, column pruning)."""

    ipc_compression: str = Field("lz4", env="IPC_COMPRESSION")
    """Compression codec for Arrow IPC files: 'lz4' (fast), 'zstd' (better ratio), 'uncompressed'."""

    request_timeout_seconds: float = Field(30.0, ge=1.0, env="REQUEST_TIMEOUT_SECONDS")

    # Cache V2 configuration
    quote_cache_subdir: str = Field("quotes_v2", env="QUOTE_CACHE_SUBDIR")
    max_cache_size_gb: float | None = Field(None, ge=0.0, env="MAX_CACHE_SIZE_GB")
    cache_stats_path: Path | None = Field(default=None, env="CACHE_STATS_PATH")

    quotes_schema_tag: str = Field("limits_v1", env="QUOTES_SCHEMA_TAG")

    # Phase 2 Patch E: ADV filter configuration
    min_adv_yen: int | None = Field(None, ge=0, env="MIN_ADV_YEN")
    adv_min_periods: int = Field(20, ge=1, env="ADV_MIN_PERIODS")

    # GPU-ETL configuration
    use_gpu_etl: bool = Field(True, env="USE_GPU_ETL")
    force_gpu: bool = Field(True, env="FORCE_GPU")
    rmm_pool_size: str = Field("40GB", env="RMM_POOL_SIZE")

    @model_validator(mode="after")
    def _ensure_directories(self) -> "DatasetBuilderSettings":
        """Make sure important directories exist to avoid runtime surprises."""

        self.data_output_dir.mkdir(parents=True, exist_ok=True)
        self.data_cache_dir.mkdir(parents=True, exist_ok=True)
        # Quote cache v2 directories
        self.quote_cache_dir.mkdir(parents=True, exist_ok=True)
        (self.quote_cache_dir / "raw" / "quotes").mkdir(parents=True, exist_ok=True)
        return self

    @property
    def latest_dataset_path(self) -> Path:
        """Return the canonical dataset path consumed by every model."""

        return self.data_output_dir / self.latest_dataset_symlink

    @property
    def default_cache_index_path(self) -> Path:
        """Return the location where cache metadata should be stored."""

        return self.data_cache_dir / "cache_index.json"

    @property
    def quote_cache_dir(self) -> Path:
        """Base directory for normalized quote shards."""

        return self.data_cache_dir / self.quote_cache_subdir

    @property
    def quote_index_path(self) -> Path:
        """Location of SQLite index for quote shards."""

        return self.quote_cache_dir / "cache_index.db"

    @property
    def quote_stats_path(self) -> Path:
        """Stream of cache metrics (jsonl)."""

        if self.cache_stats_path is not None:
            return self.cache_stats_path
        return self.quote_cache_dir / "cache_stats.jsonl"


SETTINGS_NAME: Final[str] = "dataset_builder_settings"


@lru_cache(maxsize=1)
def get_settings() -> DatasetBuilderSettings:
    """Return a cached settings instance."""

    return DatasetBuilderSettings()
