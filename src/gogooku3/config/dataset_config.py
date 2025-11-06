"""
Unified configuration management for dataset generation pipeline.

Integrates CLI arguments and environment variables using Pydantic BaseSettings.
Provides validation, defaults, and type safety.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class JQuantsAPIConfig(BaseSettings):
    """J-Quants API authentication and connection settings."""

    model_config = SettingsConfigDict(
        env_prefix="JQUANTS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Authentication
    auth_email: str = Field(..., description="J-Quants API email")
    auth_password: str = Field(..., description="J-Quants API password")
    min_available_date: str | None = Field(None, description="Subscription lower bound (YYYY-MM-DD)")

    # Connection settings
    max_concurrent_fetch: int = Field(75, description="Maximum concurrent API requests (paid plan)")
    min_concurrency: int = Field(4, description="Minimum concurrency during throttle")
    tcp_limit: int = Field(30, description="TCP connection limit")
    tcp_limit_per_host: int = Field(15, description="TCP connection limit per host")
    sock_connect_timeout: float = Field(10.0, description="Socket connect timeout (seconds)")
    sock_read_timeout: float = Field(60.0, description="Socket read timeout (seconds)")

    # Throttling
    throttle_backoff: float = Field(0.6, description="Concurrency reduction factor on throttle")
    throttle_sleep: float = Field(30.0, description="Sleep duration after throttle (seconds)")
    throttle_step: int = Field(2, description="Recovery step size")
    throttle_recovery_success: int = Field(180, description="Success count needed for recovery")

    @field_validator("auth_email", "auth_password")
    @classmethod
    def validate_credentials(cls, v: str) -> str:
        if not v:
            raise ValueError("J-Quants credentials must be provided")
        return v


class GPUConfig(BaseSettings):
    """GPU-ETL and RMM (RAPIDS Memory Manager) settings."""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    use_gpu_etl: bool = Field(
        True,
        alias="USE_GPU_ETL",
        description="Enable GPU-accelerated ETL (RAPIDS/cuDF)",
    )
    require_gpu: bool = Field(False, alias="REQUIRE_GPU", description="Require GPU (fail if unavailable)")
    rmm_allocator: Literal["pool", "cuda_async", "managed"] = Field(
        "cuda_async", alias="RMM_ALLOCATOR", description="RMM allocator type"
    )
    rmm_pool_size: str = Field(
        "70GB",
        alias="RMM_POOL_SIZE",
        description="RMM pool size (e.g., 70GB, 0 for dynamic)",
    )
    cudf_spill: bool = Field(True, alias="CUDF_SPILL", description="Enable cuDF spilling to avoid OOM")
    cuda_visible_devices: str = Field("0", alias="CUDA_VISIBLE_DEVICES", description="GPU device IDs")


class EarningsEventConfig(BaseSettings):
    """Earnings announcement feature configuration."""

    model_config = SettingsConfigDict(
        env_prefix="EARNINGS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    asof_hour: int = Field(
        15,
        alias="EARNINGS_ASOF_HOUR",
        description="Dataset as-of hour (JST) for earnings availability checks",
    )
    windows: list[int] = Field(
        default_factory=lambda: [1, 3, 5],
        alias="EARNINGS_WINDOWS",
        description="Business-day proximity windows for earnings flags",
    )

    @field_validator("asof_hour")
    @classmethod
    def validate_hour(cls, value: int) -> int:
        if not 0 <= value <= 23:
            raise ValueError("EARNINGS_ASOF_HOUR must be between 0 and 23")
        return value

    @field_validator("windows", mode="before")
    @classmethod
    def parse_windows(cls, value: object) -> list[int]:
        if isinstance(value, str):
            tokens = [token.strip() for token in value.split(",") if token.strip()]
            parsed = [int(tok) for tok in tokens if tok]
        elif isinstance(value, (list, tuple)):
            parsed = [int(v) for v in value]
        elif value is None:
            parsed = [1, 3, 5]
        else:  # pragma: no cover - defensive
            raise TypeError("EARNINGS_WINDOWS must be list-like or comma-separated str")

        parsed = sorted({w for w in parsed if w > 0})
        if not parsed:
            raise ValueError("EARNINGS_WINDOWS must contain positive integers")
        return parsed


class AMSessionConfig(BaseSettings):
    """Morning session (prices_am) configuration."""

    model_config = SettingsConfigDict(
        env_prefix="AM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    asof_policy: str = Field(
        "T+1",
        alias="AM_ASOF_POLICY",
        description="As-of policy for AM features (T+1 or SAME_DAY_PM)",
    )
    same_day_cutoff: str = Field(
        "11:35",
        alias="AM_SAME_DAY_CUTOFF",
        description="Cutoff (HH:MM) when SAME_DAY_PM policy is active",
    )

    @field_validator("asof_policy")
    @classmethod
    def validate_policy(cls, value: str) -> str:
        policy = (value or "T+1").upper()
        if policy not in {"T+1", "SAME_DAY_PM"}:
            raise ValueError("AM_ASOF_POLICY must be T+1 or SAME_DAY_PM")
        return policy

    @field_validator("same_day_cutoff")
    @classmethod
    def validate_cutoff(cls, value: str) -> str:
        token = (value or "11:35").strip()
        try:
            datetime.strptime(token, "%H:%M")
        except ValueError as exc:
            raise ValueError("AM_SAME_DAY_CUTOFF must be HH:MM") from exc
        return token


class FeatureFlagsConfig(BaseSettings):
    """Feature flags for dataset generation components."""

    model_config = SettingsConfigDict(
        env_prefix="ENABLE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Core features (always enabled by default)
    indices: bool = Field(True, description="Indices OHLC features (spreads, breadth)")
    advanced_vol: bool = Field(True, description="Yang-Zhang volatility and VoV features")
    advanced_features: bool = Field(True, description="Advanced T+0 features (RSI×Vol, momentum×volume, etc.)")

    # Graph features
    graph_features: bool = Field(
        True,
        alias="ENABLE_GRAPH_FEATURES",
        description="Graph-structured features (degree, correlation)",
    )

    # Margin features
    margin_weekly: bool = Field(True, description="Weekly margin interest features")
    daily_margin: bool = Field(True, alias="ENABLE_DAILY_MARGIN", description="Daily margin interest features")

    # Options features
    nk225_option_features: bool = Field(
        True,
        alias="ENABLE_NK225_OPTION_FEATURES",
        description="Nikkei225 index option features",
    )
    option_market_features: bool = Field(False, description="Attach NK225 option market aggregates to equity panel")
    macro_vix_features: bool = Field(
        True,
        alias="ENABLE_MACRO_VIX",
        description="Attach VIX-based macro sentiment features",
    )
    macro_fx_usdjpy: bool = Field(
        True,
        alias="ENABLE_MACRO_FX_USDJPY",
        description="Attach USD/JPY FX macro features",
    )
    macro_btc: bool = Field(
        True,
        alias="ENABLE_MACRO_BTC",
        description="Attach BTC/USD crypto macro features",
    )

    # Morning session features
    am_session: bool = Field(
        True,
        alias="ENABLE_AM_FEATURES",
        description="Attach morning session (prices_am) features",
    )

    # Futures features (disabled: API not available)
    futures: bool = Field(False, description="Futures features (API not available)")

    # Sector features
    sector_cs: bool = Field(True, alias="ENABLE_SECTOR_CS", description="Sector cross-sectional features")
    sector_onehot33: bool = Field(False, description="33-sector one-hot encodings")

    # Short selling features
    short_selling: bool = Field(True, alias="ENABLE_SHORT_SELLING", description="Short selling data integration")
    sector_short_selling: bool = Field(
        True,
        alias="ENABLE_SECTOR_SHORT_SELLING",
        description="Sector-wise short selling features",
    )
    sector_short_z_scores: bool = Field(True, description="Z-score features for sector short selling")

    # Earnings features
    earnings_events: bool = Field(True, alias="ENABLE_EARNINGS_EVENTS", description="Earnings announcement events")
    pead_features: bool = Field(
        True,
        alias="ENABLE_PEAD_FEATURES",
        description="Post-Earnings Announcement Drift features",
    )


class GraphConfig(BaseSettings):
    """Graph feature generation settings."""

    model_config = SettingsConfigDict(
        env_prefix="GRAPH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    window: int = Field(60, description="Correlation window (days)")
    threshold: float = Field(0.3, description="Absolute correlation threshold for edges")
    max_k: int = Field(4, alias="GRAPH_MAX_K", description="Max edges per node")
    cache_dir: Path | None = Field(None, alias="GRAPH_CACHE_DIR", description="Cache directory for graph artifacts")


class DatasetConfig(BaseSettings):
    """Main dataset generation configuration."""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Date range
    start_date: str | None = Field(None, description="Start date (YYYY-MM-DD, default: end - 5 years)")
    end_date: str | None = Field(None, description="End date (YYYY-MM-DD, default: today)")
    lookback_days: int = Field(1826, description="Default lookback days (~5 years)")
    support_lookback_days: int = Field(420, description="Support lookback for flow/z-score features")
    min_collection_days: int = Field(
        3650,
        alias="MIN_COLLECTION_DAYS",
        description="Minimum collection span (~10 years)",
    )

    # Pipeline settings
    use_jquants: bool = Field(True, description="Use J-Quants API (vs offline data)")
    use_calendar_api: bool = Field(True, description="Use Trading Calendar API for business days")
    disable_halt_mask: bool = Field(False, description="Disable special halt-day masking (2020-10-01)")
    futures_continuous: bool = Field(False, description="Enable continuous futures series")

    # Output settings
    output_dir: Path = Field(Path("output/datasets"), description="Output directory for datasets")
    save_intermediate: bool = Field(True, description="Save intermediate artifacts")

    # Sub-configurations
    jquants: JQuantsAPIConfig = Field(default_factory=JQuantsAPIConfig)
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    features: FeatureFlagsConfig = Field(default_factory=FeatureFlagsConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    earnings: EarningsEventConfig = Field(default_factory=EarningsEventConfig)
    am: AMSessionConfig = Field(default_factory=AMSessionConfig)

    @model_validator(mode="after")
    def validate_and_compute_dates(self) -> DatasetConfig:
        """Validate and compute default date ranges."""
        # Compute end_date if not provided
        if not self.end_date:
            self.end_date = datetime.now().strftime("%Y-%m-%d")

        # Compute start_date if not provided
        if not self.start_date:
            end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
            start_dt = end_dt - timedelta(days=self.lookback_days)
            self.start_date = start_dt.strftime("%Y-%m-%d")

        # Validate date format
        try:
            start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid date format (expected YYYY-MM-DD): {e}")

        # Validate date range
        if start_dt >= end_dt:
            raise ValueError(f"start_date ({self.start_date}) must be before end_date ({self.end_date})")

        # Check minimum span
        span_days = (end_dt - start_dt).days
        min_span = max(self.min_collection_days, self.support_lookback_days)
        if span_days < min_span:
            # Auto-extend start_date
            adjusted_start_dt = end_dt - timedelta(days=min_span)
            # Check against subscription lower bound
            if self.jquants.min_available_date:
                min_available_dt = datetime.strptime(self.jquants.min_available_date, "%Y-%m-%d")
                if adjusted_start_dt < min_available_dt:
                    adjusted_start_dt = min_available_dt
            self.start_date = adjusted_start_dt.strftime("%Y-%m-%d")

        return self

    @classmethod
    def from_cli_and_env(
        cls,
        cli_args: dict | None = None,
        env_file: str | Path | None = None,
    ) -> DatasetConfig:
        """Create config from CLI arguments and environment variables.

        Args:
            cli_args: Dictionary of CLI arguments (overrides env vars)
            env_file: Optional .env file path (default: project root)

        Returns:
            Configured DatasetConfig instance
        """
        # Set env_file if provided
        if env_file:
            os.environ["SETTINGS_ENV_FILE"] = str(env_file)

        # Create base config from environment
        config = cls()

        # Override with CLI arguments if provided
        if cli_args:
            # Filter out None values
            cli_overrides = {k: v for k, v in cli_args.items() if v is not None}

            # Handle nested configs
            for key, value in cli_overrides.items():
                if key in {"jquants", "gpu", "features", "graph"}:
                    # Update nested config
                    nested_config = getattr(config, key)
                    for nested_key, nested_value in value.items():
                        setattr(nested_config, nested_key, nested_value)
                else:
                    # Update top-level config
                    setattr(config, key, value)

        return config

    def to_dict(self) -> dict:
        """Export configuration as dictionary."""
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "lookback_days": self.lookback_days,
            "support_lookback_days": self.support_lookback_days,
            "use_jquants": self.use_jquants,
            "output_dir": str(self.output_dir),
            "jquants": {
                "max_concurrent_fetch": self.jquants.max_concurrent_fetch,
                "min_concurrency": self.jquants.min_concurrency,
                "tcp_limit": self.jquants.tcp_limit,
            },
            "gpu": {
                "use_gpu_etl": self.gpu.use_gpu_etl,
                "require_gpu": self.gpu.require_gpu,
                "rmm_allocator": self.gpu.rmm_allocator,
                "rmm_pool_size": self.gpu.rmm_pool_size,
            },
            "features": {
                "graph_features": self.features.graph_features,
                "daily_margin": self.features.daily_margin,
                "short_selling": self.features.short_selling,
                "sector_short_selling": self.features.sector_short_selling,
            },
            "graph": {
                "window": self.graph.window,
                "threshold": self.graph.threshold,
                "max_k": self.graph.max_k,
            },
        }
