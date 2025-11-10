"""
Configuration integration for gogooku5-dataset CLI.

Merges settings from three sources with priority:
1. CLI arguments (highest priority)
2. Environment variables (.env)
3. Default values (lowest priority)

Handles environment variable mappings and type conversions.
"""

import os
from argparse import Namespace
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# Environment variable mappings
ENV_MAPPING = {
    "USE_CACHE": ("cache_enabled", bool),
    "USE_GPU_ETL": ("gpu_enabled", bool),
    "RMM_POOL_SIZE": ("rmm_pool_size", str),
    "WARMUP_DAYS": ("warmup_days", str),  # Can be "auto" or integer
    "MAX_CONCURRENT_FETCH": ("max_concurrent_fetch", int),
    "MAX_PARALLEL_WORKERS": ("max_parallel_workers", int),
    "QUOTE_CACHE_DIR": ("quote_cache_dir", str),
    "GCS_ENABLED": ("gcs_enabled", bool),
    "GCS_BUCKET": ("gcs_bucket", str),
    "JQUANTS_AUTH_EMAIL": ("jquants_email", str),
    "JQUANTS_AUTH_PASSWORD": ("jquants_password", str),
    "JQUANTS_PLAN_TIER": ("jquants_plan_tier", str),
    "MIN_CACHE_COVERAGE": ("min_cache_coverage", float),
    "ENABLE_MULTI_CACHE": ("enable_multi_cache", bool),
    "CACHE_MAX_AGE_DAYS": ("cache_max_age_days", int),
    "GCS_SYNC_AFTER_SAVE": ("gcs_sync_after_save", bool),
}


class Config:
    """
    Unified configuration object.

    Attributes:
        All settings from CLI args, environment, and defaults.
    """

    def __init__(self, args: Namespace, env_file: Optional[Path] = None):
        """
        Initialize configuration.

        Args:
            args: Parsed CLI arguments
            env_file: Path to .env file (default: auto-detect)
        """
        self.args = args
        self._load_env(env_file)
        self._merge_settings()

    def _load_env(self, env_file: Optional[Path]) -> None:
        """Load environment variables from .env file."""
        if env_file:
            load_dotenv(env_file)
        else:
            # Auto-detect .env in common locations
            candidates = [
                Path.cwd() / ".env",
                Path(__file__).parents[3] / ".env",  # gogooku5/data/.env
            ]
            for candidate in candidates:
                if candidate.exists():
                    load_dotenv(candidate)
                    break

    def _merge_settings(self) -> None:
        """
        Merge settings from CLI args, env vars, and defaults.

        Priority: CLI args > env vars > defaults
        """
        # Core settings
        self.command = self.args.command

        if self.command == "build":
            self._merge_build_settings()
        elif self.command == "merge":
            self._merge_merge_settings()
        elif self.command == "check":
            self._merge_check_settings()

        # Common settings
        self._merge_logging_settings()
        self._merge_env_vars()

    def _merge_build_settings(self) -> None:
        """Merge build-specific settings."""
        args = self.args

        # Period
        self.start_date = args.start
        self.end_date = args.end
        self.lookback_years = args.lookback_years

        # Chunking
        self.chunk_months = args.chunk_months
        self.chunk_mode = args.chunk_mode
        self.resume = args.resume
        self.force = args.force
        self.latest = args.latest

        # Data sources
        self.jquants_enabled = args.jquants and not args.offline
        self.offline_mode = args.offline
        self.force_refresh = args.force_refresh
        self.refresh_listed = args.refresh_listed

        # Compute resources
        if args.no_gpu:
            self.gpu_enabled = False
        elif args.gpu:
            self.gpu_enabled = True
        else:
            # Auto-detect from env
            self.gpu_enabled = self._get_env_bool("USE_GPU_ETL", default=True)

        self.rmm_pool_size = args.rmm_pool_size or os.getenv("RMM_POOL_SIZE", "40GB")
        self.workers = args.workers

        # Features
        self.feature_preset = args.features
        self.enable_graph = not args.disable_graph if hasattr(args, "disable_graph") else args.enable_graph
        self.enable_sector = args.enable_sector
        self.enable_margin = args.enable_margin
        self.enable_short_selling = args.enable_short_selling
        self.futures_continuous = args.futures_continuous

        # Output
        self.output_dir = Path(args.output_dir)
        self.tag = args.tag
        self.auto_merge = args.merge
        self.allow_partial = args.allow_partial
        self.background = args.background

        # Debug/validation
        self.dry_run = args.dry_run
        self.check_only = args.check
        self.check_strict = args.check_strict

    def _merge_merge_settings(self) -> None:
        """Merge merge-specific settings."""
        args = self.args

        self.chunks_dir = Path(args.chunks_dir)
        self.output_dir = Path(args.output_dir)
        self.allow_partial = args.allow_partial
        self.strict = args.strict
        self.tag = args.tag

    def _merge_check_settings(self) -> None:
        """Merge check-specific settings."""
        args = self.args

        self.check_strict = args.strict
        self.gpu_required = args.gpu_required

    def _merge_logging_settings(self) -> None:
        """Merge logging settings."""
        args = self.args

        if hasattr(args, "verbose") and args.verbose:
            self.log_level = "DEBUG"
        elif hasattr(args, "quiet") and args.quiet:
            self.log_level = "WARNING"
        else:
            self.log_level = "INFO"

        self.log_file = getattr(args, "log_file", None)

    def _merge_env_vars(self) -> None:
        """Merge environment variables using ENV_MAPPING."""
        for env_key, (attr_name, type_fn) in ENV_MAPPING.items():
            if not hasattr(self, attr_name):
                value = os.getenv(env_key)
                if value is not None:
                    setattr(self, attr_name, self._convert_type(value, type_fn))

        # Apply defaults for missing values
        self._apply_defaults()

    def _apply_defaults(self) -> None:
        """Apply default values for settings not set by CLI or env."""
        defaults = {
            "cache_enabled": True,
            "max_concurrent_fetch": 75,
            "max_parallel_workers": 20,
            "quote_cache_dir": "output/raw/prices",
            "gcs_enabled": False,
            "gcs_bucket": "gogooku-ml-datasets",
            "jquants_plan_tier": "premium",
            "min_cache_coverage": 0.3,
            "enable_multi_cache": True,
            "cache_max_age_days": 7,
            "gcs_sync_after_save": True,
            "warmup_days": "auto",
        }

        for key, default_value in defaults.items():
            if not hasattr(self, key):
                setattr(self, key, default_value)

    def _convert_type(self, value: str, type_fn: type) -> Any:
        """
        Convert string value to specified type.

        Args:
            value: String value from environment
            type_fn: Target type (bool, int, float, str)

        Returns:
            Converted value
        """
        if type_fn == bool:
            return value.lower() in ("1", "true", "yes", "on")
        elif type_fn == int:
            return int(value)
        elif type_fn == float:
            return float(value)
        else:
            return value

    def _get_env_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean value from environment variable."""
        value = os.getenv(key)
        if value is None:
            return default
        return value.lower() in ("1", "true", "yes", "on")

    def to_dict(self) -> dict:
        """Export configuration as dictionary."""
        return {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.__dict__.items()
            if not k.startswith("_") and k != "args"
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        settings = self.to_dict()
        lines = [f"{k}={v}" for k, v in sorted(settings.items())]
        return "Config(\n  " + ",\n  ".join(lines) + "\n)"


def load_config(args: Namespace, env_file: Optional[Path] = None) -> Config:
    """
    Load and validate configuration.

    Args:
        args: Parsed CLI arguments
        env_file: Optional path to .env file

    Returns:
        Config object with merged settings
    """
    return Config(args, env_file)
