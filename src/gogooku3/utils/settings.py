"""Settings and configuration management for gogooku3."""

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _search_up(start: Path, markers: tuple[str, ...] = ("pyproject.toml", ".git")) -> Path | None:
    """Search upwards from `start` for a directory containing any marker file/dir."""
    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent
    while True:
        for m in markers:
            if (cur / m).exists():
                return cur
        if cur.parent == cur:
            return None
        cur = cur.parent


def _detect_project_root() -> Path:
    env = os.getenv("GOGOOKU3_PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    # Prefer CWD (interactive/scripts), fallback to this file's location
    for start in (Path.cwd(), Path(__file__).resolve()):
        found = _search_up(start)
        if found:
            return found
    return Path.cwd().resolve()


def _default_data_dir() -> Path:
    return Path(os.getenv("GOGOOKU3_DATA_DIR", str(_detect_project_root() / "data")))


def _default_output_dir() -> Path:
    return Path(os.getenv("GOGOOKU3_OUTPUT_DIR", str(_detect_project_root() / "output")))


def _default_config_dir() -> Path:
    # App configs live under `configs/` (not infra)
    return Path(os.getenv("GOGOOKU3_CONFIG_DIR", str(_detect_project_root() / "configs")))


class Gogooku3Settings(BaseSettings):
    """Main settings for gogooku3 system."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra fields from .env
    )

    # JQuants API Configuration
    jquants_email: str = Field(default="", description="JQuants API email")
    jquants_password: str = Field(default="", description="JQuants API password")

    # Storage Configuration
    minio_endpoint: str = Field(default="http://localhost:9000", description="MinIO endpoint")
    minio_access_key: str = Field(default="minioadmin", description="MinIO access key")
    minio_secret_key: str = Field(default="minioadmin", description="MinIO secret key")
    minio_secure: bool = Field(default=False, description="Use secure MinIO connection")

    # ClickHouse Configuration
    clickhouse_host: str = Field(default="localhost", description="ClickHouse host")
    clickhouse_port: int = Field(default=8123, description="ClickHouse port")
    clickhouse_database: str = Field(default="gogooku3", description="ClickHouse database")
    clickhouse_user: str = Field(default="default", description="ClickHouse user")
    clickhouse_password: str = Field(default="", description="ClickHouse password")

    # Redis Configuration
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_db: int = Field(default=0, description="Redis database")

    # Dagster Configuration
    dagster_home: str = Field(default="/opt/dagster/dagster_home", description="Dagster home directory")
    dagster_pg_host: str = Field(default="localhost", description="Dagster PostgreSQL host")
    dagster_pg_port: int = Field(default=5432, description="Dagster PostgreSQL port")
    dagster_pg_username: str = Field(default="dagster", description="Dagster PostgreSQL username")
    dagster_pg_password: str = Field(default="dagster", description="Dagster PostgreSQL password")
    dagster_pg_db: str = Field(default="dagster", description="Dagster PostgreSQL database")

    # Processing Configuration
    max_concurrent_fetch: int = Field(default=150, description="Maximum concurrent API fetches")
    max_parallel_workers: int = Field(default=24, description="Maximum parallel workers")

    # Monitoring Configuration
    prometheus_port: int = Field(default=8000, description="Prometheus metrics port")
    grafana_port: int = Field(default=3001, description="Grafana dashboard port")

    # Environment Configuration
    environment: str = Field(default="development", description="Environment (development/production)")

    # Path Configuration (portable; overridable via env)
    project_root: Path = Field(default_factory=_detect_project_root)
    data_dir: Path = Field(default_factory=_default_data_dir)
    output_dir: Path = Field(default_factory=_default_output_dir)
    config_dir: Path = Field(default_factory=_default_config_dir)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Gogooku3Settings()
