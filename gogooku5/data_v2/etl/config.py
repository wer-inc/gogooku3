"""Configuration helpers."""

from __future__ import annotations

from pathlib import Path

# Project root (data_v2)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Default DuckDB path
DEFAULT_DB_PATH: Path = PROJECT_ROOT / "output" / "jquants.duckdb"

# J-Quants endpoints
AUTH_URL = "https://api.jquants.com/v1/token/auth_user"
REFRESH_URL = "https://api.jquants.com/v1/token/auth_refresh"
TRADING_CAL_URL = "https://api.jquants.com/v1/markets/trading_calendar"
LISTED_URL = "https://api.jquants.com/v1/listed/info"
DAILY_QUOTES_URL = "https://api.jquants.com/v1/prices/daily_quotes"
BREAKDOWN_URL = "https://api.jquants.com/v1/markets/breakdown"
STATEMENTS_URL = "https://api.jquants.com/v1/fins/statements"
FS_DETAILS_URL = "https://api.jquants.com/v1/fins/fs_details"
TRADES_SPEC_URL = "https://api.jquants.com/v1/markets/trades_spec"

# HTTP defaults
HTTP_TIMEOUT = 60
RETRY_BACKOFF = 1.0
