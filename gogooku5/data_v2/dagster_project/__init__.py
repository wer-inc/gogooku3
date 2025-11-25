from __future__ import annotations

from dagster import Definitions

from .jobs import (
    duckdb_breakdown_job,
    duckdb_calendar_job,
    duckdb_daily_quotes_job,
    duckdb_features_job,
    duckdb_listed_job,
    duckdb_yfinance_job,
)

defs = Definitions(
    jobs=[
        duckdb_calendar_job,
        duckdb_listed_job,
        duckdb_yfinance_job,
        duckdb_daily_quotes_job,
        duckdb_breakdown_job,
        duckdb_features_job,
    ]
)
