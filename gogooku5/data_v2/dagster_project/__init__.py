from __future__ import annotations

from dagster import Definitions

from .jobs import (
    duckdb_breakdown_job,
    duckdb_calendar_job,
    duckdb_daily_quotes_job,
    duckdb_features_job,
    duckdb_fs_details_job,
    duckdb_listed_job,
    duckdb_listed_meta_features_job,
    duckdb_price_flow_features_job,
    duckdb_short_selling_job,
    duckdb_short_selling_positions_job,
    duckdb_statements_job,
    duckdb_weekly_margin_interest_job,
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
        duckdb_price_flow_features_job,
        duckdb_listed_meta_features_job,
        duckdb_statements_job,
        duckdb_fs_details_job,
        duckdb_weekly_margin_interest_job,
        duckdb_short_selling_job,
        duckdb_short_selling_positions_job,
    ]
)
