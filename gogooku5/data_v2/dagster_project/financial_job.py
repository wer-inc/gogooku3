from __future__ import annotations

from pathlib import Path

from dagster import In, Out, job, op
from data_v2.etl.duckdb import DEFAULT_DB_PATH, connect_db, ensure_tables, ensure_views
from data_v2.etl.feature.financial import compute_financial_features
from data_v2.etl.upsert.financial_features import upsert_financial_features


@op(
    ins={
        "start": In(str, description="Start date YYYY-MM-DD"),
        "end": In(str, description="End date YYYY-MM-DD"),
        "db_path": In(str, description="DuckDB path", default_value=str(DEFAULT_DB_PATH)),
        "threads": In(int, description="DuckDB threads (0=default)", default_value=0),
    },
    out=Out(str, description="Result string"),
)
def duckdb_build_financial_features_op(context, start: str, end: str, db_path: str, threads: int) -> str:
    con = connect_db(Path(db_path), threads=threads or None)
    ensure_tables(con)
    ensure_views(con)
    df = compute_financial_features(con, start=start, end=end)
    updated = upsert_financial_features(con, df)
    msg = f"financial_features upserted: {updated} rows into {db_path} for {start}->{end}"
    context.log.info(msg)
    return msg


@job
def duckdb_financial_features_job():
    # Default run uses env/Dagster config; op inputs can be supplied via run config.
    duckdb_build_financial_features_op()
