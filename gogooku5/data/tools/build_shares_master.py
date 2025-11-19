#!/usr/bin/env python3
"""Build a shares master snapshot from fs_details.

The output is a snapshot table keyed by (code, available_ts) that records the
latest known total shares outstanding and (optionally) free-float shares.
Dataset builder can interval-join this artifact to inject share counts into
each chunk without recomputing per build.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from ..src.builder.api.data_sources import DataSourceManager
from ..src.builder.config import get_settings
from ..src.builder.features.fundamentals.fins_asof import (
    prepare_fs_snapshot,
)

# Reuse candidate lists used by fins_asof
from ..src.builder.features.fundamentals.fins_asof import (
    _AVERAGE_SHARES_CANDIDATES,
    _ISSUED_SHARES_CANDIDATES,
    _TREASURY_SHARES_CANDIDATES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build shares master snapshot from fs_details.")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the shares master parquet (default: <DATA_CACHE_DIR>/shares_master.parquet)",
    )
    return parser.parse_args()


def _extract_share_columns(snapshot: pl.DataFrame) -> pl.DataFrame:
    """Extract share-related columns from the fs snapshot."""

    if snapshot.is_empty():
        return pl.DataFrame(
            {
                "code": pl.Series([], dtype=pl.Utf8),
                "available_ts": pl.Series([], dtype=pl.Datetime("us", "Asia/Tokyo")),
                "shares_total": pl.Series([], dtype=pl.Float64),
                "shares_free_float": pl.Series([], dtype=pl.Float64),
            }
        )

    def _resolve(candidates: tuple[str, ...]) -> str | None:
        for column in candidates:
            if column in snapshot.columns:
                return column
        return None

    issued = _resolve(_ISSUED_SHARES_CANDIDATES)
    treasury = _resolve(_TREASURY_SHARES_CANDIDATES)
    average = _resolve(_AVERAGE_SHARES_CANDIDATES)

    working = snapshot

    if issued:
        working = working.with_columns(pl.col(issued).cast(pl.Float64, strict=False).alias("_issued_shares"))
    else:
        working = working.with_columns(pl.lit(None).cast(pl.Float64).alias("_issued_shares"))

    if treasury:
        working = working.with_columns(pl.col(treasury).cast(pl.Float64, strict=False).alias("_treasury_shares"))
    else:
        working = working.with_columns(pl.lit(None).cast(pl.Float64).alias("_treasury_shares"))

    if average:
        working = working.with_columns(pl.col(average).cast(pl.Float64, strict=False).alias("_average_shares"))
    else:
        working = working.with_columns(pl.lit(None).cast(pl.Float64).alias("_average_shares"))

    working = working.with_columns(
        (pl.col("_issued_shares") - pl.col("_treasury_shares").fill_null(0.0)).alias("shares_total"),
        pl.col("_average_shares").alias("shares_free_float"),
    )

    return working.select(
        [
            pl.col("Code").cast(pl.Utf8, strict=False).alias("code"),
            pl.col("available_ts"),
            pl.col("shares_total"),
            pl.col("shares_free_float"),
        ]
    ).drop_nulls("code")


def build_shares_master(start: str, end: str, output_path: Path | None = None) -> Path:
    settings = get_settings()
    manager = DataSourceManager(settings=settings)
    fs_df = manager.fs_details(start=start, end=end)
    if fs_df.is_empty():
        raise RuntimeError("fs_details returned zero rows; cannot build shares master.")

    snapshot = prepare_fs_snapshot(
        fs_df,
        trading_calendar=None,
        availability_hour=15,
        availability_minute=0,
    )
    shares = _extract_share_columns(snapshot)
    shares = shares.filter(pl.col("shares_total").is_not_null())

    if output_path is None:
        output_path = settings.data_cache_dir / "shares_master.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shares.write_parquet(output_path, compression=settings.dataset_parquet_compression)
    return output_path


def main() -> int:
    args = parse_args()
    try:
        path = build_shares_master(args.start, args.end, args.output)
    except Exception as exc:  # pragma: no cover - CLI friendly message
        print(f"❌ Failed to build shares master: {exc}")
        return 1
    print(f"✅ Shares master written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
