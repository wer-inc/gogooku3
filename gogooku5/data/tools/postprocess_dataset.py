#!/usr/bin/env python3
"""Post-process merged ML dataset to reduce NULLs at chunk boundaries.

Capabilities:
1) Recompute forward targets across chunk boundaries (code/date sorted, lazy).
2) Optionally drop high-null columns (default: drop if null_rate >= 0.99).
3) Optional allowlist/deny-pattern to control column retention.

TTM/長期履歴依存の再計算はスコープ外（別途拡張可能）。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence

import polars as pl

from lazy_dataset_utils import (
    add_future_return_columns,
    drop_existing_target_columns,
    scan_parquet_lazy,
)

DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 10, 20)
TARGET_PREFIX = "target_"


def _detect_keys(schema: pl.Schema) -> tuple[str, str]:
    names = [c.lower() for c in schema.names()]
    code = schema.names()[names.index("code")] if "code" in names else "Code"
    date = schema.names()[names.index("date")] if "date" in names else "Date"
    return code, date


def _detect_price(schema: pl.Schema) -> str:
    for cand in ("AdjustmentClose", "Close", "adjustmentclose", "close"):
        if cand in schema:
            return cand
    raise ValueError("Price column not found (expected AdjustmentClose/Close)")


def _detect_targets(schema: pl.Schema) -> Sequence[int]:
    horizons: list[int] = []
    for name in schema.names():
        if name.startswith(TARGET_PREFIX) and name.endswith("d"):
            try:
                horizons.append(int(name[len(TARGET_PREFIX) : -1]))
            except ValueError:
                continue
    return sorted(set(horizons)) or list(DEFAULT_HORIZONS)


def recompute_forward_targets(path: Path, output: Path) -> pl.LazyFrame:
    lf = scan_parquet_lazy(path)
    horizons = _detect_targets(lf.collect_schema())
    lf, dropped = drop_existing_target_columns(lf, prefix=TARGET_PREFIX)
    schema = lf.collect_schema()
    code_col, date_col = _detect_keys(schema)
    price_col = _detect_price(schema)
    lf = add_future_return_columns(
        lf, horizons, price_col=price_col, code_col=code_col, date_col=date_col
    )
    lf.sink_parquet(output, compression="zstd", statistics=False)
    return pl.scan_parquet(output)


def drop_high_null(
    lf_or_df: pl.LazyFrame | pl.DataFrame,
    *,
    threshold: float,
    allowlist: set[str],
    deny_patterns: list[re.Pattern[str]],
) -> pl.DataFrame:
    df = lf_or_df.collect() if isinstance(lf_or_df, pl.LazyFrame) else lf_or_df
    null_rates = df.select([pl.col(c).is_null().mean().alias(c) for c in df.columns]).to_dicts()[0]
    drop_cols = []
    for col, rate in null_rates.items():
        if col in allowlist:
            continue
        if any(p.search(col) for p in deny_patterns):
            drop_cols.append(col)
            continue
        if rate >= threshold:
            drop_cols.append(col)
    if drop_cols:
        df = df.drop(drop_cols)
    return df


def coverage_report(df: pl.DataFrame, targets: Sequence[str]) -> dict:
    report = {
        "rows": df.height,
        "cols": df.width,
    }
    for col in targets:
        if col in df.columns:
            report[f"{col}_non_null_rate"] = float(df.select(pl.col(col).is_not_null().mean()).item())
    return report


def _attach_fs(df: pl.DataFrame, *, settings) -> pl.DataFrame:
    """Recompute FS-derived TTM/YoY features using DataSourceManager (raw if available)."""

    try:
        from builder.api.data_sources import DataSourceManager
        from builder.features.fundamentals.fins_asof import prepare_fs_snapshot
        from builder.utils import add_asof_timestamp
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to import DS components: {exc}") from exc

    dsm = DataSourceManager(settings=settings)
    date_col = "date" if "date" in df.columns else "Date"
    min_date = df.select(pl.col(date_col).min()).item().isoformat()
    max_date = df.select(pl.col(date_col).max()).item().isoformat()

    raw_fs = dsm.fs_details(start=min_date, end=max_date)
    if raw_fs.is_empty():
        return df

    fs_snapshot = prepare_fs_snapshot(raw_fs)
    fs_snapshot = add_asof_timestamp(fs_snapshot, date_col="DisclosedDate")
    join_keys = ["code"] if "code" in df.columns else ["Code"]
    fs_snapshot = fs_snapshot.rename({col: col.lower() for col in fs_snapshot.columns})

    fs_cols = [c for c in fs_snapshot.columns if c.startswith("fs_") or c.startswith("div_")]
    if not fs_cols:
        return df

    lhs = df
    rhs = fs_snapshot.select(join_keys + ["discloseddate", "available_ts"] + fs_cols)
    rhs = rhs.rename({"discloseddate": "asof_date"})
    lhs = lhs.with_columns(pl.col(date_col).alias("record_date"))
    rhs = rhs.with_columns(pl.col("available_ts").alias("asof_ts"))

    lhs = lhs.sort(join_keys + ["record_date"])
    rhs = rhs.sort(join_keys + ["asof_ts"])
    out = lhs.join_asof(
        rhs,
        left_on="record_date",
        right_on="asof_ts",
        by=join_keys,
        strategy="backward",
    )
    out = out.drop(["record_date", "asof_ts", "asof_date"], strict=False)
    return out


def trim_tail(
    df: pl.DataFrame,
    *,
    horizons: Sequence[int],
    date_col: str,
    code_col: str,
    trim_days: int,
) -> pl.DataFrame:
    """Drop the last `trim_days` worth of rows per code to avoid missing forward targets."""

    if trim_days <= 0:
        return df
    df_sorted = df.sort([code_col, date_col])
    max_dates = df_sorted.group_by(code_col).agg(pl.col(date_col).max().alias("_max_date"))
    df_joined = df_sorted.join(max_dates, on=code_col, how="left")
    cutoff = pl.col("_max_date") - pl.duration(days=trim_days)
    trimmed = df_joined.filter(pl.col(date_col) <= cutoff).drop("_max_date", strict=False)
    return trimmed


def load_list(path: Path | None) -> set[str]:
    if path is None:
        return set()
    ext = path.suffix.lower()
    if ext in {".json"}:
        data = json.loads(path.read_text())
        return set(data)
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Post-process merged dataset to reduce NULLs.")
    ap.add_argument("--input", required=True, type=Path, help="Merged dataset parquet/ipc")
    ap.add_argument("--output", required=False, type=Path, help="Output path (default: *_final.parquet)")
    ap.add_argument("--no-forward", action="store_true", help="Skip forward target recomputation")
    ap.add_argument("--drop-high-null", action="store_true", help="Drop high-null columns")
    ap.add_argument("--null-threshold", type=float, default=0.99, help="Null-rate threshold for drop")
    ap.add_argument("--allowlist", type=Path, help="File with columns to keep (json/lines)")
    ap.add_argument("--deny-pattern", action="append", default=[], help="Regex pattern to drop columns")
    ap.add_argument("--recompute-ttm", action="store_true", help="Recompute FS TTM/YoY using raw/cache if available")
    ap.add_argument("--trim-tail", type=int, default=0, help="Drop last N calendar days per code to avoid missing forward targets")
    ap.add_argument("--coverage-report", action="store_true", help="Print target coverage after processing")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    input_path: Path = args.input
    output_path: Path = args.output or input_path.with_name(input_path.stem + "_final.parquet")

    # Stage 1: forward targets
    if args.no_forward:
        lf = scan_parquet_lazy(input_path)
    else:
        print("[POST] Recomputing forward targets lazily...")
        lf = recompute_forward_targets(input_path, output_path)

    # Stage 2: materialize and optional recomputations
    df = lf.collect()

    # Optional: recompute FS TTM/YoY using raw/cache
    if args.recompute_ttm:
        try:
            from builder.config.settings import DatasetBuilderSettings

            settings = DatasetBuilderSettings()
            print("[POST] Recomputing FS TTM/YoY features using raw/cache if available...")
            df = _attach_fs(df, settings=settings)
        except Exception as exc:
            print(f"[POST] Skipping FS recompute due to error: {exc}")

    # Optional: trim tail (drop last N days) to avoid missing forward horizons at end
    if args.trim_tail > 0:
        code_col, date_col = _detect_keys(df.schema)
        horizons = _detect_targets(df.schema)
        print(f"[POST] Trimming last {args.trim_tail} days per code to avoid tail NULLs")
        df = trim_tail(df, horizons=horizons, date_col=date_col, code_col=code_col, trim_days=args.trim_tail)

    # High-null drop (after recomputations)
    allow = load_list(args.allowlist)
    deny_patterns = [re.compile(pat) for pat in args.deny_pattern] if args.deny_pattern else []
    if args.drop_high_null:
        print(f"[POST] Dropping columns with null_rate >= {args.null_threshold} (allowlist size={len(allow)})")
        df = drop_high_null(df, threshold=args.null_threshold, allowlist=allow, deny_patterns=deny_patterns)

    # Save final
    df.write_parquet(output_path, compression="zstd", use_pyarrow=True)
    print(f"[POST] Final dataset saved to {output_path} (rows={df.height}, cols={df.width})")
    if args.coverage_report:
        targets = [c for c in df.columns if c.startswith(TARGET_PREFIX)]
        report = coverage_report(df, targets)
        print("[POST] Coverage report:")
        for k, v in report.items():
            if isinstance(v, float):
                print(f"  {k}: {v*100:.2f}%")
            else:
                print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
