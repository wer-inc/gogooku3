#!/usr/bin/env python3
"""Drop high-NULL columns from a parquet dataset, with required keeps.

Columns listed in REQUIRED_KEEP_COLS (およびCLI指定のkeep)はNULL率に関わらず残します。
使いどころ: Full→Cleanの前処理や、列数の多いデータセットから高NULL列を一括で落としたいとき。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import polars as pl

# 常に保持する列（閾値判定の対象外）
REQUIRED_KEEP_COLS: set[str] = {
    # キー列
    "Date",
    "date",
    "Code",
    "code",
    # 決算イベントの物理列を落とさない
    "fs_E_event_date",
    "fs_days_since_E",
    "fs_window_e_pm1",
    "fs_window_e_pp3",
    "fs_window_e_pp5",
    # Earnings announcement系（高NULLでも残す）
    "days_to_earnings",
    "is_E_0",
    "is_E_pp3",
    # 需給/フロー・ベーシス・β/α・RR（高NULLでも列は保持）
    "mkt_flow_flow_foreigners_net_ratio_zscore_20d",
    "mkt_flow_flow_individuals_net_ratio_zscore_20d",
    "mkt_flow_flow_divergence_foreigners_vs_individuals_zscore_20d",
    "mkt_flow_flow_total_net_zscore_20d",
    "basis_gate",
    "basis_gate_zscore_20d",
    "beta60_topix",
    "alpha60_topix",
    "idxopt_rr_25",
}


def _find_high_null_columns(
    df: pl.DataFrame,
    keep: Iterable[str],
    *,
    threshold: float,
) -> list[tuple[str, int, float]]:
    """Return list of (column_name, null_count, null_rate) for high-NULL columns."""

    height = df.height
    if height == 0:
        return []

    keep_set = set(keep) | REQUIRED_KEEP_COLS
    null_counts_df = df.null_count()
    null_counts_dict = null_counts_df.to_dict(as_series=False)
    to_drop: list[tuple[str, int, float]] = []
    for name, null_list in null_counts_dict.items():
        if name in keep_set:
            continue
        null_count = null_list[0]
        rate = (null_count / height) * 100.0
        if rate >= threshold:
            to_drop.append((name, null_count, rate))
    return to_drop


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Drop columns whose NULL rate is greater than or equal to a threshold.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, type=Path, help="Input parquet file.")
    parser.add_argument(
        "--output", required=True, type=Path, help="Output parquet file with high-NULL columns removed."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=90.0,
        help="Drop columns with NULL率 >= threshold (percentage).",
    )
    parser.add_argument(
        "--keep-col",
        action="append",
        default=[],
        help="Column name to always keep (can be specified multiple times).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional JSON report file to write list of dropped columns and null stats.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""

    args = parse_args()
    input_path: Path = args.input
    output_path: Path = args.output
    threshold: float = args.threshold

    if not input_path.exists():
        print(f"❌ Input dataset not found: {input_path}")
        return 1

    print(f"📂 Loading dataset from {input_path}")
    df = pl.read_parquet(str(input_path))
    print(f"   rows={df.height:,}, cols={df.width}")

    keep_cols = set(args.keep_col or []) | REQUIRED_KEEP_COLS
    # Codeが大文字のみの場合に備えて、code列を追加で保持
    if "code" not in df.columns and "Code" in df.columns:
        df = df.with_columns(pl.col("Code").alias("code"))
    if "Date" in df.columns and "date" not in df.columns:
        df = df.with_columns(pl.col("Date").alias("date"))

    to_drop = _find_high_null_columns(df, keep_cols, threshold=threshold)
    if not to_drop:
        print(f"ℹ️  No columns with NULL率 >= {threshold:.2f}% found. Writing dataset unchanged.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(str(output_path), compression="zstd")
        return 0

    drop_names = [name for name, _, _ in to_drop]
    print(f"🧹 Dropping {len(drop_names)} high-NULL columns (threshold={threshold:.2f}%):")
    for name, null_count, rate in sorted(to_drop, key=lambda t: t[2], reverse=True):
        print(f"   - {name}: NULLs={null_count:,} ({rate:.2f}%)")

    df_clean = df.drop(drop_names)
    print(f"   → rows={df_clean.height:,}, cols={df_clean.width}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"💾 Writing cleaned dataset to {output_path}")
    df_clean.write_parquet(str(output_path), compression="zstd")

    if args.report:
        report = {
            "input": str(input_path),
            "output": str(output_path),
            "threshold": threshold,
            "dropped_columns": [
                {"name": name, "null_count": null_count, "null_rate": rate} for name, null_count, rate in to_drop
            ],
            "num_dropped": len(to_drop),
            "rows": df_clean.height,
            "cols": df_clean.width,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w") as f:
            json.dump(report, f, indent=2)
        print(f"📝 Report written to {args.report}")

    print("✅ Completed high-NULL column pruning.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
