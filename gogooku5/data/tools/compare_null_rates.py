#!/usr/bin/env python3
"""Compare NULL rates between two parquet datasets and emit a Markdown report.

This script is intended for before/after analysis when tuning feature
generation or pruning high-NULL columns.

Example:

.. code-block:: bash

    PYTHONPATH=gogooku5/data/src \\
      python gogooku5/data/tools/compare_null_rates.py \\
        --before data/output/datasets/ml_dataset_2023_2025_final_pruned.parquet \\
        --after  data/output/datasets/ml_dataset_2023_2024_clean.parquet \\
        --output gogooku5/docs/NULL_RATE_IMPROVEMENT_REPORT.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import polars as pl


def _null_stats(path: Path) -> Tuple[int, int, Dict[str, float]]:
    """Return (rows, cols, null_rate_map) for the given parquet dataset."""

    df = pl.read_parquet(str(path))
    rows = df.height
    cols = df.width
    null_counts_df = df.null_count()
    null_counts_dict = null_counts_df.to_dict(as_series=False)
    rates: Dict[str, float] = {}
    for name, null_list in null_counts_dict.items():
        null_count = null_list[0]
        rate = (null_count / rows) * 100.0 if rows > 0 else 0.0
        rates[name] = rate
    return rows, cols, rates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare NULL rates between two parquet datasets and emit a Markdown report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--before",
        required=True,
        type=Path,
        help="Baseline parquet file (before changes).",
    )
    parser.add_argument(
        "--after",
        required=True,
        type=Path,
        help="Parquet file after changes.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output Markdown report path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    before_path: Path = args.before
    after_path: Path = args.after
    out_path: Path = args.output

    if not before_path.exists():
        print(f"❌ Baseline dataset not found: {before_path}")
        return 1
    if not after_path.exists():
        print(f"❌ After dataset not found: {after_path}")
        return 1

    print(f"📂 Loading baseline: {before_path}")
    rows_before, cols_before, rates_before = _null_stats(before_path)
    print(f"   rows={rows_before:,}, cols={cols_before}")

    print(f"📂 Loading after:   {after_path}")
    rows_after, cols_after, rates_after = _null_stats(after_path)
    print(f"   rows={rows_after:,}, cols={cols_after}")

    # Compute average NULL率（列単位）: intersection上で比較
    common_cols = sorted(set(rates_before.keys()) & set(rates_after.keys()))
    if common_cols:
        avg_before = sum(rates_before[c] for c in common_cols) / len(common_cols)
        avg_after = sum(rates_after[c] for c in common_cols) / len(common_cols)
    else:
        avg_before = avg_after = 0.0

    # Track a few key features if present
    focus = [
        "ret_prev_60d",
        "ret_prev_120d",
        "fs_sales_yoy",
        "fs_yoy_ttm_sales",
        "fs_yoy_ttm_net_income",
        "div_ex_gap_miss",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# NULL率改善レポート\n\n")
        f.write(f"- Baseline: `{before_path}`\n")
        f.write(f"- After:    `{after_path}`\n\n")

        f.write("## サマリ\n\n")
        f.write("| 指標 | Before | After |\n")
        f.write("|------|--------|-------|\n")
        f.write(f"| 行数 | {rows_before:,} | {rows_after:,} |\n")
        f.write(f"| 列数 | {cols_before} | {cols_after} |\n")
        f.write(f"| 平均NULL率 (共通列) | {avg_before:.2f}% | {avg_after:.2f}% |\n\n")

        f.write("## 主要特徴量のNULL率比較\n\n")
        f.write("| 特徴量 | Before | After | 差分 (After-Before) |\n")
        f.write("|--------|--------|-------|---------------------|\n")
        for col in focus:
            b = rates_before.get(col)
            a = rates_after.get(col)
            if b is None and a is None:
                continue
            b_str = f"{b:.2f}%" if b is not None else "N/A"
            a_str = f"{a:.2f}%" if a is not None else "N/A"
            diff = ""
            if b is not None and a is not None:
                diff = f"{a - b:+.2f}%"
            f.write(f"| `{col}` | {b_str} | {a_str} | {diff} |\n")

        # Top-k improvements
        improvements = []
        for col in common_cols:
            diff = rates_after[col] - rates_before[col]
            if diff < -5.0:  # 5%ポイント以上改善したものだけ
                improvements.append((col, rates_before[col], rates_after[col], diff))

        improvements.sort(key=lambda t: t[3])  # diff昇順（大きくマイナス = 改善）

        if improvements:
            f.write("\n## NULL率が大きく改善した特徴量 (上位20件)\n\n")
            f.write("| 特徴量 | Before | After | 差分 (After-Before) |\n")
            f.write("|--------|--------|-------|---------------------|\n")
            for name, b_rate, a_rate, diff in improvements[:20]:
                f.write(f"| `{name}` | {b_rate:.2f}% | {a_rate:.2f}% | {diff:+.2f}% |\n")

    print(f"📝 NULL率比較レポートを書き出しました: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
