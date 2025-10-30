#!/usr/bin/env python3
"""
Prototype tool for evaluating correlation-based graph settings.

Example
-------
python scripts/test_correlation_graph.py \
    --dataset output/ml_dataset_latest_full.parquet \
    --target-column feat_ret_1d \
    --window 60 \
    --knn 20 \
    --threshold 0.3 \
    --output output/reports/correlation_prototype.md
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correlation graph prototype tester.")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the parquet dataset (must contain Code/Date and target column).",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="feat_ret_1d",
        help="Column used to compute correlations (default: feat_ret_1d).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=60,
        help="Number of most recent trading days used in the correlation window.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="pearson",
        choices=("pearson", "spearman"),
        help="Correlation method (pearson or spearman).",
    )
    parser.add_argument(
        "--knn",
        type=int,
        default=20,
        help="Number of nearest neighbours per node.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Correlation threshold; edges below this absolute value are dropped.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the prototype report (Markdown).",
    )
    parser.add_argument(
        "--max-codes",
        type=int,
        default=4000,
        help="Optional cap on the number of stocks (for faster prototyping).",
    )
    return parser.parse_args()


def _select_recent_dates(scan: pl.LazyFrame, window: int) -> list:
    """Select recent dates, preserving their type (Date or String)."""
    dates = (
        scan.select("Date")
        .unique()
        .sort("Date")
        .tail(window)
        .collect()
        .to_series()
        .to_list()
    )
    # Return dates as-is (don't convert to string)
    return dates


def _pivot_returns(
    df: pl.DataFrame,
    target_column: str,
    codes: Iterable[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    if codes is not None:
        df = df.filter(pl.col("Code").is_in(list(codes)))
    pivot = (
        df.pivot(index="Code", on="Date", values=target_column)
        .sort("Code")
    )
    codes_sorted = pivot["Code"].to_list()
    values = pivot.drop("Code").to_numpy()
    # Replace NaNs
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    return values.astype(np.float32), codes_sorted


def _spearman_corr(matrix: np.ndarray) -> np.ndarray:
    # Rank each row independently
    ranked = np.apply_along_axis(_rank_data, axis=1, arr=matrix)
    return np.corrcoef(ranked)


def _rank_data(row: np.ndarray) -> np.ndarray:
    tmp = row.argsort(kind="mergesort")
    ranks = np.empty_like(tmp, dtype=np.float32)
    ranks[tmp] = np.arange(len(row), dtype=np.float32)
    return ranks


def _build_knn_edges(
    corr: np.ndarray,
    codes: list[str],
    k: int,
    threshold: float,
) -> tuple[list[tuple[str, str, float]], dict[str, int]]:
    n = corr.shape[0]
    edges: set[tuple[int, int]] = set()
    weights = {}
    for i in range(n):
        row = corr[i].copy()
        row[i] = -np.inf  # exclude self
        # Select top-k indices
        idx = np.argpartition(row, -k)[-k:]
        idx = idx[np.argsort(row[idx])[::-1]]  # sort descending
        for j in idx:
            w = row[j]
            if not math.isfinite(w) or abs(w) < threshold:
                continue
            a, b = sorted((i, j))
            edges.add((a, b))
            weights[(a, b)] = max(weights.get((a, b), -np.inf), w)

    edge_list = [
        (codes[a], codes[b], float(weights[(a, b)])) for (a, b) in sorted(edges)
    ]

    degree_count = defaultdict(int)
    for a, b, _ in edge_list:
        degree_count[a] += 1
        degree_count[b] += 1

    return edge_list, degree_count


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(2)

    scan = pl.scan_parquet(dataset_path).select(["Code", "Date", args.target_column])
    recent_dates = _select_recent_dates(scan, args.window)
    if not recent_dates:
        raise RuntimeError("No dates found in dataset; check input path and schema.")

    df = (
        scan.filter(pl.col("Date").is_in(recent_dates))
        .filter(pl.col(args.target_column).is_not_null())
        .collect()
    )
    unique_codes = df["Code"].unique().sort().to_list()
    if args.max_codes and len(unique_codes) > args.max_codes:
        unique_codes = unique_codes[: args.max_codes]
        df = df.filter(pl.col("Code").is_in(unique_codes))

    returns_matrix, codes = _pivot_returns(df, args.target_column, unique_codes)
    if returns_matrix.shape[0] == 0 or returns_matrix.shape[1] == 0:
        raise RuntimeError("Insufficient data to compute correlations.")

    if args.method == "pearson":
        corr = np.corrcoef(returns_matrix)
    else:
        corr = _spearman_corr(returns_matrix)

    np.fill_diagonal(corr, 0.0)

    edges, degrees = _build_knn_edges(
        corr=corr,
        codes=codes,
        k=args.knn,
        threshold=args.threshold,
    )

    avg_degree = np.mean(list(degrees.values())) if degrees else 0.0
    std_degree = np.std(list(degrees.values())) if degrees else 0.0
    positive_edges = sum(1 for (_, _, w) in edges if w > 0)
    negative_edges = len(edges) - positive_edges

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        f.write("# Correlation Graph Prototype\n\n")
        f.write(f"- Dataset: `{dataset_path}`\n")
        f.write(f"- Target column: `{args.target_column}`\n")
        f.write(f"- Window: {args.window} trading days ({len(recent_dates)} selected)\n")
        f.write(f"- Method: {args.method}\n")
        f.write(f"- kNN: {args.knn}\n")
        f.write(f"- Threshold: {args.threshold}\n")
        f.write(f"- Codes analysed: {len(codes)}\n")
        f.write(f"- Dates range: {recent_dates[0]} → {recent_dates[-1]}\n\n")

        f.write("## Metrics\n")
        f.write("| Metric | Value |\n")
        f.write("|---|---|\n")
        f.write(f"| Nodes | {len(codes)} |\n")
        f.write(f"| Edges | {len(edges)} |\n")
        f.write(f"| Average degree | {avg_degree:.2f} |\n")
        f.write(f"| Degree stddev | {std_degree:.2f} |\n")
        f.write(f"| Positive edges | {positive_edges} |\n")
        f.write(f"| Negative edges | {negative_edges} |\n")
        f.write(f"| Threshold | {args.threshold} |\n")
        f.write(f"| kNN | {args.knn} |\n\n")

        f.write("## Top edges (sample)\n")
        f.write("| Code A | Code B | Corr |\n")
        f.write("|---|---|---|\n")
        for code_a, code_b, weight in edges[:20]:
            f.write(f"| {code_a} | {code_b} | {weight:.3f} |\n")

    print(
        f"Prototype complete: nodes={len(codes)}, edges={len(edges)}, "
        f"avg_degree={avg_degree:.2f} (report={args.output})"
    )


if __name__ == "__main__":
    main()
