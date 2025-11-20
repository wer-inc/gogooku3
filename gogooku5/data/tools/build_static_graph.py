#!/usr/bin/env python3
"""Build a static graph of stock relationships based on sector and beta similarity.

This script generates a static edge list connecting stocks (SecId nodes) that share the
same sector and have similar 60-day beta values. It selects the top-k nearest neighbors
(by absolute beta difference) for each stock within its sector and computes edge weights
using an exponential decay based on beta difference.

Edges:
    - Nodes: ``SecId`` (stock identifiers)
    - Edges: Connect stocks within the same ``SectorCode``. Each stock links to ``k``
      nearest stocks in terms of ``beta60_topix`` value (within its sector).
    - Weight: ``w_ij = exp(-|beta_i - beta_j| / tau)``, where ``tau`` is a scaling
      parameter.

Output:
    - ``edges_static.parquet`` with columns ``[src_sec_id, dst_sec_id, weight]``.

Example
-------

.. code-block:: bash

    PYTHONPATH=gogooku5/data/src \\
      python gogooku5/data/tools/build_static_graph.py \\
        --input output_g5/datasets/ml_dataset_latest.parquet \\
        --output output_g5/datasets/edges_static.parquet \\
        --k 5 --tau 0.2
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the static graph builder."""

    parser = argparse.ArgumentParser(
        description="Build a static sector+beta graph and output edges as Parquet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output_g5/datasets/ml_dataset_latest.parquet"),
        help=("Input Parquet dataset (merged ML dataset with SecId, SectorCode, " "beta60_topix)."),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output_g5/datasets/edges_static.parquet"),
        help="Output Parquet file to save the static graph edges.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of nearest neighbors (within the same sector) to connect for each node.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.2,
        help="Tau parameter for weight calculation (scales the beta difference in the exponential).",
    )
    return parser.parse_args()


def _mode(values: Iterable[str]) -> str | None:
    """Return the most frequent value in an iterable (or None if empty)."""

    seq = list(values)
    if not seq:
        return None
    return Counter(seq).most_common(1)[0][0]


def main() -> int:
    """Entry point: build static sector+beta graph edges."""

    args = parse_args()
    input_path: Path = args.input
    output_path: Path = args.output
    k: int = args.k
    tau: float = args.tau

    if not input_path.exists():
        print(f"❌ Input dataset not found: {input_path}")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load required columns from the dataset.
    print(f"📂 Loading dataset: {input_path}")
    try:
        df = pl.read_parquet(
            str(input_path),
            columns=["SecId", "SectorCode", "beta60_topix"],
        )
    except Exception as exc:  # pragma: no cover - defensive
        print(f"❌ Failed to read parquet: {exc}")
        return 1

    if df.is_empty():
        print("⚠️  Input dataset is empty; no edges will be created.")
        pl.DataFrame({"src_sec_id": [], "dst_sec_id": [], "weight": []}).write_parquet(
            str(output_path),
            compression="zstd",
        )
        print(f"✅ Empty edges file written to: {output_path}")
        return 0

    # Filter out nulls in key columns.
    df = df.filter(
        pl.col("SecId").is_not_null() & pl.col("SectorCode").is_not_null() & pl.col("beta60_topix").is_not_null()
    )
    if df.is_empty():
        print("⚠️  No rows with non-null SecId/SectorCode/beta60_topix; no edges created.")
        pl.DataFrame({"src_sec_id": [], "dst_sec_id": [], "weight": []}).write_parquet(
            str(output_path),
            compression="zstd",
        )
        print(f"✅ Empty edges file written to: {output_path}")
        return 0

    # Ensure SecId is integer type for safer processing.
    df = df.with_columns(pl.col("SecId").cast(pl.Int32))

    # 2. Aggregate SectorCode and beta per SecId.
    # Compute the most frequent SectorCode (mode) and median beta60_topix for each SecId.
    print("📊 Aggregating sector and beta per SecId...")
    secid_stats = (
        df.group_by("SecId")
        .agg(
            [
                pl.col("SectorCode").list().alias("sector_list"),
                pl.col("beta60_topix").list().alias("beta_list"),
            ]
        )
        .with_columns(
            [
                pl.col("sector_list").map_elements(_mode, return_dtype=pl.Utf8).alias("SectorCode"),
                pl.col("beta_list")
                .map_elements(
                    lambda bl: float(np.nanmedian(bl)) if bl else None,  # type: ignore[arg-type]
                    return_dtype=pl.Float64,
                )
                .alias("beta"),
            ]
        )
        .select(["SecId", "SectorCode", "beta"])
        .drop_nulls(["SectorCode", "beta"])
    )

    if secid_stats.is_empty():
        print("⚠️  No SecId with valid sector/beta; writing empty edges file.")
        pl.DataFrame({"src_sec_id": [], "dst_sec_id": [], "weight": []}).write_parquet(
            str(output_path),
            compression="zstd",
        )
        print(f"✅ Empty edges file written to: {output_path}")
        return 0

    # 3. Build edges within each sector group based on beta proximity.
    print("🧮 Building sector-wise neighbor edges...")
    edges: list[tuple[int, int, float]] = []

    sector_groups = (
        secid_stats.group_by("SectorCode")
        .agg(
            [
                pl.col("SecId").list().alias("secids"),
                pl.col("beta").list().alias("betas"),
            ]
        )
        .collect()
    )

    for sector, secid_list, beta_list in zip(
        sector_groups["SectorCode"],
        sector_groups["secids"],
        sector_groups["betas"],
    ):
        secids = list(secid_list)
        betas = list(beta_list)
        if len(secids) < 2:
            continue  # No edges if only one stock in this sector.

        # Sort stocks in this sector by beta value.
        sorted_pairs = sorted(zip(betas, secids), key=lambda x: x[0])
        sorted_betas = [pair[0] for pair in sorted_pairs]
        sorted_secids = [pair[1] for pair in sorted_pairs]
        n = len(sorted_secids)

        # For each stock in the sector, select up to k nearest neighbors by beta difference.
        for idx in range(n):
            src_sec = int(sorted_secids[idx])
            beta_i = sorted_betas[idx]
            left = idx - 1
            right = idx + 1
            neighbors_added = 0

            # Expand outwards from the stock in both directions (lower and higher beta).
            while neighbors_added < k:
                if left >= 0 and right < n:
                    # Both sides available: pick the closer beta neighbor.
                    dist_left = abs(beta_i - sorted_betas[left])
                    dist_right = abs(beta_i - sorted_betas[right])
                    if dist_left <= dist_right:
                        neighbor_idx = left
                        left -= 1
                    else:
                        neighbor_idx = right
                        right += 1
                elif left >= 0:
                    neighbor_idx = left
                    left -= 1
                elif right < n:
                    neighbor_idx = right
                    right += 1
                else:
                    break  # No more neighbors.

                if neighbor_idx == idx:
                    continue  # Safety: skip self.

                dst_sec = int(sorted_secids[neighbor_idx])
                beta_j = sorted_betas[neighbor_idx]

                # Compute weight = exp(-|beta_i - beta_j| / tau).
                weight_ij = math.exp(-abs(beta_i - beta_j) / tau) if tau > 0 else 1.0
                edges.append((src_sec, dst_sec, weight_ij))
                neighbors_added += 1

    # 4. Create a DataFrame of edges and write to Parquet.
    if edges:
        src_ids, dst_ids, weights = zip(*edges)
    else:
        src_ids, dst_ids, weights = [], [], []

    edges_df = pl.DataFrame(
        {
            "src_sec_id": src_ids,
            "dst_sec_id": dst_ids,
            "weight": weights,
        }
    ).sort(["src_sec_id", "dst_sec_id"])

    print(f"💾 Writing {edges_df.height} edges to {output_path}")
    edges_df.write_parquet(str(output_path), compression="zstd")
    print("✅ Static graph edges build complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
