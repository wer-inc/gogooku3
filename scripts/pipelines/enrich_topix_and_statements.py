#!/usr/bin/env python3
"""
Enrich an existing dataset with TOPIX (mkt_* + cross) and statements (stmt_*).

This does NOT fetch from network. It expects Parquet inputs under output/ or paths
provided explicitly via CLI.

Usage examples:
  - Auto-discover latest files under output/:
      python scripts/pipelines/enrich_topix_and_statements.py

  - Explicit paths:
      python scripts/pipelines/enrich_topix_and_statements.py \
        --input output/ml_dataset_latest_full.parquet \
        --topix-parquet output/topix_history_20200903_20250903.parquet \
        --statements-parquet output/event_raw_statements_20200903_20250903.parquet

Writes a new enriched file and updates the ml_dataset_latest_full.parquet symlink.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

# Ensure scripts importable
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gogooku3.pipeline.builder import MLDatasetBuilder


def find_latest(path_glob: str) -> Path | None:
    cands = sorted(Path('output').glob(path_glob))
    return cands[-1] if cands else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Enrich dataset with TOPIX and statements")
    ap.add_argument("--input", type=Path, default=Path("output/ml_dataset_latest_full.parquet"))
    ap.add_argument("--topix-parquet", type=Path, default=None)
    ap.add_argument("--statements-parquet", type=Path, default=None)
    args = ap.parse_args()

    if not args.input.exists():
        print(f"Input dataset not found: {args.input}")
        return 1

    topix_path = args.topix_parquet
    if topix_path is None:
        topix_path = find_latest("topix_history_*.parquet")
    stm_path = args.statements_parquet
    if stm_path is None:
        # Prefer symlink if present
        symlink = Path('output/event_raw_statements.parquet')
        if symlink.exists():
            stm_path = symlink
        else:
            stm_path = find_latest("event_raw_statements_*.parquet")

    if topix_path is None or not topix_path.exists():
        print("TOPIX parquet not found. Provide --topix-parquet or place it under output/.")
        return 1
    if stm_path is None or not stm_path.exists():
        print("Statements parquet not found. Provide --statements-parquet or place it under output/.")
        return 1

    df = pl.read_parquet(args.input)
    topix_df = pl.read_parquet(topix_path)
    stm_df = pl.read_parquet(stm_path)

    builder = MLDatasetBuilder(output_dir=Path('output'))

    # Add TOPIX first
    df = builder.add_topix_features(df, topix_df=topix_df)
    # Then statements
    df = builder.add_statements_features(df, stm_df)

    # Save enriched dataset
    out_dir = Path('output')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    pq = out_dir / f"ml_dataset_{ts}_full_enriched.parquet"
    df.write_parquet(pq)

    # Update symlink to latest full
    latest = out_dir / 'ml_dataset_latest_full.parquet'
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
    except Exception:
        pass
    # Use relative path for symlink
    import os
    if latest.parent == pq.parent:
        latest.symlink_to(pq.name)
    else:
        latest.symlink_to(os.path.relpath(pq, start=latest.parent))

    # Metadata
    meta = builder.create_metadata(df)
    meta_path = out_dir / f"ml_dataset_{ts}_full_enriched_metadata.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, default=str)
    latest_meta = out_dir / 'ml_dataset_latest_full_metadata.json'
    try:
        if latest_meta.exists() or latest_meta.is_symlink():
            latest_meta.unlink()
    except Exception:
        pass
    # Use relative path for symlink
    import os
    if latest_meta.parent == meta_path.parent:
        latest_meta.symlink_to(meta_path.name)
    else:
        latest_meta.symlink_to(os.path.relpath(meta_path, start=latest_meta.parent))

    print("Wrote:", pq)
    print("Metadata:", meta_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
