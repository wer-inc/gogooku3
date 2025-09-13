#!/usr/bin/env python3
"""
Dataset sanity checker (optional):

- If a dataset file is provided via DATASET_PATH or found via DATASET_GLOB,
  load it with Polars and validate minimal schema/quality:
  - Presence of id columns: date/Date and code/Code (case-insensitive)
  - Presence of at least one target column among common variants
    (returns_1d, ret_1d, feat_ret_1d, target_1d, etc.)
  - No nulls in id columns
  - No duplicate (date, code)

- If no dataset is found and ALLOW_MISSING=1 (default in CI), exit 0.
  Otherwise, exit 0 with a note (non-blocking by default).
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path
from typing import Iterable

import polars as pl


TARGET_CANDIDATES = [
    # common variants
    "returns_1d", "returns_5d", "returns_10d", "returns_20d",
    "ret_1d", "ret_5d", "ret_10d", "ret_20d",
    "feat_ret_1d", "feat_ret_5d", "feat_ret_10d", "feat_ret_20d",
    "target_1d", "target_5d", "target_10d", "target_20d",
]


def find_dataset() -> Path | None:
    env_path = os.getenv("DATASET_PATH")
    if env_path:
        p = Path(env_path)
        return p if p.exists() else None
    glob_pat = os.getenv("DATASET_GLOB", "output/*.parquet")
    cands = sorted(glob.glob(glob_pat))
    return Path(cands[0]) if cands else None


def main() -> int:
    ds = find_dataset()
    allow_missing = os.getenv("ALLOW_MISSING", "1") == "1"
    if not ds:
        print("[dataset-sanity] No dataset found.")
        if allow_missing:
            print("[dataset-sanity] ALLOW_MISSING=1; skipping check.")
            return 0
        return 0

    print(f"[dataset-sanity] Checking: {ds}")
    df = pl.read_parquet(ds)
    cols = set(df.columns)

    # id columns
    date_col = "date" if "date" in cols else ("Date" if "Date" in cols else None)
    code_col = "code" if "code" in cols else ("Code" if "Code" in cols else None)
    if not date_col or not code_col:
        print(f"[dataset-sanity] Missing id columns: date/date or code/Code not found.")
        return 1

    # target columns
    present_targets = [c for c in TARGET_CANDIDATES if c in cols]
    if not present_targets:
        print("[dataset-sanity] No known target columns found (returns_1d/ret_1d/feat_ret_1d/target_1d, etc.)")
        # non-blocking: just warn
    else:
        print(f"[dataset-sanity] Found targets: {present_targets[:6]}{'...' if len(present_targets)>6 else ''}")

    # nulls in id columns
    id_nulls = df.select(pl.col(date_col).is_null().sum().alias("date_nulls"), pl.col(code_col).is_null().sum().alias("code_nulls"))
    dn, cn = id_nulls.to_dicts()[0]["date_nulls"], id_nulls.to_dicts()[0]["code_nulls"]
    if dn or cn:
        print(f"[dataset-sanity] Nulls in id columns: date={dn}, code={cn}")
        return 1

    # duplicates
    dup = (
        df.group_by([date_col, code_col])
        .len()
        .filter(pl.col("len") > 1)
    )
    dup_count = len(dup)
    if dup_count:
        print(f"[dataset-sanity] Duplicate (date, code) pairs: {dup_count}")
        return 1

    print("[dataset-sanity] OK: id columns present, no nulls, no duplicates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

