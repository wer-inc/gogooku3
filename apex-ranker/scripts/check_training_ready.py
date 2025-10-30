#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
from apex_ranker.data import FeatureSelector
from apex_ranker.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("APEX-Ranker readiness check")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the training YAML configuration.",
    )
    return parser.parse_args()


def col_summary(coll):
    preview = ", ".join(sorted(coll)[:5])
    return f"{len(coll)} columns (first: {preview})"


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    parquet_path = Path(data_cfg["parquet_path"])
    if not parquet_path.exists():
        raise SystemExit(f"[ERROR] dataset not found: {parquet_path}")

    # Feature selection (supports optional plus30)
    feature_selector = FeatureSelector(data_cfg["feature_groups_config"])
    groups = list(data_cfg.get("feature_groups", []))
    optional_groups = list(data_cfg.get("optional_groups", []))
    if data_cfg.get("use_plus30"):
        groups = groups + ["plus30"]
    selection = feature_selector.select(
        groups=groups,
        optional_groups=optional_groups,
        metadata_path=data_cfg.get("metadata_path"),
    )

    target_map = data_cfg["target_columns"]
    horizons = [int(h) for h in train_cfg["horizons"]]

    targets: list[str] = []
    for h in horizons:
        key_str = str(h)
        if key_str in target_map:
            targets.append(target_map[key_str])
        elif h in target_map:
            targets.append(target_map[h])
        else:
            raise SystemExit(f"[ERROR] target column missing for horizon {h}")

    required_columns = (
        [data_cfg["date_column"], data_cfg["code_column"]]
        + selection.features
        + selection.masks
        + targets
    )
    required_columns = list(dict.fromkeys(required_columns))

    frame = pl.read_parquet(parquet_path, n_rows=1)
    available_columns = set(frame.columns)

    missing = [col for col in required_columns if col not in available_columns]
    if missing:
        missing_str = ", ".join(missing)
        raise SystemExit(f"[ERROR] dataset is missing required columns: {missing_str}")

    scan = pl.scan_parquet(parquet_path)
    distinct_dates = scan.select(pl.col(data_cfg["date_column"]).n_unique()).collect().item()
    lookback = data_cfg["lookback"]
    if distinct_dates <= lookback:
        raise SystemExit(
            f"[ERROR] dataset has only {distinct_dates} unique dates "
            f"(need > lookback={lookback})"
        )

    if selection.masks:
        coverage_exprs = [
            pl.col(mask).fill_null(0).gt(0.5).mean().alias(mask) for mask in selection.masks
        ]
        coverage = scan.select(coverage_exprs).collect()
        for mask in selection.masks:
            ratio = float(coverage[0, mask])
            if ratio == 0.0:
                print(f"[WARN] Mask '{mask}' has zero positive coverage; it will be ignored.")
            else:
                pct = ratio * 100.0
                print(f"[OK] Mask '{mask}' coverage: {pct:.1f}%")

    print("[OK] dataset located:", parquet_path)
    print("[OK] feature groups:", groups, "(+ optional:", optional_groups, ")")
    print("[OK] features:", col_summary(selection.features))
    print("[OK] mask columns:", col_summary(selection.masks))
    print("[OK] target columns:", ", ".join(targets))
    print(f"[OK] unique dates: {distinct_dates} (lookback={lookback})")
    print("[READY] training can be launched with the specified configuration.")


if __name__ == "__main__":
    main()
