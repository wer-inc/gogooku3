#!/usr/bin/env python3
"""Diagnose feature health: missing rates, outliers, distribution stats."""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl


def main():
    parser = argparse.ArgumentParser(description="Feature health diagnosis")
    parser.add_argument("--data", type=Path, required=True, help="Parquet dataset")
    parser.add_argument(
        "--features", type=Path, required=True, help="Feature names JSON"
    )
    parser.add_argument("--output", type=Path, required=True, help="Output CSV")
    parser.add_argument("--z-threshold", type=float, default=5.0, help="Z-score threshold")

    args = parser.parse_args()

    # Load feature names
    with open(args.features) as f:
        feature_data = json.load(f)
        if isinstance(feature_data, dict):
            features = feature_data.get("feature_names", feature_data.get("features", []))
        else:
            features = feature_data

    print(f"Checking health for {len(features)} features")

    # Load data
    df = pl.read_parquet(args.data)
    total_rows = len(df)
    print(f"Total rows: {total_rows:,}")

    # Check each feature
    results = []
    for feat in features:
        if feat not in df.columns:
            print(f"⚠️  Feature not in dataset: {feat}")
            continue

        col = df[feat]

        # Missing rate
        missing_count = col.null_count()
        missing_rate = missing_count / total_rows

        # Filter out nulls for stats
        valid = col.drop_nulls()
        if len(valid) == 0:
            results.append({
                "feature": feat,
                "missing_rate": missing_rate,
                "gt_z_threshold_rate": np.nan,
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "q25": np.nan,
                "q50": np.nan,
                "q75": np.nan,
            })
            continue

        # Stats
        mean_val = valid.mean()
        std_val = valid.std()

        # Outliers (|z| > threshold)
        if std_val and std_val > 1e-9:
            z_scores = (valid - mean_val) / std_val
            outlier_count = (z_scores.abs() > args.z_threshold).sum()
            outlier_rate = outlier_count / len(valid)
        else:
            outlier_rate = 0.0

        # Quantiles
        quantiles = valid.quantile([0.25, 0.5, 0.75], interpolation="linear")

        results.append({
            "feature": feat,
            "missing_rate": missing_rate,
            "gt_z_threshold_rate": outlier_rate,
            "mean": mean_val,
            "std": std_val,
            "min": valid.min(),
            "max": valid.max(),
            "q25": quantiles[0],
            "q50": quantiles[1],
            "q75": quantiles[2],
        })

    # Convert to DataFrame and save
    results_df = pl.DataFrame(results)

    # Sort by missing_rate descending (most problematic first)
    results_df = results_df.sort("missing_rate", descending=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results_df.write_csv(args.output)

    print(f"\n✅ Feature health saved to: {args.output}")
    print("\nTop 5 features with highest missing rates:")
    print(results_df.select(["feature", "missing_rate"]).head(5))

    print("\nTop 5 features with highest outlier rates:")
    print(results_df.sort("gt_z_threshold_rate", descending=True).select(["feature", "gt_z_threshold_rate"]).head(5))


if __name__ == "__main__":
    main()
