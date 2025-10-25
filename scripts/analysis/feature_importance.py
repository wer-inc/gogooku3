#!/usr/bin/env python3
"""
Feature Importance Analysis

Purpose: Analyze feature importance from LightGBM model
Output: Top-K features for downstream models
"""

import argparse
import json
import logging
from pathlib import Path

import lightgbm as lgb
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Feature Importance")
    parser.add_argument(
        "--model-path",
        type=str,
        default="output/baselines/lgbm_baseline.txt",
        help="Path to LightGBM model",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of top features to select",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/analysis",
        help="Output directory",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Feature Importance Analysis")
    logger.info("=" * 80)

    # Load model
    logger.info(f"Loading model from: {args.model_path}")
    model = lgb.Booster(model_file=args.model_path)

    # Get feature importance
    logger.info("Extracting feature importance...")
    feature_names = model.feature_name()
    importance_gain = model.feature_importance(importance_type="gain")
    importance_split = model.feature_importance(importance_type="split")

    # Create DataFrame
    df_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_gain": importance_gain,
            "importance_split": importance_split,
        }
    )

    # Sort by gain
    df_importance = df_importance.sort_values(
        "importance_gain", ascending=False
    ).reset_index(drop=True)

    # Add rank
    df_importance["rank"] = range(1, len(df_importance) + 1)

    # Calculate cumulative importance
    total_gain = df_importance["importance_gain"].sum()
    df_importance["cumulative_gain"] = df_importance["importance_gain"].cumsum()
    df_importance["cumulative_gain_pct"] = (
        df_importance["cumulative_gain"] / total_gain * 100
    ).round(2)

    # Log statistics
    logger.info(f"\nTotal features: {len(df_importance)}")
    logger.info(
        f"Top 10 features account for: {df_importance.iloc[:10]['cumulative_gain_pct'].iloc[-1]:.2f}% of total gain"
    )
    logger.info(
        f"Top 50 features account for: {df_importance.iloc[:50]['cumulative_gain_pct'].iloc[-1]:.2f}% of total gain"
    )
    logger.info(
        f"Top 100 features account for: {df_importance.iloc[:min(100, len(df_importance))-1]['cumulative_gain_pct'].iloc[-1]:.2f}% of total gain"
    )

    # Display top features
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Top {min(20, len(df_importance))} Features")
    logger.info(f"{'=' * 80}")
    for _, row in df_importance.head(20).iterrows():
        logger.info(
            f"{row['rank']:3d}. {row['feature']:40s} | "
            f"Gain: {row['importance_gain']:12,.2f} | "
            f"Split: {row['importance_split']:8,.0f} | "
            f"Cum%: {row['cumulative_gain_pct']:6.2f}%"
        )

    # Extract top-K features
    top_k_features = df_importance.head(args.top_k)["feature"].tolist()

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Top {args.top_k} Features Selected")
    logger.info(f"{'=' * 80}")
    logger.info(
        f"These {args.top_k} features account for {df_importance.iloc[args.top_k-1]['cumulative_gain_pct']:.2f}% of total gain"
    )

    # Save full importance
    csv_path = output_dir / "feature_importance_full.csv"
    df_importance.to_csv(csv_path, index=False)
    logger.info(f"\nFull importance saved: {csv_path}")

    # Save top-K features
    top_k_path = output_dir / f"top_{args.top_k}_features.json"
    with open(top_k_path, "w") as f:
        json.dump(
            {
                "top_k": args.top_k,
                "features": top_k_features,
                "total_features": len(feature_names),
                "coverage_pct": float(
                    df_importance.iloc[args.top_k - 1]["cumulative_gain_pct"]
                ),
            },
            f,
            indent=2,
        )
    logger.info(f"Top-{args.top_k} features saved: {top_k_path}")

    # Save top-K CSV
    top_k_csv_path = output_dir / f"top_{args.top_k}_features.csv"
    df_importance.head(args.top_k).to_csv(top_k_csv_path, index=False)
    logger.info(f"Top-{args.top_k} CSV saved: {top_k_csv_path}")

    # Category analysis (if feature names have prefixes)
    logger.info(f"\n{'=' * 80}")
    logger.info("Feature Category Analysis")
    logger.info(f"{'=' * 80}")

    # Extract prefixes
    prefixes = []
    for feat in feature_names:
        # Common prefixes: ret_, stmt_, margin_, flow_, graph_, sec_, x_, etc.
        if "_" in feat:
            prefix = feat.split("_")[0]
        else:
            prefix = "other"
        prefixes.append(prefix)

    df_importance["category"] = prefixes

    # Group by category
    category_stats = (
        df_importance.groupby("category")
        .agg(
            {
                "importance_gain": ["sum", "mean", "count"],
            }
        )
        .sort_values(("importance_gain", "sum"), ascending=False)
    )

    logger.info("\nTop Categories by Total Gain:")
    for idx, (category, row) in enumerate(category_stats.head(15).iterrows(), 1):
        total_gain = row[("importance_gain", "sum")]
        avg_gain = row[("importance_gain", "mean")]
        count = row[("importance_gain", "count")]
        pct = total_gain / df_importance["importance_gain"].sum() * 100
        logger.info(
            f"{idx:2d}. {category:20s} | "
            f"Total: {total_gain:12,.2f} ({pct:5.2f}%) | "
            f"Avg: {avg_gain:10,.2f} | "
            f"Count: {count:4.0f}"
        )

    # Save category stats
    category_path = output_dir / "feature_category_importance.csv"
    category_stats.to_csv(category_path)
    logger.info(f"\nCategory stats saved: {category_path}")

    logger.info("\n🎉 Feature importance analysis complete!")


if __name__ == "__main__":
    main()
