#!/usr/bin/env python3
"""
Granger Causality Test for Features

Purpose: Test if features have causal relationship with targets
Output: Features with statistically significant causality (p < 0.05)

Note: Granger causality tests if past values of X help predict future Y
"""

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from statsmodels.tsa.stattools import grangercausalitytests

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_granger_causality(
    feature_series: np.ndarray,
    target_series: np.ndarray,
    max_lag: int = 5,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Test Granger causality between feature and target

    Returns:
        dict with test results and p-values
    """
    # Remove NaN
    mask = ~(np.isnan(feature_series) | np.isnan(target_series))
    feature_clean = feature_series[mask]
    target_clean = target_series[mask]

    if len(feature_clean) < max_lag + 10:
        return {
            "causal": False,
            "min_p_value": 1.0,
            "best_lag": 0,
            "reason": "insufficient_data",
        }

    # Check variance
    if np.std(feature_clean) < 1e-8 or np.std(target_clean) < 1e-8:
        return {
            "causal": False,
            "min_p_value": 1.0,
            "best_lag": 0,
            "reason": "zero_variance",
        }

    try:
        # Prepare data (target, feature)
        data = np.column_stack([target_clean, feature_clean])

        # Run test
        results = grangercausalitytests(data, maxlag=max_lag, verbose=False)

        # Extract p-values (using F-test)
        p_values = []
        for lag in range(1, max_lag + 1):
            # F-test p-value
            p_value = results[lag][0]["ssr_ftest"][1]
            p_values.append(p_value)

        min_p_value = min(p_values)
        best_lag = p_values.index(min_p_value) + 1

        is_causal = min_p_value < alpha

        return {
            "causal": is_causal,
            "min_p_value": float(min_p_value),
            "best_lag": int(best_lag),
            "reason": "tested",
        }

    except Exception as e:
        return {
            "causal": False,
            "min_p_value": 1.0,
            "best_lag": 0,
            "reason": f"error: {str(e)[:100]}",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Granger Causality Test")
    parser.add_argument(
        "--data-path",
        type=str,
        default="output/ml_dataset_latest_full.parquet",
        help="Path to dataset",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="target_5d",
        help="Target column",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=5,
        help="Maximum lag to test",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level",
    )
    parser.add_argument(
        "--sample-codes",
        type=int,
        default=100,
        help="Number of stock codes to sample (0=all, takes long time)",
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
    logger.info("Granger Causality Test")
    logger.info("=" * 80)
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Target: {args.target}")
    logger.info(f"Max lag: {args.max_lag}")
    logger.info(f"Alpha: {args.alpha}")
    logger.info(
        f"Sample codes: {args.sample_codes if args.sample_codes > 0 else 'All'}"
    )

    # Load dataset
    logger.info("\nLoading dataset...")
    df = pl.read_parquet(args.data_path)
    logger.info(f"Dataset shape: {df.shape}")

    # Define feature columns
    exclude_cols = [
        "Code",
        "Date",
        "Section",
        "MarketCode",
        "LocalCode",
        "CompanyName",
        "row_idx",
        "sector17_code",
        "sector17_name",
        "sector17_id",
        "sector33_code",
        "sector33_name",
        "sector33_id",
        "target_1d",
        "target_5d",
        "target_10d",
        "target_20d",
        "target_1d_binary",
        "target_5d_binary",
        "target_10d_binary",
        "target_20d_binary",
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]
    logger.info(f"Feature columns: {len(feature_cols)}")

    # Sample codes if specified
    if args.sample_codes > 0:
        all_codes = df["Code"].unique().to_list()
        np.random.seed(42)
        sample_codes = np.random.choice(
            all_codes, min(args.sample_codes, len(all_codes)), replace=False
        )
        df = df.filter(pl.col("Code").is_in(sample_codes))
        logger.info(f"Sampled {len(sample_codes)} codes for testing")
        logger.info(f"Reduced dataset shape: {df.shape}")

    # Test each feature
    logger.info(f"\nTesting {len(feature_cols)} features for Granger causality...")
    logger.info("This may take a while...")

    results = []

    for idx, feature in enumerate(feature_cols, 1):
        if idx % 50 == 0:
            logger.info(f"Progress: {idx}/{len(feature_cols)} features tested")

        # Get feature and target series
        feature_series = df.select(feature).to_numpy().flatten()
        target_series = df.select(args.target).to_numpy().flatten()

        # Test causality
        test_result = test_granger_causality(
            feature_series,
            target_series,
            max_lag=args.max_lag,
            alpha=args.alpha,
        )

        results.append(
            {
                "feature": feature,
                "causal": test_result["causal"],
                "p_value": test_result["min_p_value"],
                "best_lag": test_result["best_lag"],
                "reason": test_result["reason"],
            }
        )

    # Filter causal features
    causal_features = [r for r in results if r["causal"]]
    non_causal_features = [r for r in results if not r["causal"]]

    logger.info(f"\n{'=' * 80}")
    logger.info("Results Summary")
    logger.info(f"{'=' * 80}")
    logger.info(f"Total features tested: {len(results)}")
    logger.info(
        f"Causal features (p < {args.alpha}): {len(causal_features)} ({len(causal_features)/len(results)*100:.1f}%)"
    )
    logger.info(
        f"Non-causal features: {len(non_causal_features)} ({len(non_causal_features)/len(results)*100:.1f}%)"
    )

    # Display top causal features
    if causal_features:
        logger.info(f"\n{'=' * 80}")
        logger.info("Top 30 Causal Features (sorted by p-value)")
        logger.info(f"{'=' * 80}")

        causal_sorted = sorted(causal_features, key=lambda x: x["p_value"])

        for idx, feat in enumerate(causal_sorted[:30], 1):
            logger.info(
                f"{idx:3d}. {feat['feature']:40s} | "
                f"p-value: {feat['p_value']:.6f} | "
                f"Best lag: {feat['best_lag']}"
            )

    # Save results
    result_summary = {
        "config": {
            "data_path": args.data_path,
            "target": args.target,
            "max_lag": args.max_lag,
            "alpha": args.alpha,
            "sample_codes": args.sample_codes,
        },
        "summary": {
            "total_features": len(results),
            "causal_features": len(causal_features),
            "non_causal_features": len(non_causal_features),
            "causal_ratio": len(causal_features) / len(results),
        },
        "causal_features": sorted(causal_features, key=lambda x: x["p_value"]),
        "non_causal_features": non_causal_features,
    }

    result_path = output_dir / "granger_causality_results.json"
    with open(result_path, "w") as f:
        json.dump(result_summary, f, indent=2)
    logger.info(f"\nResults saved: {result_path}")

    # Save causal features list
    causal_list_path = output_dir / "causal_features.json"
    with open(causal_list_path, "w") as f:
        json.dump(
            {
                "features": [
                    f["feature"]
                    for f in sorted(causal_features, key=lambda x: x["p_value"])
                ],
                "count": len(causal_features),
                "alpha": args.alpha,
            },
            f,
            indent=2,
        )
    logger.info(f"Causal features list saved: {causal_list_path}")

    # Interpretation
    logger.info(f"\n{'=' * 80}")
    logger.info("🎯 Interpretation")
    logger.info(f"{'=' * 80}")

    if len(causal_features) >= 50:
        logger.info(
            f"✅ Found {len(causal_features)} causal features - Good predictive power!"
        )
        logger.info("   Recommendation: Use these features for modeling")
    elif len(causal_features) >= 20:
        logger.info(f"⚠️  Found {len(causal_features)} causal features - Moderate")
        logger.info("   Recommendation: Consider feature engineering to increase count")
    else:
        logger.info(f"❌ Found only {len(causal_features)} causal features - Low")
        logger.info("   Recommendation: Fundamental feature engineering needed")

    logger.info("\n🎉 Granger causality test complete!")


if __name__ == "__main__":
    main()
