#!/usr/bin/env python3
"""Residual Method Analysis for Feature Contribution Testing.

This script evaluates whether additional features (X_add) provide independent
predictive contribution beyond core features (X_core) using:
1. Ridge regression residualization: X_add⊥ = X_add - Ridge(X_add ~ X_core)
2. Conditional P@K evaluation: Does X_add⊥ improve P@K given X_core?
3. Diebold-Mariano test: Statistical significance of improvement
4. Bootstrap CI: Confidence intervals for contribution estimates

Usage:
    python apex-ranker/scripts/residual_analysis.py \
      --data output/ml_dataset_latest_clean_with_adv.parquet \
      --features-core apex-ranker/configs/feature_names_core64.json \
      --features-add apex-ranker/configs/feature_names_add25.json \
      --output results/residual_contrib.csv \
      --alpha 0.05 \
      --n-bootstrap 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from sklearn.linear_model import Ridge


def diebold_mariano_test(
    errors1: np.ndarray, errors2: np.ndarray
) -> tuple[float, float]:
    """Diebold-Mariano test for predictive accuracy.

    H0: errors1 and errors2 have equal predictive accuracy
    H1: errors1 is more accurate (smaller error)

    Args:
        errors1: Error series for model 1
        errors2: Error series for model 2

    Returns:
        Tuple of (dm_stat, p_value):
            - dm_stat: DM statistic (>1.96 rejects H0 at 5% significance)
            - p_value: One-sided p-value
    """
    d = errors1**2 - errors2**2  # Loss differential
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)
    n = len(d)

    # Adjust for autocorrelation (Newey-West with lag=5)
    gamma = [
        np.mean((d[i:] - d_mean) * (d[:-i] - d_mean))
        for i in range(1, min(6, n // 2))
    ]
    var_adjusted = d_var + 2 * sum(gamma)

    dm_stat = d_mean / np.sqrt(var_adjusted / n)
    p_value = 1 - stats.norm.cdf(dm_stat)

    return dm_stat, p_value


def bootstrap_ci(
    metric_values: np.ndarray, n_bootstrap: int = 1000, alpha: float = 0.05
) -> tuple[float, float]:
    """Bootstrap 95% CI for metric.

    Args:
        metric_values: Array of metric values
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        Tuple of (lower, upper) confidence interval bounds
    """
    bootstrap_samples = np.random.choice(
        metric_values, size=(n_bootstrap, len(metric_values)), replace=True
    )
    bootstrap_means = bootstrap_samples.mean(axis=1)

    lower = np.percentile(bootstrap_means, alpha / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

    return lower, upper


def residualize_features(
    X_core: np.ndarray, X_add: np.ndarray, alpha: float = 1.0
) -> np.ndarray:
    """Residualize X_add against X_core using Ridge regression.

    Args:
        X_core: Core features (n_samples, n_core_features)
        X_add: Additional features (n_samples, n_add_features)
        alpha: Ridge regularization parameter

    Returns:
        Residualized features X_add⊥ (n_samples, n_add_features)
    """
    # Handle NaN values - fill with column median
    # (Ridge doesn't accept NaN natively)
    from sklearn.impute import SimpleImputer

    imputer_core = SimpleImputer(strategy='median')
    imputer_add = SimpleImputer(strategy='median')

    X_core_filled = imputer_core.fit_transform(X_core)
    X_add_filled = imputer_add.fit_transform(X_add)

    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(X_core_filled, X_add_filled)
    X_add_pred = ridge.predict(X_core_filled)
    X_add_resid = X_add_filled - X_add_pred
    return X_add_resid


def compute_precision_at_k(
    scores: np.ndarray, targets: np.ndarray, k: int
) -> float:
    """Compute Precision@K for ranking task.

    Args:
        scores: Predicted scores (higher = better)
        targets: Target values (higher = positive)
        k: Number of top predictions to evaluate

    Returns:
        Precision@K: Fraction of top-k with positive target
    """
    if len(scores) < k:
        return 0.0

    top_k_idx = np.argsort(-scores)[:k]  # Descending order
    top_k_targets = targets[top_k_idx]
    return np.mean(top_k_targets > 0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Residual method analysis for feature contribution"
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to parquet dataset",
    )
    parser.add_argument(
        "--features-core",
        type=Path,
        required=True,
        help="JSON file with core feature names (64 features)",
    )
    parser.add_argument(
        "--features-add",
        type=Path,
        required=True,
        help="JSON file with additional feature names (25 features)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path for results",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for CI (default: 0.05)",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap samples (default: 1000)",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="target_5d",
        help="Target column name (default: target_5d)",
    )
    parser.add_argument(
        "--date-col",
        type=str,
        default="Date",
        help="Date column name (default: Date)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of top stocks for P@K (default: 50)",
    )

    args = parser.parse_args()

    # Load feature names
    with open(args.features_core) as f:
        core_data = json.load(f)
        # Handle both dict with metadata and plain list
        if isinstance(core_data, dict):
            core_features = core_data.get('feature_names', core_data.get('features', []))
        else:
            core_features = core_data

    with open(args.features_add) as f:
        add_data = json.load(f)
        # Handle both dict with metadata and plain list
        if isinstance(add_data, dict):
            add_features = add_data.get('feature_names', add_data.get('features', []))
        else:
            add_features = add_data

    print(f"Core features: {len(core_features)}")
    print(f"Additional features: {len(add_features)}")

    # Load data
    print(f"\nLoading data from {args.data}")
    df = pl.read_parquet(args.data)
    print(f"Loaded {len(df):,} samples")

    # Check feature availability
    available_cols = set(df.columns)
    missing_core = set(core_features) - available_cols
    missing_add = set(add_features) - available_cols
    if missing_core:
        raise ValueError(f"Missing core features: {missing_core}")
    if missing_add:
        raise ValueError(f"Missing additional features: {missing_add}")

    # Extract features and targets by date
    unique_dates = df[args.date_col].unique().sort()
    print(f"Unique dates: {len(unique_dates)}")

    results_by_date: list[dict] = []

    for date in unique_dates:
        date_df = df.filter(pl.col(args.date_col) == date)

        # Skip if insufficient samples
        if len(date_df) < args.top_k:
            continue

        # Extract features and targets
        X_core = date_df.select(core_features).to_numpy()
        X_add = date_df.select(add_features).to_numpy()
        y = date_df[args.target_col].to_numpy()

        # Skip if targets are all zero/nan
        if np.std(y) < 1e-6:
            continue

        # Residualize additional features
        X_add_resid = residualize_features(X_core, X_add, alpha=1.0)

        # Compute simple linear scores
        # Core only: Average of core features
        scores_core = X_core.mean(axis=1)

        # Core + Residual: Average of core + residualized additional
        X_combined = np.hstack([X_core, X_add_resid])
        scores_combined = X_combined.mean(axis=1)

        # Compute P@K for both
        pak_core = compute_precision_at_k(scores_core, y, args.top_k)
        pak_combined = compute_precision_at_k(scores_combined, y, args.top_k)

        results_by_date.append(
            {
                "date": date,
                "pak_core": pak_core,
                "pak_combined": pak_combined,
                "delta_pak": pak_combined - pak_core,
            }
        )

    # Convert to DataFrame
    results_df = pl.DataFrame(results_by_date)
    print(f"\nProcessed {len(results_df)} dates with valid targets")

    # Aggregate statistics
    delta_pak_values = results_df["delta_pak"].to_numpy()

    # Diebold-Mariano test (treating delta as loss differential)
    # Here we test if combined is better than core
    # Negative delta = core better, positive delta = combined better
    errors_core = -results_df["pak_core"].to_numpy()  # Negative P@K as "error"
    errors_combined = -results_df["pak_combined"].to_numpy()
    dm_stat, p_value = diebold_mariano_test(errors_core, errors_combined)

    # Bootstrap CI for delta P@K
    ci_lower, ci_upper = bootstrap_ci(
        delta_pak_values, n_bootstrap=args.n_bootstrap, alpha=args.alpha
    )

    # Summary statistics
    summary = {
        "n_dates": len(results_df),
        "mean_pak_core": results_df["pak_core"].mean(),
        "mean_pak_combined": results_df["pak_combined"].mean(),
        "mean_delta_pak": results_df["delta_pak"].mean(),
        "std_delta_pak": results_df["delta_pak"].std(),
        "dm_stat": dm_stat,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "pass_dm": dm_stat > 1.96,
        "pass_ci": ci_lower > 0,
        "pass_overall": (dm_stat > 1.96) and (ci_lower > 0),
    }

    # Print summary
    print("\n" + "=" * 80)
    print("RESIDUAL METHOD ANALYSIS RESULTS")
    print("=" * 80)
    print(f"Dates processed: {summary['n_dates']}")
    print(f"Mean P@K (core only): {summary['mean_pak_core']:.4f}")
    print(f"Mean P@K (core + resid): {summary['mean_pak_combined']:.4f}")
    print(f"Mean ΔP@K: {summary['mean_delta_pak']:.4f} ± {summary['std_delta_pak']:.4f}")
    print("\nDiebold-Mariano Test:")
    print(f"  DM statistic: {summary['dm_stat']:.4f}")
    print(f"  p-value: {summary['p_value']:.4f}")
    print(f"  Pass (DM > 1.96): {'✅' if summary['pass_dm'] else '❌'}")
    print(f"\nBootstrap {int((1-args.alpha)*100)}% CI:")
    print(f"  [{summary['ci_lower']:.4f}, {summary['ci_upper']:.4f}]")
    print(f"  Pass (CI lower > 0): {'✅' if summary['pass_ci'] else '❌'}")
    print(f"\nOverall: {'✅ PASS' if summary['pass_overall'] else '❌ FAIL'}")
    print("=" * 80)

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Save daily results
    daily_path = args.output.with_suffix(".daily.csv")
    results_df.write_csv(daily_path)
    print(f"\n✅ Daily results saved to: {daily_path}")

    # Save summary
    summary_path = args.output.with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
