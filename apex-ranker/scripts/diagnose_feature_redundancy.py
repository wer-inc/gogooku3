#!/usr/bin/env python3
"""Feature redundancy diagnosis: correlation, VIF, and overlap analysis.

Usage:
    python diagnose_feature_redundancy.py \
        --data output/ml_dataset_latest_clean_with_adv.parquet \
        --features apex-ranker/configs/feature_groups_v0_latest_89.yaml \
        --output results/feature_redundancy_report.json
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from scipy.stats import spearmanr


def calculate_vif(df: pl.DataFrame, feature_cols: list[str]) -> dict[str, float]:
    """Calculate Variance Inflation Factor for each feature.

    VIF > 10 indicates severe multicollinearity.
    VIF > 5 indicates moderate multicollinearity.
    """
    from sklearn.linear_model import LinearRegression

    # Convert to numpy for sklearn
    X = df.select(feature_cols).to_numpy()

    # Handle NaNs
    X = np.nan_to_num(X, nan=0.0)

    vif_dict = {}
    for i, col in enumerate(feature_cols):
        # Regress column i on all other columns
        X_others = np.delete(X, i, axis=1)
        y = X[:, i]

        # Fit regression
        model = LinearRegression()
        model.fit(X_others, y)

        # Calculate R²
        y_pred = model.predict(X_others)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # VIF = 1 / (1 - R²)
        vif = 1 / (1 - r_squared) if r_squared < 0.999 else 999.0
        vif_dict[col] = vif

    return vif_dict


def find_high_correlation_pairs(
    df: pl.DataFrame,
    feature_cols: list[str],
    threshold: float = 0.9,
) -> list[dict]:
    """Find feature pairs with |correlation| > threshold."""
    # Convert to numpy
    X = df.select(feature_cols).to_numpy()
    X = np.nan_to_num(X, nan=0.0)

    # Calculate Spearman correlation (rank-based, more robust)
    corr_matrix, _ = spearmanr(X, axis=0, nan_policy='omit')

    high_corr_pairs = []
    n = len(feature_cols)

    for i in range(n):
        for j in range(i + 1, n):
            corr = corr_matrix[i, j]
            if abs(corr) > threshold:
                high_corr_pairs.append({
                    'feature_1': feature_cols[i],
                    'feature_2': feature_cols[j],
                    'correlation': float(corr),
                })

    # Sort by absolute correlation
    high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)

    return high_corr_pairs


def analyze_redundancy(
    data_path: Path,
    features_config: Path,
    sample_size: int = 50000,
) -> dict:
    """Comprehensive redundancy analysis.

    Returns:
        Dictionary with:
        - high_vif: Features with VIF > 10
        - moderate_vif: Features with VIF > 5
        - high_correlation: Pairs with |ρ| > 0.9
        - moderate_correlation: Pairs with |ρ| > 0.7
        - removal_candidates: Recommended features to remove
    """
    # Load feature configuration
    with features_config.open('r') as f:
        config = yaml.safe_load(f)

    feature_cols = []
    # Handle both 'groups' and 'feature_groups' keys
    groups = config.get('groups', config.get('feature_groups', {}))
    for group_name, group_data in groups.items():
        if isinstance(group_data, dict) and 'include' in group_data:
            feature_cols.extend(group_data['include'])
        elif isinstance(group_data, list):
            feature_cols.extend(group_data)

    print(f"Loaded {len(feature_cols)} features from config")

    # Load data sample
    print(f"Loading data sample ({sample_size:,} rows)...")
    df = pl.read_parquet(data_path)

    # Filter to recent dates (more representative)
    df = df.sort('Date', descending=True).head(sample_size)

    # Filter valid rows (no NaN targets, sufficient data)
    df = df.filter(
        pl.col('target_5d').is_not_null()
    )

    print(f"Loaded {len(df):,} rows")

    # Calculate VIF
    print("Calculating VIF...")
    vif_dict = calculate_vif(df, feature_cols)

    high_vif = {k: v for k, v in vif_dict.items() if v > 10}
    moderate_vif = {k: v for k, v in vif_dict.items() if 5 < v <= 10}

    print(f"High VIF (>10): {len(high_vif)}")
    print(f"Moderate VIF (5-10): {len(moderate_vif)}")

    # Find high correlation pairs
    print("Calculating correlations...")
    high_corr = find_high_correlation_pairs(df, feature_cols, threshold=0.9)
    moderate_corr = find_high_correlation_pairs(df, feature_cols, threshold=0.7)
    moderate_corr = [x for x in moderate_corr if x not in high_corr]

    print(f"High correlation (|ρ|>0.9): {len(high_corr)} pairs")
    print(f"Moderate correlation (|ρ|>0.7): {len(moderate_corr)} pairs")

    # Generate removal candidates
    # Priority: features appearing multiple times in high-corr pairs + high VIF
    feature_counts = {}
    for pair in high_corr:
        for feat in [pair['feature_1'], pair['feature_2']]:
            feature_counts[feat] = feature_counts.get(feat, 0) + 1

    removal_candidates = []
    for feat, count in sorted(feature_counts.items(), key=lambda x: -x[1]):
        vif = vif_dict.get(feat, 0)
        removal_candidates.append({
            'feature': feat,
            'high_corr_count': count,
            'vif': vif,
            'severity': 'high' if vif > 10 else 'moderate',
        })

    # Add high-VIF-only features
    for feat, vif in high_vif.items():
        if feat not in feature_counts:
            removal_candidates.append({
                'feature': feat,
                'high_corr_count': 0,
                'vif': vif,
                'severity': 'high',
            })

    return {
        'total_features': len(feature_cols),
        'high_vif': high_vif,
        'moderate_vif': moderate_vif,
        'high_correlation_pairs': high_corr,
        'moderate_correlation_pairs': moderate_corr,
        'removal_candidates': removal_candidates,
    }


def print_report(analysis: dict) -> None:
    """Print human-readable redundancy report."""
    print("\n" + "=" * 80)
    print("FEATURE REDUNDANCY DIAGNOSIS REPORT")
    print("=" * 80)
    print()

    print(f"📊 TOTAL FEATURES: {analysis['total_features']}")
    print()

    # VIF summary
    print("🔴 HIGH VIF (>10) - Severe Multicollinearity")
    print("-" * 80)
    if analysis['high_vif']:
        for feat, vif in sorted(analysis['high_vif'].items(), key=lambda x: -x[1])[:10]:
            print(f"  {feat:40s}  VIF: {vif:>8.2f}")
    else:
        print("  None")
    print()

    print("🟡 MODERATE VIF (5-10) - Moderate Multicollinearity")
    print("-" * 80)
    if analysis['moderate_vif']:
        for feat, vif in sorted(analysis['moderate_vif'].items(), key=lambda x: -x[1])[:10]:
            print(f"  {feat:40s}  VIF: {vif:>8.2f}")
    else:
        print("  None")
    print()

    # Correlation summary
    print("🔴 HIGH CORRELATION (|ρ|>0.9) - Redundant Pairs")
    print("-" * 80)
    if analysis['high_correlation_pairs']:
        for pair in analysis['high_correlation_pairs'][:10]:
            print(f"  {pair['feature_1']:30s} ↔ {pair['feature_2']:30s}  ρ={pair['correlation']:>6.3f}")
    else:
        print("  None")
    print()

    # Removal candidates
    print("💡 REMOVAL CANDIDATES (Ranked by Severity)")
    print("-" * 80)
    if analysis['removal_candidates']:
        print(f"  {'Feature':40s}  {'Corr Count':>12s}  {'VIF':>8s}  {'Severity':>10s}")
        print("  " + "-" * 78)
        for cand in analysis['removal_candidates'][:15]:
            print(
                f"  {cand['feature']:40s}  "
                f"{cand['high_corr_count']:>12d}  "
                f"{cand['vif']:>8.2f}  "
                f"{cand['severity']:>10s}"
            )
    else:
        print("  None")
    print()
    print("=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Diagnose feature redundancy')
    parser.add_argument('--data', type=Path, required=True, help='Parquet dataset path')
    parser.add_argument('--features', type=Path, required=True, help='Feature groups YAML')
    parser.add_argument('--output', type=Path, required=True, help='Output JSON path')
    parser.add_argument('--sample-size', type=int, default=50000, help='Sample size')

    args = parser.parse_args()

    # Run analysis
    analysis = analyze_redundancy(
        args.data,
        args.features,
        sample_size=args.sample_size,
    )

    # Print report
    print_report(analysis)

    # Save JSON
    with args.output.open('w') as f:
        json.dump(analysis, f, indent=2)

    print(f"\n📁 Full report saved to: {args.output}")


if __name__ == '__main__':
    main()
