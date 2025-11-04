#!/usr/bin/env python3
"""5-item sanity check for zero residual contribution result."""

import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge


def load_features(json_path):
    """Load feature names from JSON."""
    with open(json_path) as f:
        data = json.load(f)
        if isinstance(data, dict):
            return data.get('feature_names', data.get('features', []))
        return data

def main():
    print("=" * 80)
    print("SANITY CHECK: Zero Residual Contribution")
    print("=" * 80)

    # Load data
    df = pl.read_parquet('output/ml_dataset_latest_clean_with_adv.parquet')
    print(f"\nDataset: {len(df):,} samples")

    # Load feature lists
    core_features = load_features('apex-ranker/configs/feature_names_core62.json')
    add_features = load_features('apex-ranker/configs/feature_names_add25.json')
    print(f"Core: {len(core_features)}, Add: {len(add_features)}")

    # Filter to available features
    available = set(df.columns)
    core_features = [f for f in core_features if f in available]
    add_features = [f for f in add_features if f in available]
    print(f"After filtering: Core={len(core_features)}, Add={len(add_features)}\n")

    # Sample 100 random dates for sanity check (full dataset too slow)
    dates = df['date'].unique().sort()
    np.random.seed(42)
    sample_dates = np.random.choice(dates, size=min(100, len(dates)), replace=False)
    print(f"Sampling {len(sample_dates)} dates for sanity check\n")

    results = {
        'check1_residual_variance': [],
        'check2_regression_r2': [],
        'check3_score_diff_distribution': [],
        'check4_alpha_sensitivity': {},
        'check5_topk_identity': []
    }

    for date in sample_dates[:10]:  # First 10 dates for detailed check
        date_df = df.filter(pl.col('date') == date)
        if len(date_df) < 80:
            continue

        # Extract features
        X_core = date_df.select(core_features).to_numpy()
        X_add = date_df.select(add_features).to_numpy()
        y = date_df['target_5d'].to_numpy()

        # Impute NaNs
        imputer_core = SimpleImputer(strategy='median')
        imputer_add = SimpleImputer(strategy='median')
        X_core = imputer_core.fit_transform(X_core)
        X_add = imputer_add.fit_transform(X_add)

        # Check 1: Residual variance
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_core, X_add)
        X_add_pred = ridge.predict(X_core)
        X_add_resid = X_add - X_add_pred
        residual_var = np.var(X_add_resid, axis=0).mean()
        results['check1_residual_variance'].append(residual_var)

        # Check 2: Regression R²
        ss_total = np.sum((X_add - X_add.mean(axis=0))**2)
        ss_residual = np.sum((X_add - X_add_pred)**2)
        r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
        results['check2_regression_r2'].append(r2)

        # Check 3: Score difference distribution
        # Ridge on core only
        ridge_core = Ridge(alpha=1.0)
        ridge_core.fit(X_core, y)
        scores_core = ridge_core.predict(X_core)

        # Ridge on combined
        X_combined = np.hstack([X_core, X_add_resid])
        ridge_combined = Ridge(alpha=1.0)
        ridge_combined.fit(X_combined, y)
        scores_combined = ridge_combined.predict(X_combined)

        score_diff = scores_combined - scores_core
        results['check3_score_diff_distribution'].append({
            'date': str(date),
            'mean': float(np.mean(score_diff)),
            'std': float(np.std(score_diff)),
            'max_abs': float(np.max(np.abs(score_diff)))
        })

    # Check 4: Alpha sensitivity (on one sample date)
    date = sample_dates[0]
    date_df = df.filter(pl.col('date') == date)
    X_core = date_df.select(core_features).to_numpy()
    X_add = date_df.select(add_features).to_numpy()
    y = date_df['target_5d'].to_numpy()

    imputer_core = SimpleImputer(strategy='median')
    imputer_add = SimpleImputer(strategy='median')
    X_core = imputer_core.fit_transform(X_core)
    X_add = imputer_add.fit_transform(X_add)

    for alpha in [1e-6, 1e-4, 1e-2, 1.0]:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_core, X_add)
        X_add_pred = ridge.predict(X_core)
        X_add_resid = X_add - X_add_pred

        # P@K with residualized features
        X_combined = np.hstack([X_core, X_add_resid])
        ridge_combined = Ridge(alpha=alpha)
        ridge_combined.fit(X_combined, y)
        scores_combined = ridge_combined.predict(X_combined)

        # P@K with core only
        ridge_core = Ridge(alpha=alpha)
        ridge_core.fit(X_core, y)
        scores_core = ridge_core.predict(X_core)

        k = len(y) // 10
        topk_combined = np.argpartition(-scores_combined, k)[:k]
        topk_core = np.argpartition(-scores_core, k)[:k]

        pak_combined = (y[topk_combined] > 0).mean()
        pak_core = (y[topk_core] > 0).mean()

        results['check4_alpha_sensitivity'][f'alpha_{alpha}'] = {
            'pak_core': float(pak_core),
            'pak_combined': float(pak_combined),
            'delta': float(pak_combined - pak_core)
        }

    # Check 5: Top-K identity (with stable sort)
    overlap_ratios = []
    for date in sample_dates[:20]:
        date_df = df.filter(pl.col('date') == date)
        if len(date_df) < 80:
            continue

        X_core = date_df.select(core_features).to_numpy()
        X_add = date_df.select(add_features).to_numpy()
        y = date_df['target_5d'].to_numpy()

        imputer_core = SimpleImputer(strategy='median')
        imputer_add = SimpleImputer(strategy='median')
        X_core = imputer_core.fit_transform(X_core)
        X_add = imputer_add.fit_transform(X_add)

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_core, X_add)
        X_add_resid = X_add - ridge.predict(X_core)

        ridge_core = Ridge(alpha=1.0)
        ridge_core.fit(X_core, y)
        scores_core = ridge_core.predict(X_core)

        X_combined = np.hstack([X_core, X_add_resid])
        ridge_combined = Ridge(alpha=1.0)
        ridge_combined.fit(X_combined, y)
        scores_combined = ridge_combined.predict(X_combined)

        k = len(y) // 10
        topk_core = set(np.argpartition(-scores_core, k)[:k])
        topk_combined = set(np.argpartition(-scores_combined, k)[:k])

        overlap = len(topk_core & topk_combined) / k
        overlap_ratios.append(overlap)

    results['check5_topk_identity'] = {
        'mean_overlap': float(np.mean(overlap_ratios)),
        'min_overlap': float(np.min(overlap_ratios)),
        'max_overlap': float(np.max(overlap_ratios))
    }

    # Print results
    print("\n" + "=" * 80)
    print("CHECK 1: RESIDUAL VARIANCE")
    print("=" * 80)
    print(f"Mean variance of X_add⊥: {np.mean(results['check1_residual_variance']):.6f}")
    print(f"Interpretation: {'✅ TRUE ZERO (var ≈ 0)' if np.mean(results['check1_residual_variance']) < 1e-6 else '⚠️ NON-ZERO VARIANCE'}")

    print("\n" + "=" * 80)
    print("CHECK 2: REGRESSION R²")
    print("=" * 80)
    print(f"Mean R² (X_add ~ X_core): {np.mean(results['check2_regression_r2']):.6f}")
    print(f"Interpretation: {'✅ COMPLETE REDUNDANCY (R² > 0.999)' if np.mean(results['check2_regression_r2']) > 0.999 else '⚠️ PARTIAL REDUNDANCY'}")

    print("\n" + "=" * 80)
    print("CHECK 3: SCORE DIFFERENCE DISTRIBUTION")
    print("=" * 80)
    diffs = results['check3_score_diff_distribution']
    print(f"Mean of score differences: {np.mean([d['mean'] for d in diffs]):.6e}")
    print(f"Max absolute difference: {np.max([d['max_abs'] for d in diffs]):.6e}")
    print(f"Interpretation: {'✅ FLOATING POINT ZERO (diff < 1e-10)' if np.max([d['max_abs'] for d in diffs]) < 1e-10 else '⚠️ NON-ZERO DIFFERENCE'}")

    print("\n" + "=" * 80)
    print("CHECK 4: ALPHA SENSITIVITY")
    print("=" * 80)
    for alpha_key, vals in results['check4_alpha_sensitivity'].items():
        print(f"{alpha_key}: ΔP@K = {vals['delta']:.6f}")
    all_deltas = [v['delta'] for v in results['check4_alpha_sensitivity'].values()]
    print(f"Interpretation: {'✅ STRUCTURALLY ZERO (invariant to α)' if np.std(all_deltas) < 1e-6 else '⚠️ ALPHA-DEPENDENT'}")

    print("\n" + "=" * 80)
    print("CHECK 5: TOP-K IDENTITY")
    print("=" * 80)
    print(f"Mean overlap: {results['check5_topk_identity']['mean_overlap']:.4f}")
    print(f"Min overlap: {results['check5_topk_identity']['min_overlap']:.4f}")
    print(f"Interpretation: {'✅ IDENTICAL TOP-K (overlap > 0.99)' if results['check5_topk_identity']['mean_overlap'] > 0.99 else '⚠️ DIFFERENT TOP-K'}")

    # Save results
    output_path = Path('results/diagnostics/sanity_check_zero_residual.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Sanity check results saved to: {output_path}")

    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    checks_passed = 0
    if np.mean(results['check1_residual_variance']) < 1e-6:
        checks_passed += 1
    if np.mean(results['check2_regression_r2']) > 0.999:
        checks_passed += 1
    if np.max([d['max_abs'] for d in results['check3_score_diff_distribution']]) < 1e-10:
        checks_passed += 1
    if np.std([v['delta'] for v in results['check4_alpha_sensitivity'].values()]) < 1e-6:
        checks_passed += 1
    if results['check5_topk_identity']['mean_overlap'] > 0.99:
        checks_passed += 1

    print(f"Checks passed: {checks_passed}/5")
    if checks_passed >= 4:
        print("✅ CONFIRMED: Zero contribution is TRUE (not a measurement artifact)")
        print("   → Additional 25 features are completely redundant with Core 62")
        print("   → Proceed with Core64-only retrain")
    else:
        print("⚠️  WARNING: Some checks failed - investigate measurement artifacts")
        print("   → May need to debug residual analysis implementation")

if __name__ == '__main__':
    main()
