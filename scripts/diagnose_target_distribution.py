#!/usr/bin/env python3
"""
Phase 0: Diagnose target variable distribution
目的: returns_*d の分布を確認し、学習可能な状態かを検証
"""

import pandas as pd
import numpy as np
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def analyze_target_distribution():
    """ターゲット変数の分布を詳細に分析"""

    print("=" * 60)
    print("🔍 ATFT-GAT-FAN Target Distribution Diagnosis")
    print("=" * 60)

    # 1. データ読み込み
    data_path = Path("/home/ubuntu/gogooku3-standalone/output/ml_dataset_latest_full.parquet")

    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return

    print(f"📂 Loading data from: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"✅ Data loaded: {df.shape}")

    # 2. ターゲット変数の特定
    target_columns = [col for col in df.columns if col.startswith('returns_') and col.endswith('d')]
    print(f"\n📊 Target columns found: {target_columns}")

    if not target_columns:
        print("❌ No target columns (returns_*d) found!")
        return

    # 3. 各ターゲットの統計分析
    print("\n" + "=" * 60)
    print("📈 Target Distribution Statistics")
    print("=" * 60)

    analysis_results = {}

    for col in target_columns:
        print(f"\n🎯 Analyzing: {col}")
        print("-" * 40)

        values = df[col].dropna()

        # 基本統計
        stats_dict = {
            'count': len(values),
            'mean': values.mean(),
            'std': values.std(),
            'min': values.min(),
            'max': values.max(),
            'median': values.median(),
            'skew': values.skew(),
            'kurtosis': values.kurtosis(),
            'q01': values.quantile(0.01),
            'q05': values.quantile(0.05),
            'q25': values.quantile(0.25),
            'q75': values.quantile(0.75),
            'q95': values.quantile(0.95),
            'q99': values.quantile(0.99),
            'zero_ratio': (values == 0).sum() / len(values),
            'near_zero_ratio': (np.abs(values) < 1e-6).sum() / len(values),
            'unique_values': values.nunique(),
            'iqr': values.quantile(0.75) - values.quantile(0.25)
        }

        analysis_results[col] = stats_dict

        # 統計値表示
        print(f"  Count:        {stats_dict['count']:,.0f}")
        print(f"  Mean:         {stats_dict['mean']:.6f}")
        print(f"  Std:          {stats_dict['std']:.6f}")
        print(f"  Min:          {stats_dict['min']:.6f}")
        print(f"  Max:          {stats_dict['max']:.6f}")
        print(f"  Median:       {stats_dict['median']:.6f}")
        print(f"  IQR:          {stats_dict['iqr']:.6f}")
        print(f"  Skewness:     {stats_dict['skew']:.3f}")
        print(f"  Kurtosis:     {stats_dict['kurtosis']:.3f}")

        # 分位点
        print(f"\n  Quantiles:")
        print(f"    1%:         {stats_dict['q01']:.6f}")
        print(f"    5%:         {stats_dict['q05']:.6f}")
        print(f"    25%:        {stats_dict['q25']:.6f}")
        print(f"    75%:        {stats_dict['q75']:.6f}")
        print(f"    95%:        {stats_dict['q95']:.6f}")
        print(f"    99%:        {stats_dict['q99']:.6f}")

        # 問題の検出
        print(f"\n  ⚠️ Potential Issues:")
        print(f"    Zero ratio:      {stats_dict['zero_ratio']:.2%}")
        print(f"    Near-zero ratio: {stats_dict['near_zero_ratio']:.2%}")
        print(f"    Unique values:   {stats_dict['unique_values']:,}")

        # スケールの問題判定
        if stats_dict['std'] < 0.001:
            print(f"    🔴 CRITICAL: Standard deviation too small ({stats_dict['std']:.6f})")
            print(f"       → Model cannot learn from such small variations!")
        elif stats_dict['std'] < 0.01:
            print(f"    🟡 WARNING: Small standard deviation ({stats_dict['std']:.6f})")
            print(f"       → May cause learning difficulties")
        else:
            print(f"    🟢 OK: Standard deviation acceptable ({stats_dict['std']:.6f})")

        # IQRチェック
        if stats_dict['iqr'] < 0.001:
            print(f"    🔴 CRITICAL: IQR too small ({stats_dict['iqr']:.6f})")
            print(f"       → Most values are nearly identical!")

        # 外れ値の検出
        outlier_threshold = 3
        z_scores = np.abs((values - values.mean()) / values.std())
        outlier_ratio = (z_scores > outlier_threshold).sum() / len(values)
        print(f"    Outliers (>3σ): {outlier_ratio:.2%}")

    # 4. 推奨事項
    print("\n" + "=" * 60)
    print("💡 Recommendations")
    print("=" * 60)

    for col, stats_dict in analysis_results.items():
        print(f"\n{col}:")

        if stats_dict['std'] < 0.001:
            print("  🔴 URGENT: Data scaling required!")
            print("     Actions:")
            print("     1. Multiply by 100 (convert to percentage)")
            print("     2. Apply robust scaling")
            print("     3. Use percentile normalization")

            # スケーリング後の予測値
            scaled_std = stats_dict['std'] * 100
            print(f"     → After scaling (*100): std = {scaled_std:.4f}")

        elif stats_dict['std'] < 0.01:
            print("  🟡 Consider data scaling:")
            print("     - StandardScaler or RobustScaler")
            print("     - Clip outliers at 99th percentile")
        else:
            print("  🟢 Data scale acceptable")
            print("     - Consider outlier clipping if needed")

    # 5. 相関分析
    print("\n" + "=" * 60)
    print("📊 Target Correlations")
    print("=" * 60)

    if len(target_columns) > 1:
        corr_matrix = df[target_columns].corr()
        print("\nCorrelation Matrix:")
        print(corr_matrix.round(3))

        # 高相関の警告
        for i in range(len(target_columns)):
            for j in range(i+1, len(target_columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.9:
                    print(f"\n⚠️ High correlation ({corr_val:.3f}) between {target_columns[i]} and {target_columns[j]}")

    # 6. サンプル値の表示
    print("\n" + "=" * 60)
    print("📝 Sample Values (first 10 non-zero)")
    print("=" * 60)

    for col in target_columns[:2]:  # 最初の2つだけ表示
        non_zero = df[df[col] != 0][col].head(10).values
        print(f"\n{col}:")
        print(f"  {non_zero}")

    # 7. 最終診断
    print("\n" + "=" * 60)
    print("🏁 Final Diagnosis")
    print("=" * 60)

    critical_issues = []
    warnings = []

    for col, stats_dict in analysis_results.items():
        if stats_dict['std'] < 0.001:
            critical_issues.append(f"{col}: Scale too small (std={stats_dict['std']:.6f})")
        elif stats_dict['std'] < 0.01:
            warnings.append(f"{col}: Small scale (std={stats_dict['std']:.6f})")

        if stats_dict['near_zero_ratio'] > 0.5:
            warnings.append(f"{col}: {stats_dict['near_zero_ratio']:.1%} near-zero values")

    if critical_issues:
        print("\n🔴 CRITICAL ISSUES (must fix):")
        for issue in critical_issues:
            print(f"  - {issue}")
        print("\n  → This explains why Val RankIC is stuck at 0.0719!")
        print("  → Model cannot learn from such small variations")

    if warnings:
        print("\n🟡 WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")

    if not critical_issues and not warnings:
        print("\n🟢 No major issues detected in target distribution")

    # 8. 次のステップ
    print("\n" + "=" * 60)
    print("🚀 Next Steps")
    print("=" * 60)

    if critical_issues:
        print("\n1. Immediate action: Scale target variables")
        print("   - Create scripts/fix_target_scaling.py")
        print("   - Apply scaling factor (e.g., *100 for percentage)")
        print("   - Regenerate ATFT data with scaled targets")
        print("\n2. Verify RankIC calculation logic")
        print("   - Check if RankIC uses correct scale")
        print("   - Ensure no division by near-zero std")
        print("\n3. Re-run training with fixed data")
    else:
        print("\n1. Proceed with Phase 1: Data quality improvements")
        print("2. Check feature distributions next")

if __name__ == "__main__":
    analyze_target_distribution()