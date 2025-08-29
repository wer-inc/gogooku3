#!/usr/bin/env python3
"""
Great Expectations Data Quality Suite for gogooku3-standalone
========================================

This module provides data quality validation for ML datasets and features.
Integrates seamlessly with existing pipelines without modifying core business logic.

Usage:
    # Enable data quality checks
    export DATA_QUALITY_ENABLED=1

    # Run validation
    python data_quality/great_expectations_suite.py validate --input data/processed/dataset.parquet

Integration:
    from data_quality.great_expectations_suite import DataQualityValidator
    validator = DataQualityValidator()
    results = validator.validate_dataset(df)
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Data validation result container."""
    passed: bool
    checks_passed: int
    checks_failed: int
    total_checks: int
    details: Dict[str, Any]
    recommendations: List[str]


class DataQualityValidator:
    """Great Expectations-inspired data quality validator."""

    def __init__(self, enable_warnings: bool = True):
        self.enable_warnings = enable_warnings
        self.project_root = Path(__file__).parent.parent
        self.results_dir = self.project_root / "data_quality" / "results"
        self.results_dir.mkdir(exist_ok=True)

        # Quality thresholds (configurable via environment)
        self.thresholds = {
            'missing_data_ratio': float(os.getenv('DATA_QUALITY_MISSING_RATIO', '0.05')),
            'duplicate_ratio': float(os.getenv('DATA_QUALITY_DUPLICATE_RATIO', '0.01')),
            'outlier_ratio': float(os.getenv('DATA_QUALITY_OUTLIER_RATIO', '0.05')),
            'correlation_threshold': float(os.getenv('DATA_QUALITY_CORRELATION_THRESHOLD', '0.95')),
            'min_rows': int(os.getenv('DATA_QUALITY_MIN_ROWS', '100')),
            'max_rows': int(os.getenv('DATA_QUALITY_MAX_ROWS', '1000000'))
        }

    def validate_dataset(self, df: pd.DataFrame, dataset_name: str = "dataset") -> ValidationResult:
        """
        Comprehensive dataset validation.

        Args:
            df: DataFrame to validate
            dataset_name: Name for reporting

        Returns:
            ValidationResult with detailed findings
        """
        logger.info(f"🔍 Starting data quality validation for {dataset_name}")

        checks_passed = 0
        checks_failed = 0
        details = {}
        recommendations = []

        # Basic structure validation
        structure_result = self._validate_structure(df)
        details['structure'] = structure_result
        if structure_result['passed']:
            checks_passed += 1
        else:
            checks_failed += 1
            recommendations.extend(structure_result.get('recommendations', []))

        # Data completeness validation
        completeness_result = self._validate_completeness(df)
        details['completeness'] = completeness_result
        if completeness_result['passed']:
            checks_passed += 1
        else:
            checks_failed += 1
            recommendations.extend(completeness_result.get('recommendations', []))

        # Data uniqueness validation
        uniqueness_result = self._validate_uniqueness(df)
        details['uniqueness'] = uniqueness_result
        if uniqueness_result['passed']:
            checks_passed += 1
        else:
            checks_failed += 1
            recommendations.extend(uniqueness_result.get('recommendations', []))

        # Statistical validation
        statistics_result = self._validate_statistics(df)
        details['statistics'] = statistics_result
        if statistics_result['passed']:
            checks_passed += 1
        else:
            checks_failed += 1
            recommendations.extend(statistics_result.get('recommendations', []))

        # Domain-specific validation (financial data)
        domain_result = self._validate_financial_domain(df)
        details['financial_domain'] = domain_result
        if domain_result['passed']:
            checks_passed += 1
        else:
            checks_failed += 1
            recommendations.extend(domain_result.get('recommendations', []))

        # Cross-correlation validation
        correlation_result = self._validate_correlations(df)
        details['correlations'] = correlation_result
        if correlation_result['passed']:
            checks_passed += 1
        else:
            checks_failed += 1
            recommendations.extend(correlation_result.get('recommendations', []))

        total_checks = checks_passed + checks_failed
        passed = checks_failed == 0

        result = ValidationResult(
            passed=passed,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            total_checks=total_checks,
            details=details,
            recommendations=recommendations
        )

        # Save results
        self._save_results(result, dataset_name)

        logger.info(f"✅ Validation completed: {checks_passed}/{total_checks} checks passed")
        if not passed:
            logger.warning(f"⚠️  Issues found: {recommendations}")

        return result

    def _validate_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate DataFrame structure."""
        recommendations = []

        # Check row count
        row_count = len(df)
        if row_count < self.thresholds['min_rows']:
            recommendations.append(f"Dataset has only {row_count} rows (minimum: {self.thresholds['min_rows']})")
        elif row_count > self.thresholds['max_rows']:
            recommendations.append(f"Dataset has {row_count} rows (maximum: {self.thresholds['max_rows']})")

        # Check column count
        col_count = len(df.columns)
        expected_cols = 169  # Based on project specifications
        if col_count != expected_cols:
            recommendations.append(f"Expected {expected_cols} columns, found {col_count}")

        # Check required columns
        required_cols = ['date', 'symbol', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            recommendations.append(f"Missing required columns: {missing_cols}")

        # Check data types
        dtype_issues = []
        if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
            dtype_issues.append("date column should be datetime type")
        if 'symbol' in df.columns and not df['symbol'].dtype == 'object':
            dtype_issues.append("symbol column should be string type")

        if dtype_issues:
            recommendations.extend(dtype_issues)

        passed = len(recommendations) == 0

        return {
            'passed': passed,
            'row_count': row_count,
            'column_count': col_count,
            'recommendations': recommendations
        }

    def _validate_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data completeness."""
        recommendations = []

        # Calculate missing data ratios
        missing_ratios = df.isnull().mean()

        # Overall missing ratio
        overall_missing = missing_ratios.mean()
        if overall_missing > self.thresholds['missing_data_ratio']:
            recommendations.append(".2%"
        # Column-specific missing ratios
        high_missing_cols = missing_ratios[missing_ratios > self.thresholds['missing_data_ratio']]
        if len(high_missing_cols) > 0:
            recommendations.append(f"High missing data in columns: {list(high_missing_cols.index)}")

        # Check for completely empty columns
        empty_cols = missing_ratios[missing_ratios == 1.0]
        if len(empty_cols) > 0:
            recommendations.append(f"Completely empty columns: {list(empty_cols.index)}")

        passed = overall_missing <= self.thresholds['missing_data_ratio']

        return {
            'passed': passed,
            'overall_missing_ratio': overall_missing,
            'high_missing_columns': list(high_missing_cols.index),
            'empty_columns': list(empty_cols.index),
            'recommendations': recommendations
        }

    def _validate_uniqueness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data uniqueness."""
        recommendations = []

        # Check for duplicate rows
        duplicate_ratio = df.duplicated().mean()
        if duplicate_ratio > self.thresholds['duplicate_ratio']:
            recommendations.append(".2%"
        # Check for duplicate primary keys (if applicable)
        if 'date' in df.columns and 'symbol' in df.columns:
            pk_duplicates = df.duplicated(subset=['date', 'symbol']).sum()
            if pk_duplicates > 0:
                recommendations.append(f"{pk_duplicates} duplicate date-symbol combinations found")

        passed = duplicate_ratio <= self.thresholds['duplicate_ratio']

        return {
            'passed': passed,
            'duplicate_ratio': duplicate_ratio,
            'primary_key_duplicates': pk_duplicates if 'pk_duplicates' in locals() else 0,
            'recommendations': recommendations
        }

    def _validate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate statistical properties."""
        recommendations = []

        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Check for constant columns
        constant_cols = []
        for col in numeric_cols:
            if df[col].nunique() <= 1:
                constant_cols.append(col)

        if constant_cols:
            recommendations.append(f"Constant columns (no variation): {constant_cols}")

        # Check for infinite values
        infinite_counts = {}
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                infinite_counts[col] = inf_count

        if infinite_counts:
            recommendations.append(f"Infinite values found: {infinite_counts}")

        # Basic statistical summaries
        stats_summary = {}
        for col in numeric_cols[:10]:  # Limit to first 10 for performance
            stats = df[col].describe()
            stats_summary[col] = {
                'mean': stats['mean'],
                'std': stats['std'],
                'min': stats['min'],
                'max': stats['max']
            }

        passed = len(constant_cols) == 0 and len(infinite_counts) == 0

        return {
            'passed': passed,
            'constant_columns': constant_cols,
            'infinite_values': infinite_counts,
            'stats_summary': stats_summary,
            'recommendations': recommendations
        }

    def _validate_financial_domain(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate financial data domain rules."""
        recommendations = []

        # Check price columns
        price_cols = ['open', 'high', 'low', 'close']
        existing_price_cols = [col for col in price_cols if col in df.columns]

        if existing_price_cols:
            # Check for negative prices
            negative_prices = {}
            for col in existing_price_cols:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    negative_prices[col] = negative_count

            if negative_prices:
                recommendations.append(f"Negative prices found: {negative_prices}")

            # Check OHLC relationship (O ≤ H, O ≥ L, C ≤ H, C ≥ L)
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                invalid_ohlc = (
                    (df['open'] > df['high']) |
                    (df['open'] < df['low']) |
                    (df['close'] > df['high']) |
                    (df['close'] < df['low'])
                ).sum()

                if invalid_ohlc > 0:
                    recommendations.append(f"{invalid_ohlc} rows have invalid OHLC relationships")

        # Check volume column
        if 'volume' in df.columns:
            negative_volume = (df['volume'] < 0).sum()
            if negative_volume > 0:
                recommendations.append(f"{negative_volume} rows have negative volume")

        # Check date continuity (if applicable)
        if 'date' in df.columns and 'symbol' in df.columns:
            # Group by symbol and check date gaps
            date_gaps = {}
            for symbol in df['symbol'].unique()[:5]:  # Check first 5 symbols
                symbol_data = df[df['symbol'] == symbol].copy()
                symbol_data = symbol_data.sort_values('date')
                date_diffs = symbol_data['date'].diff().dt.days
                gaps = (date_diffs > 1).sum()
                if gaps > 0:
                    date_gaps[symbol] = gaps

            if date_gaps:
                recommendations.append(f"Date gaps found in symbols: {date_gaps}")

        passed = len(recommendations) == 0

        return {
            'passed': passed,
            'negative_prices': negative_prices if 'negative_prices' in locals() else {},
            'invalid_ohlc': invalid_ohlc if 'invalid_ohlc' in locals() else 0,
            'negative_volume': negative_volume if 'negative_volume' in locals() else 0,
            'date_gaps': date_gaps if 'date_gaps' in locals() else {},
            'recommendations': recommendations
        }

    def _validate_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate feature correlations."""
        recommendations = []

        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Limit to reasonable number of columns for performance
        if len(numeric_cols) > 50:
            numeric_cols = numeric_cols[:50]

        if len(numeric_cols) > 1:
            # Calculate correlation matrix
            corr_matrix = df[numeric_cols].corr()

            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr = abs(corr_matrix.iloc[i, j])
                    if corr > self.thresholds['correlation_threshold']:
                        high_corr_pairs.append((
                            numeric_cols[i],
                            numeric_cols[j],
                            corr
                        ))

            if high_corr_pairs:
                recommendations.append(f"High correlations found: {len(high_corr_pairs)} pairs")

            # Check for perfect correlations (potential duplicates)
            perfect_corr = [pair for pair in high_corr_pairs if pair[2] > 0.999]
            if perfect_corr:
                recommendations.append(f"Perfect correlations (potential duplicates): {perfect_corr}")

        passed = len(recommendations) == 0

        return {
            'passed': passed,
            'high_correlation_pairs': len(high_corr_pairs) if 'high_corr_pairs' in locals() else 0,
            'perfect_correlations': len(perfect_corr) if 'perfect_corr' in locals() else 0,
            'correlation_threshold': self.thresholds['correlation_threshold'],
            'recommendations': recommendations
        }

    def _save_results(self, result: ValidationResult, dataset_name: str):
        """Save validation results to file."""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"validation_{dataset_name}_{timestamp}.json"

        result_dict = {
            'timestamp': timestamp,
            'dataset_name': dataset_name,
            'passed': result.passed,
            'checks_passed': result.checks_passed,
            'checks_failed': result.checks_failed,
            'total_checks': result.total_checks,
            'recommendations': result.recommendations,
            'details': result.details
        }

        filepath = self.results_dir / filename
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)

        logger.info(f"📊 Validation results saved to {filepath}")


def validate_file(input_path: str, output_path: Optional[str] = None) -> ValidationResult:
    """
    Validate a dataset file.

    Args:
        input_path: Path to input dataset file
        output_path: Optional path to save results

    Returns:
        ValidationResult
    """
    logger.info(f"🔍 Validating dataset: {input_path}")

    # Load dataset
    if input_path.endswith('.parquet'):
        df = pd.read_parquet(input_path)
    elif input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path}")

    logger.info(f"📊 Dataset loaded: {len(df)} rows × {len(df.columns)} columns")

    # Validate
    validator = DataQualityValidator()
    result = validator.validate_dataset(df, Path(input_path).stem)

    # Save results if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump({
                'passed': result.passed,
                'checks_passed': result.checks_passed,
                'checks_failed': result.checks_failed,
                'total_checks': result.total_checks,
                'recommendations': result.recommendations,
                'details': result.details
            }, f, indent=2, default=str)

    return result


def main():
    """Command-line interface for data quality validation."""
    import argparse

    parser = argparse.ArgumentParser(
        description='gogooku3 Data Quality Validator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_quality/great_expectations_suite.py validate --input data/processed/dataset.parquet
  python data_quality/great_expectations_suite.py validate --input data/features/features.csv --output results.json

Environment Variables:
  DATA_QUALITY_ENABLED=1          # Enable data quality checks
  DATA_QUALITY_MISSING_RATIO=0.05 # Max missing data ratio
  DATA_QUALITY_DUPLICATE_RATIO=0.01 # Max duplicate ratio
  DATA_QUALITY_OUTLIER_RATIO=0.05  # Max outlier ratio
  DATA_QUALITY_CORRELATION_THRESHOLD=0.95 # High correlation threshold
        """
    )

    parser.add_argument(
        'command',
        choices=['validate'],
        help='Command to execute'
    )

    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input dataset file path'
    )

    parser.add_argument(
        '--output', '-o',
        help='Output results file path'
    )

    parser.add_argument(
        '--format',
        choices=['json', 'summary'],
        default='summary',
        help='Output format'
    )

    args = parser.parse_args()

    try:
        result = validate_file(args.input, args.output)

        if args.format == 'summary':
            print(f"\n📊 Data Quality Validation Summary")
            print("=" * 50)
            print(f"Status: {'✅ PASSED' if result.passed else '❌ FAILED'}")
            print(f"Checks: {result.checks_passed}/{result.total_checks} passed")
            print(f"Dataset: {args.input}")

            if result.recommendations:
                print(f"\n⚠️  Recommendations ({len(result.recommendations)}):")
                for i, rec in enumerate(result.recommendations, 1):
                    print(f"  {i}. {rec}")

        sys.exit(0 if result.passed else 1)

    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
