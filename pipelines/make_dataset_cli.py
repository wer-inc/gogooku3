#!/usr/bin/env python
"""
MLデータセット作成CLI
期間指定、列選択、メモリ最適化されたデータロード
"""

import argparse
import polars as pl
from pathlib import Path
import sys
import os
from datetime import datetime
from typing import List, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from src.gogooku3.features.ta_core import CrossSectionalNormalizer
from src.gogooku3.contracts.schemas import DataSchemas, SchemaValidator


def parse_args():
    parser = argparse.ArgumentParser(description="ML Dataset Builder CLI")
    
    # Data selection
    parser.add_argument("--start-date", type=str, required=True, 
                       help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True,
                       help="End date (YYYY-MM-DD)")
    parser.add_argument("--codes", type=str, nargs="+",
                       help="Stock codes to include (default: all)")
    parser.add_argument("--sections", type=str, nargs="+",
                       choices=["Prime", "Standard", "Growth"],
                       help="Market sections to include")
    
    # Feature selection
    parser.add_argument("--feature-groups", type=str, nargs="+",
                       choices=["price", "market", "cross", "flow", "financial", "all"],
                       default=["all"],
                       help="Feature groups to include")
    parser.add_argument("--exclude-features", type=str, nargs="+",
                       help="Specific features to exclude")
    
    # Processing options
    parser.add_argument("--normalize", action="store_true",
                       help="Apply cross-sectional normalization")
    parser.add_argument("--winsorize", type=float, default=0.01,
                       help="Winsorize percentile (default: 0.01)")
    parser.add_argument("--fill-method", type=str, default="forward",
                       choices=["forward", "median", "zero", "none"],
                       help="Missing value fill method")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="output/datasets",
                       help="Output directory")
    parser.add_argument("--output-format", type=str, default="parquet",
                       choices=["parquet", "csv", "feather"],
                       help="Output format")
    parser.add_argument("--partition-by", type=str,
                       choices=["date", "code", "month", "none"],
                       default="none",
                       help="Partitioning strategy")
    
    # Memory optimization
    parser.add_argument("--chunk-size", type=int, default=100000,
                       help="Chunk size for processing")
    parser.add_argument("--lazy", action="store_true",
                       help="Use lazy evaluation")
    parser.add_argument("--memory-limit", type=float, default=8.0,
                       help="Memory limit in GB")
    
    # Validation
    parser.add_argument("--validate", action="store_true",
                       help="Run data validation checks")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
    return parser.parse_args()


def load_data(
    start_date: str,
    end_date: str,
    codes: Optional[List[str]] = None,
    sections: Optional[List[str]] = None,
    lazy: bool = False
) -> pl.DataFrame:
    """Load ML panel data with filters"""
    
    # In production, this would load from actual data store
    # For now, create sample data
    print(f"Loading data from {start_date} to {end_date}...")
    
    # Sample implementation
    dates = pl.date_range(
        pl.datetime.strptime(start_date, "%Y-%m-%d").date(),
        pl.datetime.strptime(end_date, "%Y-%m-%d").date(),
        "1d"
    )
    
    if codes is None:
        codes = ["1301", "1332", "1333", "1605", "1721"]
    
    # Create sample panel
    data = []
    for code in codes:
        for date in dates:
            data.append({
                "meta_code": code,
                "meta_date": date,
                "meta_section": "Prime",
                "px_returns_1d": 0.01,
                "px_volatility_20d": 0.2,
                "mkt_ret_1d": 0.005,
                "mkt_vol_20d": 0.15,
                "y_1d": 0.01,
                "y_5d": 0.05,
            })
    
    df = pl.DataFrame(data)
    
    if sections:
        df = df.filter(pl.col("meta_section").is_in(sections))
    
    return df.lazy() if lazy else df


def select_features(
    df: pl.DataFrame,
    feature_groups: List[str],
    exclude_features: Optional[List[str]] = None
) -> pl.DataFrame:
    """Select feature columns based on groups"""
    
    if "all" in feature_groups:
        selected_cols = df.columns
    else:
        selected_cols = ["meta_code", "meta_date", "meta_section"]
        
        group_prefixes = {
            "price": "px_",
            "market": "mkt_",
            "cross": "cross_",
            "flow": "flow_",
            "financial": "fin_"
        }
        
        for group in feature_groups:
            if group in group_prefixes:
                prefix = group_prefixes[group]
                selected_cols.extend([col for col in df.columns if col.startswith(prefix)])
        
        # Always include targets
        selected_cols.extend([col for col in df.columns if col.startswith("y_")])
    
    if exclude_features:
        selected_cols = [col for col in selected_cols if col not in exclude_features]
    
    return df.select(selected_cols)


def process_missing_values(
    df: pl.DataFrame,
    method: str = "forward"
) -> pl.DataFrame:
    """Handle missing values"""
    
    print(f"Processing missing values with method: {method}")
    
    if method == "forward":
        # Forward fill per stock
        df = df.sort(["meta_code", "meta_date"])
        for col in df.columns:
            if col not in ["meta_code", "meta_date", "meta_section"]:
                df = df.with_columns(
                    pl.col(col).forward_fill().over("meta_code").alias(col)
                )
    
    elif method == "median":
        # Cross-sectional median per date
        for col in df.columns:
            if col not in ["meta_code", "meta_date", "meta_section"]:
                median = pl.col(col).median().over("meta_date")
                df = df.with_columns(
                    pl.col(col).fill_null(median).alias(col)
                )
    
    elif method == "zero":
        df = df.fill_null(0)
    
    return df


def apply_normalization(
    df: pl.DataFrame,
    winsorize_pct: float = 0.01,
    verbose: bool = False
) -> pl.DataFrame:
    """Apply cross-sectional normalization"""
    
    if verbose:
        print("Applying cross-sectional normalization...")
    
    feature_cols = [col for col in df.columns 
                   if not col.startswith("meta_") and not col.startswith("y_")]
    
    df = CrossSectionalNormalizer.normalize_daily(
        df,
        feature_cols,
        method="zscore",
        robust=True,
        winsorize_pct=winsorize_pct
    )
    
    return df


def validate_dataset(df: pl.DataFrame, verbose: bool = False) -> dict:
    """Run validation checks on dataset"""
    
    print("Running validation checks...")
    
    validator = SchemaValidator()
    results = {}
    
    # Check for duplicates
    duplicates = len(df) - len(df.unique(["meta_code", "meta_date"]))
    results["duplicates"] = duplicates
    
    # Check null rates
    null_rates = {}
    for col in df.columns:
        null_rate = df[col].null_count() / len(df)
        if null_rate > 0:
            null_rates[col] = null_rate
    results["null_rates"] = null_rates
    
    # Check data leakage
    leakage = validator.check_data_leakage(df)
    results["potential_leakage"] = leakage["has_leakage"]
    
    if verbose:
        print(f"Duplicates: {duplicates}")
        print(f"Columns with nulls: {len(null_rates)}")
        print(f"Potential leakage: {leakage['has_leakage']}")
    
    return results


def save_dataset(
    df: pl.DataFrame,
    output_dir: str,
    output_format: str,
    partition_by: str = "none",
    start_date: str = None,
    end_date: str = None
):
    """Save dataset to disk"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if partition_by == "none":
        # Single file
        filename = f"ml_dataset_{start_date}_{end_date}_{timestamp}.{output_format}"
        filepath = output_path / filename
        
        if output_format == "parquet":
            df.write_parquet(filepath)
        elif output_format == "csv":
            df.write_csv(filepath)
        elif output_format == "feather":
            df.write_ipc(filepath)
        
        print(f"Dataset saved to: {filepath}")
    
    else:
        # Partitioned output
        if partition_by == "date":
            partition_col = "meta_date"
        elif partition_by == "code":
            partition_col = "meta_code"
        elif partition_by == "month":
            df = df.with_columns(
                pl.col("meta_date").dt.strftime("%Y-%m").alias("_month")
            )
            partition_col = "_month"
        
        # Write partitioned
        base_dir = output_path / f"ml_dataset_{timestamp}"
        
        for partition_val in df[partition_col].unique():
            partition_df = df.filter(pl.col(partition_col) == partition_val)
            
            partition_dir = base_dir / f"{partition_col}={partition_val}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            
            filepath = partition_dir / f"data.{output_format}"
            
            if output_format == "parquet":
                partition_df.write_parquet(filepath)
            elif output_format == "csv":
                partition_df.write_csv(filepath)
        
        print(f"Partitioned dataset saved to: {base_dir}")


def main():
    args = parse_args()
    
    if args.verbose:
        print("ML Dataset Builder")
        print("=" * 50)
        print(f"Date range: {args.start_date} to {args.end_date}")
        print(f"Feature groups: {args.feature_groups}")
        print(f"Output format: {args.output_format}")
        print()
    
    # Load data
    df = load_data(
        args.start_date,
        args.end_date,
        args.codes,
        args.sections,
        args.lazy
    )
    
    # Select features
    df = select_features(df, args.feature_groups, args.exclude_features)
    
    # Process missing values
    if args.fill_method != "none":
        df = process_missing_values(df, args.fill_method)
    
    # Apply normalization
    if args.normalize:
        df = apply_normalization(df, args.winsorize, args.verbose)
    
    # Collect if lazy
    if args.lazy:
        if args.verbose:
            print("Collecting lazy DataFrame...")
        df = df.collect()
    
    # Validate
    if args.validate:
        validation_results = validate_dataset(df, args.verbose)
    
    # Save
    save_dataset(
        df,
        args.output_dir,
        args.output_format,
        args.partition_by,
        args.start_date,
        args.end_date
    )
    
    if args.verbose:
        print()
        print("Dataset creation complete!")
        print(f"Shape: {df.shape}")
        print(f"Memory usage: {df.estimated_size() / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()