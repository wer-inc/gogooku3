#!/usr/bin/env python3
"""
Clean ML dataset by fixing Inf and extreme values
"""

import polars as pl
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_ml_dataset(input_path: str, output_path: str):
    """Clean ML dataset by handling Inf values and extreme outliers"""
    
    logger.info(f"Loading dataset from: {input_path}")
    df = pl.read_parquet(input_path)
    logger.info(f"Original shape: {df.shape}")
    
    # Check for Inf values
    numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
    
    # Fix Inf values
    for col in numeric_cols:
        # Count Inf values
        inf_mask = df[col].is_infinite()
        inf_count = inf_mask.sum()
        
        if inf_count > 0:
            logger.info(f"Found {inf_count} Inf values in column '{col}'")
            
            # Replace Inf with NaN first
            df = df.with_columns(
                pl.when(inf_mask)
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
            )
            
            # Then forward fill NaN values
            df = df.with_columns(
                pl.col(col).fill_null(strategy="forward").fill_null(strategy="backward")
            )
    
    # Check for extreme values
    for col in numeric_cols:
        if col in ['Code', 'Date', 'row_idx']:  # Skip non-feature columns
            continue
            
        # Calculate percentiles
        values = df[col].drop_nulls()
        if len(values) > 0:
            p01 = values.quantile(0.01)
            p99 = values.quantile(0.99)
            
            # Clip extreme values
            if p01 is not None and p99 is not None:
                df = df.with_columns(
                    pl.col(col).clip(p01, p99)
                )
    
    # Final check
    logger.info("\nFinal data quality check:")
    null_counts = df.null_count().sum_horizontal()[0]
    logger.info(f"Total null values: {null_counts}")
    
    # Check for remaining Inf values
    remaining_inf = 0
    for col in numeric_cols:
        inf_count = df[col].is_infinite().sum()
        if inf_count > 0:
            remaining_inf += inf_count
            logger.warning(f"Still has {inf_count} Inf values in '{col}'")
    
    logger.info(f"Total remaining Inf values: {remaining_inf}")
    
    # Save cleaned dataset
    logger.info(f"\nSaving cleaned dataset to: {output_path}")
    df.write_parquet(output_path)
    logger.info(f"Cleaned dataset saved: {df.shape}")
    
    return df

if __name__ == "__main__":
    input_file = "output/ml_dataset_latest_full.parquet"
    output_file = "output/ml_dataset_cleaned.parquet"
    
    clean_ml_dataset(input_file, output_file)
