"""Data scaling and normalization components."""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CrossSectionalNormalizerV2:
    """
    Cross-sectional normalizer for financial data.
    
    Normalizes features across stocks at each time point to prevent look-ahead bias.
    """
    
    def __init__(
        self,
        method: str = "zscore",
        clip_outliers: bool = True,
        outlier_std: float = 3.0,
        min_observations: int = 10
    ):
        """
        Initialize cross-sectional normalizer.
        
        Args:
            method: Normalization method ('zscore', 'minmax', 'robust')
            clip_outliers: Whether to clip outliers
            outlier_std: Standard deviations for outlier clipping
            min_observations: Minimum observations required for normalization
        """
        self.method = method
        self.clip_outliers = clip_outliers
        self.outlier_std = outlier_std
        self.min_observations = min_observations
        self.is_fitted = False
        
        logger.info(f"Initialized CrossSectionalNormalizerV2 with method={method}")
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform data with cross-sectional normalization.
        
        Args:
            df: DataFrame with datetime index and stock columns
            
        Returns:
            Normalized DataFrame
        """
        logger.info("Applying cross-sectional normalization...")
        
        if df.empty:
            return df
        
        normalized_df = df.copy()
        
        for timestamp in df.index:
            row_data = df.loc[timestamp]
            
            valid_data = row_data.dropna()
            if len(valid_data) < self.min_observations:
                continue
            
            if self.method == "zscore":
                mean_val = valid_data.mean()
                std_val = valid_data.std()
                if std_val > 0:
                    normalized_values = (valid_data - mean_val) / std_val
                else:
                    normalized_values = valid_data - mean_val
            elif self.method == "minmax":
                min_val = valid_data.min()
                max_val = valid_data.max()
                if max_val > min_val:
                    normalized_values = (valid_data - min_val) / (max_val - min_val)
                else:
                    normalized_values = valid_data * 0
            elif self.method == "robust":
                median_val = valid_data.median()
                mad_val = (valid_data - median_val).abs().median()
                if mad_val > 0:
                    normalized_values = (valid_data - median_val) / mad_val
                else:
                    normalized_values = valid_data - median_val
            else:
                normalized_values = valid_data
            
            if self.clip_outliers:
                normalized_values = normalized_values.clip(
                    -self.outlier_std, self.outlier_std
                )
            
            normalized_df.loc[timestamp, normalized_values.index] = normalized_values
        
        self.is_fitted = True
        logger.info("Cross-sectional normalization completed")
        
        return normalized_df
    
    def validate_transform(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the normalization results.
        
        Args:
            df: Normalized DataFrame
            
        Returns:
            Validation results dictionary
        """
        validation = {
            "warnings": [],
            "errors": [],
            "stats": {}
        }
        
        if df.empty:
            validation["warnings"].append("Empty DataFrame provided")
            return validation
        
        extreme_count = (df.abs() > 5).sum().sum()
        if extreme_count > 0:
            validation["warnings"].append(f"Found {extreme_count} extreme values (|x| > 5)")
        
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            validation["warnings"].append(f"Found {nan_count} NaN values after normalization")
        
        validation["stats"] = {
            "mean": float(df.mean().mean()),
            "std": float(df.std().mean()),
            "min": float(df.min().min()),
            "max": float(df.max().max()),
            "nan_count": int(nan_count),
            "extreme_count": int(extreme_count)
        }
        
        return validation


class WalkForwardSplitterV2:
    """
    Walk-forward splitter for time series cross-validation.
    
    Creates time-based splits with embargo periods to prevent data leakage.
    """
    
    def __init__(
        self,
        embargo_days: int = 20,
        min_train_days: int = 252,
        test_days: int = 63,
        step_days: int = 21
    ):
        """
        Initialize walk-forward splitter.
        
        Args:
            embargo_days: Embargo period between train and test
            min_train_days: Minimum training period in days
            test_days: Test period length in days
            step_days: Step size between splits in days
        """
        self.embargo_days = embargo_days
        self.min_train_days = min_train_days
        self.test_days = test_days
        self.step_days = step_days
        
        logger.info(f"Initialized WalkForwardSplitterV2 with embargo={embargo_days} days")
    
    def split(self, df: pd.DataFrame) -> List[Tuple[pd.Index, pd.Index]]:
        """
        Generate walk-forward splits.
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            List of (train_idx, test_idx) tuples
        """
        if df.empty:
            return []
        
        dates = pd.to_datetime(df.index).sort_values()
        splits = []
        
        start_date = dates[0]
        end_date = dates[-1]
        
        current_date = start_date + timedelta(days=self.min_train_days)
        
        while current_date + timedelta(days=self.embargo_days + self.test_days) <= end_date:
            train_end = current_date
            train_start = start_date
            
            test_start = current_date + timedelta(days=self.embargo_days)
            test_end = test_start + timedelta(days=self.test_days)
            
            train_mask = (dates >= train_start) & (dates < train_end)
            test_mask = (dates >= test_start) & (dates < test_end)
            
            train_idx = dates[train_mask]
            test_idx = dates[test_mask]
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))
            
            current_date += timedelta(days=self.step_days)
        
        logger.info(f"Generated {len(splits)} walk-forward splits")
        return splits
    
    def get_n_splits(self, df: pd.DataFrame) -> int:
        """Get number of splits that would be generated."""
        return len(self.split(df))
