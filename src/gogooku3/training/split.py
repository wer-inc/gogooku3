"""
Walk-forward splitting with embargo for time series data.
Prevents data leakage with proper temporal separation.
"""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from typing import Iterator, Tuple

@dataclass
class WalkForwardSplitterV2:
    """
    Walk-Forward分割器（embargo対応）
    - 時系列データの正しい分割
    - embargo_days でリーク防止
    """
    date_col: str = "date"
    embargo_days: int = 20
    min_train_days: int = 252  # 約1年

    def split(self, df: pd.DataFrame, n_splits: int = 3) -> Iterator[Tuple[pd.Index, pd.Index]]:
        """Generate walk-forward splits with embargo."""
        dates = pd.to_datetime(df[self.date_col].sort_values().unique())
        
        if len(dates) < self.min_train_days + self.embargo_days + 30:
            raise ValueError(f"Insufficient data: need at least {self.min_train_days + self.embargo_days + 30} days")
        
        total_days = len(dates)
        test_days = (total_days - self.min_train_days) // n_splits
        
        for i in range(n_splits):
            train_end_idx = self.min_train_days + i * test_days
            train_end_date = dates[train_end_idx]
            
            test_start_date = train_end_date + pd.Timedelta(days=self.embargo_days)
            test_end_idx = min(train_end_idx + test_days + self.embargo_days, total_days - 1)
            test_end_date = dates[test_end_idx]
            
            train_mask = pd.to_datetime(df[self.date_col]) <= train_end_date
            test_mask = (pd.to_datetime(df[self.date_col]) >= test_start_date) & \
                       (pd.to_datetime(df[self.date_col]) <= test_end_date)
            
            train_idx = df.index[train_mask]
            test_idx = df.index[test_mask]
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

    def validate_split(self, df: pd.DataFrame) -> dict:
        """Validate that splits don't have temporal overlap."""
        splits = list(self.split(df))
        overlaps = []
        
        for i, (train_idx, test_idx) in enumerate(splits):
            train_dates = pd.to_datetime(df.loc[train_idx, self.date_col])
            test_dates = pd.to_datetime(df.loc[test_idx, self.date_col])
            
            train_max = train_dates.max()
            test_min = test_dates.min()
            
            actual_embargo = (test_min - train_max).days
            if actual_embargo < self.embargo_days:
                overlaps.append({
                    "split": i,
                    "train_max": train_max,
                    "test_min": test_min,
                    "actual_embargo": actual_embargo
                })
        
        return {
            "n_splits": len(splits),
            "overlaps": overlaps,
            "embargo_violations": len(overlaps)
        }
