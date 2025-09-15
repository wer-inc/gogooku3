from __future__ import annotations

"""Short position features extraction from J-Quants short selling data.

Extracts short interest features:
- short_ratio: Short shares / total shares outstanding
- short_ratio_change: Week-over-week change in short ratio
- days_to_cover: Short shares / average daily volume
- short_squeeze_risk: Composite risk score
- short_trend: Moving average of short ratio
"""

import polars as pl
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ShortPositionFeatureExtractor:
    """Extract short position features from J-Quants short selling data."""
    
    def __init__(self, 
                 ma_windows: List[int] = [5, 20],
                 volume_lookback: int = 20):
        """Initialize short position feature extractor.
        
        Args:
            ma_windows: Moving average windows for trend calculation
            volume_lookback: Days to look back for average volume
        """
        self.ma_windows = ma_windows
        self.volume_lookback = volume_lookback
    
    def extract_features(self,
                        df_base: pl.DataFrame,
                        df_shorts: pl.DataFrame,
                        df_listed: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """Extract short position features and merge with base dataset.
        
        Args:
            df_base: Base dataset with columns [code, date, volume]
            df_shorts: Short selling positions from J-Quants API
                      with columns [code, date, short_shares, ratio]
            df_listed: Listed info with shares_outstanding (optional)
        
        Returns:
            DataFrame with short position features added
        """
        if df_shorts.is_empty():
            logger.warning("No short position data provided, adding null features")
            return self._add_null_features(df_base)
        
        # Ensure date columns are datetime
        if 'date' in df_base.columns and df_base['date'].dtype == pl.Utf8:
            df_base = df_base.with_columns(pl.col('date').str.to_date())
        
        if 'date' in df_shorts.columns and df_shorts['date'].dtype == pl.Utf8:
            df_shorts = df_shorts.with_columns(pl.col('date').str.to_date())
        
        # Process each stock
        features = []
        
        for code in df_base['code'].unique().to_list():
            df_stock = df_base.filter(pl.col('code') == code)
            df_short_stock = df_shorts.filter(pl.col('code') == code)
            
            if df_short_stock.is_empty():
                stock_features = self._add_null_features(df_stock)
            else:
                stock_features = self._extract_stock_features(
                    df_stock, df_short_stock, df_listed
                )
            
            features.append(stock_features)
        
        # Combine all features
        df_features = pl.concat(features, how='vertical')
        
        return df_features.sort(['code', 'date'])
    
    def _extract_stock_features(self,
                              df_stock: pl.DataFrame,
                              df_shorts: pl.DataFrame,
                              df_listed: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """Extract short features for a single stock.
        
        Args:
            df_stock: Stock data with dates and volume
            df_shorts: Short position data for this stock
            df_listed: Listed info with shares outstanding
        
        Returns:
            DataFrame with short features
        """
        # Sort by date
        df_shorts = df_shorts.sort('date')
        
        # Join short data with stock data
        df_merged = df_stock.join(
            df_shorts.select(['date', 'short_shares', 'ratio']),
            on='date',
            how='left'
        )
        
        # Forward fill short position data (positions reported weekly)
        df_merged = df_merged.with_columns([
            pl.col('short_shares').forward_fill(),
            pl.col('ratio').forward_fill()
        ])
        
        # Calculate short ratio (already provided in data)
        df_merged = df_merged.with_columns(
            pl.col('ratio').alias('short_ratio')
        )
        
        # Calculate short ratio change (week-over-week)
        df_merged = df_merged.with_columns(
            (pl.col('short_ratio') - pl.col('short_ratio').shift(5)).alias('short_ratio_change_5d'),
        )
        
        # Calculate days to cover (short shares / average daily volume)
        if 'volume' in df_stock.columns:
            df_merged = df_merged.with_columns(
                pl.col('volume').rolling_mean(window_size=self.volume_lookback).alias('avg_volume')
            )
            df_merged = df_merged.with_columns(
                (pl.col('short_shares') / pl.col('avg_volume')).alias('days_to_cover')
            )
        else:
            df_merged = df_merged.with_columns(
                pl.lit(None).cast(pl.Float32).alias('days_to_cover')
            )
        
        # Calculate short trends (moving averages)
        for window in self.ma_windows:
            df_merged = df_merged.with_columns(
                pl.col('short_ratio').rolling_mean(window_size=window).alias(f'short_ratio_ma{window}')
            )
        
        # Calculate short squeeze risk score
        df_merged = self._calculate_squeeze_risk(df_merged)
        
        # Select relevant columns
        feature_cols = ['code', 'date', 'short_ratio', 'short_ratio_change_5d', 
                       'days_to_cover', 'short_squeeze_risk']
        feature_cols.extend([f'short_ratio_ma{w}' for w in self.ma_windows])
        
        # Ensure all columns exist
        for col in feature_cols:
            if col not in df_merged.columns:
                df_merged = df_merged.with_columns(
                    pl.lit(None).cast(pl.Float32).alias(col)
                )
        
        return df_merged.select(feature_cols)
    
    def _calculate_squeeze_risk(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate short squeeze risk score.
        
        Short squeeze risk is higher when:
        - Short ratio is high (many shorts)
        - Days to cover is high (hard to exit)
        - Short ratio is increasing recently
        
        Args:
            df: DataFrame with short features
        
        Returns:
            DataFrame with squeeze risk score added
        """
        # Normalize components to 0-1 scale
        df = df.with_columns([
            # High short ratio component (percentile rank)
            pl.col('short_ratio').rank(method='average').alias('short_ratio_rank'),
            # High days to cover component
            pl.col('days_to_cover').rank(method='average').alias('days_cover_rank'),
            # Increasing short ratio component
            pl.when(pl.col('short_ratio_change_5d') > 0)
              .then(pl.col('short_ratio_change_5d').rank(method='average'))
              .otherwise(0)
              .alias('short_increase_rank')
        ])
        
        # Normalize ranks to 0-1
        n_rows = len(df)
        df = df.with_columns([
            (pl.col('short_ratio_rank') / n_rows).alias('short_ratio_score'),
            (pl.col('days_cover_rank') / n_rows).alias('days_cover_score'),
            (pl.col('short_increase_rank') / n_rows).alias('short_increase_score'),
        ])
        
        # Composite squeeze risk score (weighted average)
        df = df.with_columns(
            (
                pl.col('short_ratio_score') * 0.4 +
                pl.col('days_cover_score') * 0.4 +
                pl.col('short_increase_score') * 0.2
            ).alias('short_squeeze_risk')
        )
        
        # Drop intermediate columns
        cols_to_drop = ['short_ratio_rank', 'days_cover_rank', 'short_increase_rank',
                       'short_ratio_score', 'days_cover_score', 'short_increase_score']
        df = df.drop(cols_to_drop)
        
        return df
    
    def _add_null_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add null short position features when no data available.
        
        Args:
            df: Input dataframe
        
        Returns:
            DataFrame with null short features
        """
        null_cols = [
            pl.lit(None).cast(pl.Float32).alias('short_ratio'),
            pl.lit(None).cast(pl.Float32).alias('short_ratio_change_5d'),
            pl.lit(None).cast(pl.Float32).alias('days_to_cover'),
            pl.lit(None).cast(pl.Float32).alias('short_squeeze_risk'),
        ]
        
        for window in self.ma_windows:
            null_cols.append(
                pl.lit(None).cast(pl.Float32).alias(f'short_ratio_ma{window}')
            )
        
        return df.with_columns(null_cols)


def add_short_position_features(df_base: pl.DataFrame,
                              fetcher,
                              start_date: str,
                              end_date: str,
                              ma_windows: List[int] = [5, 20]) -> pl.DataFrame:
    """Convenience function to add short position features to dataset.

    Args:
        df_base: Base dataset with [code, date, volume] columns
        fetcher: JQuantsAsyncFetcher instance
        start_date: Start date for short data
        end_date: End date for short data
        ma_windows: Moving average windows for trends

    Returns:
        DataFrame with short position features added
    """
    import asyncio
    import aiohttp
    from datetime import datetime, timedelta

    # Get unique codes for filtering
    codes = df_base['code'].unique().to_list()

    # Fetch short position data for all stocks
    async def fetch_shorts():
        async with aiohttp.ClientSession() as session:
            # Authenticate first
            await fetcher.authenticate(session)

            # Fetch short selling data for the date range
            df_shorts = await fetcher.get_short_selling_positions(
                session,
                from_date=start_date,
                to_date=end_date
            )
            return df_shorts

    # Run async fetch
    try:
        loop = asyncio.get_running_loop()
        # We're already in an async context
        import nest_asyncio
        nest_asyncio.apply()
        df_shorts = asyncio.run(fetch_shorts())
    except RuntimeError:
        # No event loop running
        df_shorts = asyncio.run(fetch_shorts())

    # Filter to only codes in our dataset
    if not df_shorts.is_empty() and 'Code' in df_shorts.columns:
        # Rename Code to code for consistency
        df_shorts = df_shorts.rename({'Code': 'code', 'Date': 'date'})
        df_shorts = df_shorts.filter(pl.col('code').is_in(codes))

    # Extract features
    extractor = ShortPositionFeatureExtractor(ma_windows=ma_windows)

    return extractor.extract_features(df_base, df_shorts)