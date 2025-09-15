from __future__ import annotations

"""Earnings event features extraction from J-Quants earnings announcements.

Extracts decision event features:
- days_to_earnings: Days until next earnings announcement
- days_since_earnings: Days since last earnings announcement  
- is_earnings_week: Boolean flag for earnings week
- earnings_surprise: Actual vs forecast comparison
- earnings_momentum: Quarter-over-quarter growth
"""

import polars as pl
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class EarningsFeatureExtractor:
    """Extract earnings-related features from J-Quants earnings announcements."""
    
    def __init__(self, lookback_days: int = 90, lookahead_days: int = 90):
        """Initialize earnings feature extractor.
        
        Args:
            lookback_days: Days to look back for past earnings
            lookahead_days: Days to look ahead for future earnings
        """
        self.lookback_days = lookback_days
        self.lookahead_days = lookahead_days
    
    def extract_features(self, 
                        df_base: pl.DataFrame,
                        df_earnings: pl.DataFrame) -> pl.DataFrame:
        """Extract earnings features and merge with base dataset.
        
        Args:
            df_base: Base dataset with columns [code, date]
            df_earnings: Earnings announcements from J-Quants API
                        with columns [code, announced_date, fiscal_year, fiscal_quarter,
                                    actual_value, forecast_value, result_value]
        
        Returns:
            DataFrame with earnings features added
        """
        if df_earnings.is_empty():
            logger.warning("No earnings data provided, adding null features")
            return self._add_null_features(df_base)
        
        # Ensure date columns are datetime
        if 'date' in df_base.columns and df_base['date'].dtype == pl.Utf8:
            df_base = df_base.with_columns(pl.col('date').str.to_date())
        
        if 'announced_date' in df_earnings.columns and df_earnings['announced_date'].dtype == pl.Utf8:
            df_earnings = df_earnings.with_columns(pl.col('announced_date').str.to_date())
        
        # Extract features for each stock-date combination
        features = []
        
        for code in df_base['code'].unique().to_list():
            df_stock = df_base.filter(pl.col('code') == code)
            df_earn_stock = df_earnings.filter(pl.col('code') == code)
            
            if df_earn_stock.is_empty():
                # No earnings data for this stock
                stock_features = self._add_null_features(df_stock)
            else:
                stock_features = self._extract_stock_features(df_stock, df_earn_stock)
            
            features.append(stock_features)
        
        # Combine all features
        df_features = pl.concat(features, how='vertical')
        
        # Sort by original order
        df_features = df_features.sort(['code', 'date'])
        
        return df_features
    
    def _extract_stock_features(self, 
                               df_stock: pl.DataFrame,
                               df_earnings: pl.DataFrame) -> pl.DataFrame:
        """Extract earnings features for a single stock.
        
        Args:
            df_stock: Stock data with dates
            df_earnings: Earnings announcements for this stock
        
        Returns:
            DataFrame with earnings features
        """
        # Sort earnings by announcement date
        df_earnings = df_earnings.sort('announced_date')
        
        # For each date in stock data, calculate features
        dates = df_stock['date'].to_list()
        earnings_dates = df_earnings['announced_date'].to_list()
        
        days_to_earnings = []
        days_since_earnings = []
        is_earnings_week = []
        earnings_surprise = []
        earnings_momentum = []
        
        for date in dates:
            # Find next and previous earnings dates
            future_earnings = [e for e in earnings_dates if e > date]
            past_earnings = [e for e in earnings_dates if e <= date]
            
            # Days to next earnings
            if future_earnings:
                days_to = (future_earnings[0] - date).days
                days_to_earnings.append(min(days_to, self.lookahead_days))
            else:
                days_to_earnings.append(self.lookahead_days)
            
            # Days since last earnings
            if past_earnings:
                days_since = (date - past_earnings[-1]).days
                days_since_earnings.append(min(days_since, self.lookback_days))
            else:
                days_since_earnings.append(self.lookback_days)
            
            # Is earnings week (within 5 days)
            is_week = False
            if future_earnings and (future_earnings[0] - date).days <= 5:
                is_week = True
            elif past_earnings and (date - past_earnings[-1]).days <= 5:
                is_week = True
            is_earnings_week.append(is_week)
            
            # Calculate surprise and momentum for most recent earnings
            if past_earnings:
                recent_earn = df_earnings.filter(
                    pl.col('announced_date') == past_earnings[-1]
                ).to_dicts()[0]
                
                # Earnings surprise (if forecast available)
                if recent_earn.get('forecast_value') and recent_earn.get('actual_value'):
                    surprise = (recent_earn['actual_value'] - recent_earn['forecast_value']) / abs(recent_earn['forecast_value']) 
                    earnings_surprise.append(surprise)
                else:
                    earnings_surprise.append(None)
                
                # Earnings momentum (QoQ growth)
                if len(past_earnings) >= 2:
                    prev_earn = df_earnings.filter(
                        pl.col('announced_date') == past_earnings[-2]
                    ).to_dicts()
                    if prev_earn and recent_earn.get('actual_value') and prev_earn[0].get('actual_value'):
                        momentum = (recent_earn['actual_value'] - prev_earn[0]['actual_value']) / abs(prev_earn[0]['actual_value'])
                        earnings_momentum.append(momentum)
                    else:
                        earnings_momentum.append(None)
                else:
                    earnings_momentum.append(None)
            else:
                earnings_surprise.append(None)
                earnings_momentum.append(None)
        
        # Add features to dataframe
        df_features = df_stock.with_columns([
            pl.Series('days_to_earnings', days_to_earnings, dtype=pl.Int32),
            pl.Series('days_since_earnings', days_since_earnings, dtype=pl.Int32),
            pl.Series('is_earnings_week', is_earnings_week, dtype=pl.Boolean),
            pl.Series('earnings_surprise', earnings_surprise, dtype=pl.Float32),
            pl.Series('earnings_momentum', earnings_momentum, dtype=pl.Float32),
        ])
        
        return df_features
    
    def _add_null_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add null earnings features when no data available.
        
        Args:
            df: Input dataframe
        
        Returns:
            DataFrame with null earnings features
        """
        return df.with_columns([
            pl.lit(self.lookahead_days).cast(pl.Int32).alias('days_to_earnings'),
            pl.lit(self.lookback_days).cast(pl.Int32).alias('days_since_earnings'),
            pl.lit(False).alias('is_earnings_week'),
            pl.lit(None).cast(pl.Float32).alias('earnings_surprise'),
            pl.lit(None).cast(pl.Float32).alias('earnings_momentum'),
        ])


def add_earnings_features(df_base: pl.DataFrame,
                         fetcher,
                         start_date: str,
                         end_date: str,
                         lookback_days: int = 90,
                         lookahead_days: int = 90) -> pl.DataFrame:
    """Convenience function to add earnings features to dataset.

    Args:
        df_base: Base dataset with [code, date] columns
        fetcher: JQuantsAsyncFetcher instance
        start_date: Start date for earnings data
        end_date: End date for earnings data
        lookback_days: Days to look back for past earnings
        lookahead_days: Days to look ahead for future earnings

    Returns:
        DataFrame with earnings features added
    """
    import asyncio
    import aiohttp
    from datetime import datetime, timedelta

    # Expand date range to capture earnings outside the base range
    start_dt = datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=lookback_days)
    end_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=lookahead_days)

    # Get unique codes for filtering
    codes = df_base['code'].unique().to_list()

    # Fetch earnings data for all stocks in date range
    async def fetch_earnings():
        async with aiohttp.ClientSession() as session:
            # Authenticate first
            await fetcher.authenticate(session)

            # Fetch earnings data for the date range
            df_earnings = await fetcher.get_earnings_announcements(
                session,
                from_date=start_dt.strftime('%Y-%m-%d'),
                to_date=end_dt.strftime('%Y-%m-%d')
            )
            return df_earnings

    # Run async fetch
    try:
        loop = asyncio.get_running_loop()
        # We're already in an async context
        import nest_asyncio
        nest_asyncio.apply()
        df_earnings = asyncio.run(fetch_earnings())
    except RuntimeError:
        # No event loop running
        df_earnings = asyncio.run(fetch_earnings())

    # Filter to only codes in our dataset
    if not df_earnings.is_empty() and 'code' in df_earnings.columns:
        df_earnings = df_earnings.filter(pl.col('code').is_in(codes))

    # Extract features
    extractor = EarningsFeatureExtractor(
        lookback_days=lookback_days,
        lookahead_days=lookahead_days
    )

    return extractor.extract_features(df_base, df_earnings)