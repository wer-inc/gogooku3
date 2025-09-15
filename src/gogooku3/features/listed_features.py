from __future__ import annotations

"""Enhanced listed info features extraction from J-Quants listed info.

Extracts company characteristic features:
- market_cap_log: Log-transformed market capitalization
- liquidity_score: Trading liquidity metric
- sector_momentum: Sector-level momentum
- market_segment_premium: Premium/discount vs market segment
- shares_float_ratio: Free float ratio estimation
"""

import polars as pl
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class ListedInfoFeatureExtractor:
    """Extract enhanced features from J-Quants listed info data."""
    
    def __init__(self,
                 sector_momentum_window: int = 20,
                 liquidity_window: int = 20):
        """Initialize listed info feature extractor.
        
        Args:
            sector_momentum_window: Window for sector momentum calculation
            liquidity_window: Window for liquidity score calculation
        """
        self.sector_momentum_window = sector_momentum_window
        self.liquidity_window = liquidity_window
    
    def extract_features(self,
                        df_base: pl.DataFrame,
                        df_listed: pl.DataFrame,
                        df_prices: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """Extract listed info features and merge with base dataset.
        
        Args:
            df_base: Base dataset with columns [code, date]
            df_listed: Listed info from J-Quants API with columns
                      [code, market_code, sector_33_name, market_cap, 
                       shares_outstanding, shares_float_ratio]
            df_prices: Daily prices for additional calculations (optional)
        
        Returns:
            DataFrame with listed info features added
        """
        if df_listed.is_empty():
            logger.warning("No listed info data provided, adding null features")
            return self._add_null_features(df_base)
        
        # Ensure date column is datetime
        if 'date' in df_base.columns and df_base['date'].dtype == pl.Utf8:
            df_base = df_base.with_columns(pl.col('date').str.to_date())
        
        # Add basic listed info features
        df_features = self._add_basic_features(df_base, df_listed)
        
        # Add market segment features
        df_features = self._add_market_segment_features(df_features, df_listed)
        
        # Add sector momentum features if prices available
        if df_prices is not None and not df_prices.is_empty():
            df_features = self._add_sector_momentum(df_features, df_listed, df_prices)
        
        # Add liquidity features if volume data available
        if 'volume' in df_base.columns:
            df_features = self._add_liquidity_features(df_features)
        
        return df_features
    
    def _add_basic_features(self, 
                           df_base: pl.DataFrame,
                           df_listed: pl.DataFrame) -> pl.DataFrame:
        """Add basic listed info features.
        
        Args:
            df_base: Base dataset
            df_listed: Listed info data
        
        Returns:
            DataFrame with basic features added
        """
        # Select relevant columns from listed info
        listed_cols = ['code', 'market_code', 'sector_33_code', 'sector_33_name',
                      'market_cap', 'shares_outstanding']
        
        # Filter to available columns
        available_cols = [col for col in listed_cols if col in df_listed.columns]
        df_listed_subset = df_listed.select(available_cols)
        
        # Join with base data
        df_features = df_base.join(df_listed_subset, on='code', how='left')
        
        # Add log market cap
        if 'market_cap' in df_features.columns:
            df_features = df_features.with_columns(
                pl.when(pl.col('market_cap') > 0)
                  .then(pl.col('market_cap').log())
                  .otherwise(None)
                  .alias('market_cap_log')
            )
        else:
            df_features = df_features.with_columns(
                pl.lit(None).cast(pl.Float32).alias('market_cap_log')
            )
        
        # Add shares float ratio (if not directly available, estimate)
        if 'shares_float_ratio' not in df_features.columns:
            # Simple estimation: assume 70% float for most stocks
            df_features = df_features.with_columns(
                pl.lit(0.7).cast(pl.Float32).alias('shares_float_ratio')
            )
        
        return df_features
    
    def _add_market_segment_features(self,
                                    df_features: pl.DataFrame,
                                    df_listed: pl.DataFrame) -> pl.DataFrame:
        """Add market segment premium/discount features.
        
        Args:
            df_features: DataFrame with basic features
            df_listed: Listed info data
        
        Returns:
            DataFrame with market segment features added
        """
        if 'market_code' not in df_features.columns:
            df_features = df_features.with_columns(
                pl.lit(None).cast(pl.Float32).alias('market_segment_premium')
            )
            return df_features
        
        # Calculate average metrics by market segment
        segment_stats = df_features.group_by(['date', 'market_code']).agg([
            pl.col('market_cap_log').mean().alias('segment_avg_mcap'),
            pl.col('market_cap_log').std().alias('segment_std_mcap'),
        ])
        
        # Join back to calculate premium/discount
        df_features = df_features.join(
            segment_stats,
            on=['date', 'market_code'],
            how='left'
        )
        
        # Calculate z-score within market segment
        df_features = df_features.with_columns(
            pl.when((pl.col('segment_std_mcap') > 0) & pl.col('market_cap_log').is_not_null())
              .then((pl.col('market_cap_log') - pl.col('segment_avg_mcap')) / pl.col('segment_std_mcap'))
              .otherwise(0)
              .alias('market_segment_premium')
        )
        
        # Drop intermediate columns
        df_features = df_features.drop(['segment_avg_mcap', 'segment_std_mcap'])
        
        return df_features
    
    def _add_sector_momentum(self,
                            df_features: pl.DataFrame,
                            df_listed: pl.DataFrame,
                            df_prices: pl.DataFrame) -> pl.DataFrame:
        """Add sector momentum features.
        
        Args:
            df_features: DataFrame with basic features
            df_listed: Listed info with sector information
            df_prices: Daily price data
        
        Returns:
            DataFrame with sector momentum added
        """
        if 'sector_33_code' not in df_features.columns:
            df_features = df_features.with_columns(
                pl.lit(None).cast(pl.Float32).alias('sector_momentum')
            )
            return df_features
        
        # Calculate returns if not present
        if 'return' not in df_prices.columns:
            df_prices = df_prices.with_columns(
                (pl.col('close').pct_change()).alias('return')
            )
        
        # Join sector info with prices
        df_prices_sector = df_prices.join(
            df_listed.select(['code', 'sector_33_code']),
            on='code',
            how='left'
        )
        
        # Calculate sector average returns
        sector_returns = df_prices_sector.group_by(['date', 'sector_33_code']).agg(
            pl.col('return').mean().alias('sector_return')
        )
        
        # Calculate rolling sector momentum
        sector_returns = sector_returns.sort(['sector_33_code', 'date'])
        sector_returns = sector_returns.with_columns(
            pl.col('sector_return')
              .rolling_mean(window_size=self.sector_momentum_window)
              .over('sector_33_code')
              .alias('sector_momentum')
        )
        
        # Join back to features
        df_features = df_features.join(
            sector_returns.select(['date', 'sector_33_code', 'sector_momentum']),
            on=['date', 'sector_33_code'],
            how='left'
        )
        
        return df_features
    
    def _add_liquidity_features(self, df_features: pl.DataFrame) -> pl.DataFrame:
        """Add liquidity score based on volume and market cap.
        
        Args:
            df_features: DataFrame with volume and market cap
        
        Returns:
            DataFrame with liquidity score added
        """
        if 'volume' not in df_features.columns:
            df_features = df_features.with_columns(
                pl.lit(None).cast(pl.Float32).alias('liquidity_score')
            )
            return df_features
        
        # Calculate dollar volume
        if 'close' in df_features.columns:
            df_features = df_features.with_columns(
                (pl.col('volume') * pl.col('close')).alias('dollar_volume')
            )
        else:
            df_features = df_features.with_columns(
                pl.col('volume').alias('dollar_volume')
            )
        
        # Calculate rolling average dollar volume
        df_features = df_features.sort(['code', 'date'])
        df_features = df_features.with_columns(
            pl.col('dollar_volume')
              .rolling_mean(window_size=self.liquidity_window)
              .over('code')
              .alias('avg_dollar_volume')
        )
        
        # Liquidity score: log(dollar_volume) / log(market_cap)
        df_features = df_features.with_columns(
            pl.when((pl.col('avg_dollar_volume') > 0) & (pl.col('market_cap') > 0))
              .then(pl.col('avg_dollar_volume').log() / pl.col('market_cap').log())
              .otherwise(None)
              .alias('liquidity_score')
        )
        
        # Drop intermediate columns
        df_features = df_features.drop(['dollar_volume', 'avg_dollar_volume'])
        
        return df_features
    
    def _add_null_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add null listed info features when no data available.
        
        Args:
            df: Input dataframe
        
        Returns:
            DataFrame with null features
        """
        return df.with_columns([
            pl.lit(None).cast(pl.Float32).alias('market_cap_log'),
            pl.lit(None).cast(pl.Float32).alias('shares_float_ratio'),
            pl.lit(None).cast(pl.Float32).alias('market_segment_premium'),
            pl.lit(None).cast(pl.Float32).alias('sector_momentum'),
            pl.lit(None).cast(pl.Float32).alias('liquidity_score'),
        ])


def add_listed_info_features(df_base: pl.DataFrame,
                            fetcher,
                            df_prices: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    """Convenience function to add listed info features to dataset.

    Args:
        df_base: Base dataset with [code, date] columns
        fetcher: JQuantsAsyncFetcher instance
        df_prices: Price data for momentum calculations (optional)

    Returns:
        DataFrame with listed info features added
    """
    import asyncio
    import aiohttp

    # Get unique codes for filtering
    codes = df_base['code'].unique().to_list()

    # Fetch listed info data for all stocks
    async def fetch_listed():
        async with aiohttp.ClientSession() as session:
            # Authenticate first
            await fetcher.authenticate(session)

            # Fetch listed info data
            df_listed = await fetcher.get_listed_info(session)
            return df_listed

    # Run async fetch
    try:
        loop = asyncio.get_running_loop()
        # We're already in an async context
        import nest_asyncio
        nest_asyncio.apply()
        df_listed = asyncio.run(fetch_listed())
    except RuntimeError:
        # No event loop running
        df_listed = asyncio.run(fetch_listed())

    # Filter to only codes in our dataset and rename columns
    if not df_listed.is_empty():
        # Rename columns for consistency
        if 'Code' in df_listed.columns:
            df_listed = df_listed.rename({'Code': 'code'})
        if 'code' in df_listed.columns:
            df_listed = df_listed.filter(pl.col('code').is_in(codes))

    # Extract features
    extractor = ListedInfoFeatureExtractor()

    return extractor.extract_features(df_base, df_listed, df_prices)