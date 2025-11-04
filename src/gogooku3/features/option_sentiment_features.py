from __future__ import annotations

"""Index option sentiment features extraction from J-Quants option data.

Extracts market sentiment indicators from Nikkei 225 options:
- put_call_ratio: Put volume / Call volume (fear gauge)
- implied_volatility_skew: OTM put IV - OTM call IV (tail risk)
- option_flow_imbalance: Net option flow direction
- term_structure_slope: Near-term vs far-term IV spread
- smart_money_indicator: Large trade positioning
- volatility_risk_premium: IV vs realized volatility
"""

import logging
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)


class OptionSentimentExtractor:
    """Extract sentiment features from index option data."""

    def __init__(self,
                 iv_percentile_window: int = 60,
                 flow_window: int = 5,
                 smart_money_threshold: float = 0.9):
        """Initialize option sentiment extractor.

        Args:
            iv_percentile_window: Window for IV percentile calculation
            flow_window: Window for option flow analysis
            smart_money_threshold: Volume percentile for smart money detection
        """
        self.iv_percentile_window = iv_percentile_window
        self.flow_window = flow_window
        self.smart_money_threshold = smart_money_threshold

    def extract_features(self,
                        df_base: pl.DataFrame,
                        df_options: pl.DataFrame,
                        df_spot: pl.DataFrame | None = None) -> pl.DataFrame:
        """Extract option sentiment features.

        Args:
            df_base: Base dataset with columns [date]
            df_options: Index option data from J-Quants
            df_spot: Spot index prices for moneyness calculation

        Returns:
            DataFrame with option sentiment features added
        """
        if df_options.is_empty():
            logger.warning("No option data provided, adding null features")
            return self._add_null_features(df_base)

        # Ensure date columns are datetime
        if 'date' in df_base.columns and df_base['date'].dtype == pl.Utf8:
            df_base = df_base.with_columns(pl.col('date').str.to_date())

        if 'Date' in df_options.columns:
            df_options = df_options.rename({'Date': 'date'})
        if 'date' in df_options.columns and df_options['date'].dtype == pl.Utf8:
            df_options = df_options.with_columns(pl.col('date').str.to_date())

        # Calculate daily aggregates
        daily_features = self._calculate_daily_aggregates(df_options, df_spot)

        # Join with base dataset
        df_features = df_base.join(
            daily_features,
            on='date',
            how='left'
        )

        # Add rolling features
        df_features = self._add_rolling_features(df_features)

        return df_features

    def _calculate_daily_aggregates(self,
                                   df_options: pl.DataFrame,
                                   df_spot: pl.DataFrame | None = None) -> pl.DataFrame:
        """Calculate daily option sentiment aggregates.

        Args:
            df_options: Option data
            df_spot: Spot prices for moneyness

        Returns:
            DataFrame with daily aggregates
        """
        # Identify put/call from option code or other fields
        df_options = self._identify_option_type(df_options)

        # Group by date and calculate metrics
        daily_aggs = df_options.group_by('date').agg([
            # Put-Call Ratio
            (pl.col('Volume').filter(pl.col('option_type') == 'PUT').sum() /
             (pl.col('Volume').filter(pl.col('option_type') == 'CALL').sum() + 1e-10))
            .alias('put_call_ratio'),

            # Volume-weighted average IV for puts and calls
            ((pl.col('ImpliedVolatility') * pl.col('Volume'))
             .filter(pl.col('option_type') == 'PUT').sum() /
             (pl.col('Volume').filter(pl.col('option_type') == 'PUT').sum() + 1e-10))
            .alias('put_iv_weighted'),

            ((pl.col('ImpliedVolatility') * pl.col('Volume'))
             .filter(pl.col('option_type') == 'CALL').sum() /
             (pl.col('Volume').filter(pl.col('option_type') == 'CALL').sum() + 1e-10))
            .alias('call_iv_weighted'),

            # Total option volume
            pl.col('Volume').sum().alias('total_option_volume'),

            # Open interest imbalance
            ((pl.col('OpenInterest').filter(pl.col('option_type') == 'CALL').sum() -
              pl.col('OpenInterest').filter(pl.col('option_type') == 'PUT').sum()) /
             (pl.col('OpenInterest').sum() + 1e-10))
            .alias('oi_call_put_imbalance'),

            # Average IV across all options
            pl.col('ImpliedVolatility').mean().alias('avg_iv'),

            # IV standard deviation (dispersion)
            pl.col('ImpliedVolatility').std().alias('iv_dispersion'),
        ])

        # Calculate IV skew
        daily_aggs = daily_aggs.with_columns(
            (pl.col('put_iv_weighted') - pl.col('call_iv_weighted')).alias('iv_skew')
        )

        # Add term structure if multiple expirations exist
        daily_aggs = self._add_term_structure(daily_aggs, df_options)

        # Add smart money indicator
        daily_aggs = self._add_smart_money_indicator(daily_aggs, df_options)

        return daily_aggs

    def _identify_option_type(self, df: pl.DataFrame) -> pl.DataFrame:
        """Identify put/call type from option data.

        Args:
            df: Option dataframe

        Returns:
            DataFrame with option_type column added
        """
        # Check if we have a PutCallDivision column or similar
        if 'PutCallDivision' in df.columns:
            df = df.with_columns(
                pl.when(pl.col('PutCallDivision') == 'P')
                  .then(pl.lit('PUT'))
                  .when(pl.col('PutCallDivision') == 'C')
                  .then(pl.lit('CALL'))
                  .otherwise(pl.lit('UNKNOWN'))
                  .alias('option_type')
            )
        elif 'Code' in df.columns:
            # Try to infer from option code (e.g., codes ending in P or C)
            df = df.with_columns(
                pl.when(pl.col('Code').str.contains('P$'))
                  .then(pl.lit('PUT'))
                  .when(pl.col('Code').str.contains('C$'))
                  .then(pl.lit('CALL'))
                  .otherwise(pl.lit('UNKNOWN'))
                  .alias('option_type')
            )
        else:
            # Default: assume roughly equal put/call based on strike vs spot
            df = df.with_columns(
                pl.lit('UNKNOWN').alias('option_type')
            )

        return df

    def _add_term_structure(self,
                           daily_aggs: pl.DataFrame,
                           df_options: pl.DataFrame) -> pl.DataFrame:
        """Add term structure slope features.

        Args:
            daily_aggs: Daily aggregate features
            df_options: Raw option data with expirations

        Returns:
            DataFrame with term structure features
        """
        if 'ContractMonth' not in df_options.columns:
            daily_aggs = daily_aggs.with_columns(
                pl.lit(None).cast(pl.Float32).alias('term_structure_slope')
            )
            return daily_aggs

        # Calculate near-term and far-term IV
        term_structure = df_options.group_by(['date', 'ContractMonth']).agg([
            pl.col('ImpliedVolatility').mean().alias('avg_iv_by_expiry'),
            pl.col('Volume').sum().alias('volume_by_expiry')
        ])

        # Sort by expiration and get first (near) and last (far) for each date
        term_structure = term_structure.sort(['date', 'ContractMonth'])

        near_term = term_structure.group_by('date').first()
        far_term = term_structure.group_by('date').last()

        # Calculate slope
        slopes = near_term.join(
            far_term.select([
                pl.col('date'),
                pl.col('avg_iv_by_expiry').alias('far_iv')
            ]),
            on='date'
        ).with_columns(
            (pl.col('far_iv') - pl.col('avg_iv_by_expiry')).alias('term_structure_slope')
        ).select(['date', 'term_structure_slope'])

        # Join back to daily aggregates
        daily_aggs = daily_aggs.join(slopes, on='date', how='left')

        return daily_aggs

    def _add_smart_money_indicator(self,
                                  daily_aggs: pl.DataFrame,
                                  df_options: pl.DataFrame) -> pl.DataFrame:
        """Add smart money positioning indicator.

        Args:
            daily_aggs: Daily aggregate features
            df_options: Raw option data

        Returns:
            DataFrame with smart money indicator
        """
        # Calculate large trade indicator (high volume trades)
        if 'Volume' not in df_options.columns:
            daily_aggs = daily_aggs.with_columns(
                pl.lit(None).cast(pl.Float32).alias('smart_money_indicator')
            )
            return daily_aggs

        # Get volume percentiles by date
        volume_stats = df_options.group_by('date').agg([
            pl.col('Volume').quantile(self.smart_money_threshold).alias('volume_threshold')
        ])

        # Identify large trades
        df_large = df_options.join(volume_stats, on='date')
        df_large = df_large.filter(pl.col('Volume') >= pl.col('volume_threshold'))

        # Calculate smart money positioning
        if not df_large.is_empty():
            smart_money = df_large.group_by('date').agg([
                # Net positioning of large trades
                ((pl.col('Volume').filter(pl.col('option_type') == 'CALL').sum() -
                  pl.col('Volume').filter(pl.col('option_type') == 'PUT').sum()) /
                 (pl.col('Volume').sum() + 1e-10))
                .alias('smart_money_indicator')
            ])

            daily_aggs = daily_aggs.join(
                smart_money.select(['date', 'smart_money_indicator']),
                on='date',
                how='left'
            )
        else:
            daily_aggs = daily_aggs.with_columns(
                pl.lit(None).cast(pl.Float32).alias('smart_money_indicator')
            )

        return daily_aggs

    def _add_rolling_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add rolling window features.

        Args:
            df: DataFrame with daily features

        Returns:
            DataFrame with rolling features added
        """
        df = df.sort('date')

        # IV percentile rank
        if 'avg_iv' in df.columns:
            df = df.with_columns(
                pl.col('avg_iv')
                  .rolling_quantile(quantile=0.5, window_size=self.iv_percentile_window)
                  .alias('iv_median')
            )

            df = df.with_columns(
                pl.when(pl.col('iv_median') > 0)
                  .then((pl.col('avg_iv') - pl.col('iv_median')) / pl.col('iv_median'))
                  .otherwise(0)
                  .alias('iv_percentile_rank')
            )

        # Option flow momentum
        if 'total_option_volume' in df.columns:
            df = df.with_columns(
                pl.col('total_option_volume')
                  .rolling_mean(window_size=self.flow_window)
                  .alias('option_flow_ma')
            )

            df = df.with_columns(
                ((pl.col('total_option_volume') - pl.col('option_flow_ma')) /
                 (pl.col('option_flow_ma') + 1e-10))
                .alias('option_flow_momentum')
            )

        # Put-call ratio momentum
        if 'put_call_ratio' in df.columns:
            df = df.with_columns(
                pl.col('put_call_ratio')
                  .rolling_mean(window_size=self.flow_window)
                  .alias('pcr_ma')
            )

            df = df.with_columns(
                (pl.col('put_call_ratio') - pl.col('pcr_ma')).alias('pcr_momentum')
            )

        return df

    def _add_null_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add null option sentiment features when no data available.

        Args:
            df: Input dataframe

        Returns:
            DataFrame with null features
        """
        return df.with_columns([
            pl.lit(None).cast(pl.Float32).alias('put_call_ratio'),
            pl.lit(None).cast(pl.Float32).alias('iv_skew'),
            pl.lit(None).cast(pl.Float32).alias('oi_call_put_imbalance'),
            pl.lit(None).cast(pl.Float32).alias('avg_iv'),
            pl.lit(None).cast(pl.Float32).alias('iv_dispersion'),
            pl.lit(None).cast(pl.Float32).alias('term_structure_slope'),
            pl.lit(None).cast(pl.Float32).alias('smart_money_indicator'),
            pl.lit(None).cast(pl.Float32).alias('iv_percentile_rank'),
            pl.lit(None).cast(pl.Float32).alias('option_flow_momentum'),
            pl.lit(None).cast(pl.Float32).alias('pcr_momentum'),
        ])


def add_option_sentiment_features(df_base: pl.DataFrame,
                                 fetcher: Any,
                                 start_date: str,
                                 end_date: str) -> pl.DataFrame:
    """Convenience function to add option sentiment features.

    Args:
        df_base: Base dataset with [date] columns
        fetcher: JQuantsAsyncFetcher instance
        start_date: Start date for option data
        end_date: End date for option data

    Returns:
        DataFrame with option sentiment features added
    """
    import asyncio

    import aiohttp

    async def fetch_option_data():
        async with aiohttp.ClientSession() as session:
            # Authenticate first
            await fetcher.authenticate(session)

            # Fetch index option data
            df_options = await fetcher.get_index_option(
                session,
                from_date=start_date,
                to_date=end_date
            )

            return df_options

    # Run async fetch
    try:
        asyncio.get_running_loop()
        try:
            import nest_asyncio  # type: ignore
            nest_asyncio.apply()
        except Exception:
            pass
        df_options = asyncio.run(fetch_option_data())
    except RuntimeError:
        df_options = asyncio.run(fetch_option_data())

    # Extract features
    extractor = OptionSentimentExtractor()

    # Remove code column if present (these are market-wide features)
    if 'code' in df_base.columns:
        df_dates = df_base.select('date').unique()
        df_with_sentiment = extractor.extract_features(df_dates, df_options)
        # Join back to original data
        df_result = df_base.join(df_with_sentiment, on='date', how='left')
        return df_result
    else:
        return extractor.extract_features(df_base, df_options)
