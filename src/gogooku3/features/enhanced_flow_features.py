from __future__ import annotations

"""Enhanced flow analysis features from J-Quants trades_spec data.

Extracts advanced institutional flow signals:
- institutional_accumulation: Net institutional buying pressure
- foreign_sentiment: Foreign investor positioning
- retail_divergence: Retail vs institutional divergence
- flow_persistence: Consistency of flow direction
- smart_flow_indicator: Quality of institutional flow
- flow_concentration: Concentration of buying/selling
"""

import polars as pl
from typing import Optional, List, Any
import logging

logger = logging.getLogger(__name__)


class EnhancedFlowAnalyzer:
    """Extract enhanced flow features from investor type data."""
    
    def __init__(self,
                 momentum_window: int = 20,
                 persistence_window: int = 10,
                 concentration_threshold: float = 0.7):
        """Initialize enhanced flow analyzer.
        
        Args:
            momentum_window: Window for flow momentum calculation
            persistence_window: Window for flow persistence measurement
            concentration_threshold: Threshold for flow concentration
        """
        self.momentum_window = momentum_window
        self.persistence_window = persistence_window
        self.concentration_threshold = concentration_threshold
    
    def extract_features(self,
                        df_base: pl.DataFrame,
                        df_flows: pl.DataFrame) -> pl.DataFrame:
        """Extract enhanced flow features.
        
        Args:
            df_base: Base dataset with columns [code, date]
            df_flows: Weekly trades_spec data from J-Quants
        
        Returns:
            DataFrame with enhanced flow features added
        """
        if df_flows.is_empty():
            logger.warning("No flow data provided, adding null features")
            return self._add_null_features(df_base)
        
        # Ensure date columns are datetime
        if 'date' in df_base.columns and df_base['date'].dtype == pl.Utf8:
            df_base = df_base.with_columns(pl.col('date').str.to_date())
        
        # Process flow data
        df_flows = self._process_flow_data(df_flows)
        
        # Calculate flow features by investor type
        df_features = self._calculate_investor_flows(df_base, df_flows)
        
        # Add divergence metrics
        df_features = self._add_divergence_metrics(df_features)
        
        # Add persistence and quality metrics
        df_features = self._add_persistence_metrics(df_features)
        
        # Add concentration metrics
        df_features = self._add_concentration_metrics(df_features)
        
        return df_features
    
    def _process_flow_data(self, df_flows: pl.DataFrame) -> pl.DataFrame:
        """Process and normalize flow data.
        
        Args:
            df_flows: Raw trades_spec data
        
        Returns:
            Processed flow DataFrame
        """
        # Rename columns for consistency
        rename_map = {
            'PublishedDate': 'date',
            'StartDate': 'week_start', 
            'EndDate': 'week_end',
            'Section': 'section',
            'Code': 'code'
        }
        
        for old, new in rename_map.items():
            if old in df_flows.columns:
                df_flows = df_flows.rename({old: new})
        
        # Ensure date is datetime
        if 'date' in df_flows.columns and df_flows['date'].dtype == pl.Utf8:
            df_flows = df_flows.with_columns(pl.col('date').str.to_date())
        
        # Identify investor type columns
        investor_columns = self._identify_investor_columns(df_flows)
        
        # Calculate net flows for each investor type
        for investor_type in investor_columns:
            buy_col = f'{investor_type}PurchaseValue'
            sell_col = f'{investor_type}SalesValue'
            
            if buy_col in df_flows.columns and sell_col in df_flows.columns:
                df_flows = df_flows.with_columns(
                    (pl.col(buy_col) - pl.col(sell_col)).alias(f'{investor_type}_net_flow')
                )
                
                # Calculate flow ratio (buy vs sell pressure)
                df_flows = df_flows.with_columns(
                    ((pl.col(buy_col) - pl.col(sell_col)) /
                     (pl.col(buy_col) + pl.col(sell_col) + 1e-10))
                    .alias(f'{investor_type}_flow_ratio')
                )
        
        return df_flows
    
    def _identify_investor_columns(self, df: pl.DataFrame) -> List[str]:
        """Identify investor type columns in the data.
        
        Args:
            df: Flow DataFrame
        
        Returns:
            List of investor type prefixes
        """
        investor_types = []
        
        # Common investor types in J-Quants data
        potential_types = [
            'Proprietary',  # Proprietary trading
            'Investment',   # Investment trusts
            'Business',     # Business corporations
            'Individual',   # Individual investors
            'Foreigners',   # Foreign investors
            'Securities',   # Securities companies
            'Other'         # Other institutions
        ]
        
        for inv_type in potential_types:
            if f'{inv_type}PurchaseValue' in df.columns:
                investor_types.append(inv_type)
        
        return investor_types
    
    def _calculate_investor_flows(self,
                                 df_base: pl.DataFrame,
                                 df_flows: pl.DataFrame) -> pl.DataFrame:
        """Calculate flow features by investor type.
        
        Args:
            df_base: Base dataset
            df_flows: Processed flow data
        
        Returns:
            DataFrame with investor flow features
        """
        # Join flow data with base (forward fill weekly to daily)
        df_merged = df_base.join(
            df_flows,
            on=['code', 'date'],
            how='left'
        )
        
        # Forward fill weekly values
        investor_types = self._identify_investor_columns(df_flows)
        
        fill_columns = []
        for inv_type in investor_types:
            fill_columns.extend([
                f'{inv_type}_net_flow',
                f'{inv_type}_flow_ratio'
            ])
        
        # Forward fill by code
        df_merged = df_merged.sort(['code', 'date'])
        for col in fill_columns:
            if col in df_merged.columns:
                df_merged = df_merged.with_columns(
                    pl.col(col).forward_fill().over('code')
                )
        
        # Calculate institutional accumulation (non-individual net flow)
        institutional_cols = [f'{t}_net_flow' for t in investor_types 
                            if t != 'Individual' and f'{t}_net_flow' in df_merged.columns]
        
        if institutional_cols:
            df_merged = df_merged.with_columns(
                sum([pl.col(c) for c in institutional_cols]).alias('institutional_accumulation')
            )
        
        # Foreign sentiment
        if 'Foreigners_flow_ratio' in df_merged.columns:
            df_merged = df_merged.with_columns(
                pl.col('Foreigners_flow_ratio').alias('foreign_sentiment')
            )
        
        return df_merged
    
    def _add_divergence_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add divergence metrics between investor types.
        
        Args:
            df: DataFrame with flow features
        
        Returns:
            DataFrame with divergence metrics added
        """
        # Retail vs Institutional divergence
        if 'Individual_flow_ratio' in df.columns and 'institutional_accumulation' in df.columns:
            df = df.with_columns(
                (pl.col('Individual_flow_ratio') * -1 * 
                 pl.col('institutional_accumulation').sign())
                .alias('retail_institutional_divergence')
            )
        
        # Foreign vs Domestic divergence
        if 'Foreigners_flow_ratio' in df.columns:
            domestic_cols = ['Individual_flow_ratio', 'Business_flow_ratio', 
                           'Investment_flow_ratio']
            available_domestic = [c for c in domestic_cols if c in df.columns]
            
            if available_domestic:
                df = df.with_columns(
                    sum([pl.col(c) for c in available_domestic]).alias('domestic_flow_ratio')
                )
                
                df = df.with_columns(
                    (pl.col('Foreigners_flow_ratio') - pl.col('domestic_flow_ratio'))
                    .alias('foreign_domestic_divergence')
                )
        
        return df
    
    def _add_persistence_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add flow persistence and quality metrics.
        
        Args:
            df: DataFrame with flow features
        
        Returns:
            DataFrame with persistence metrics added
        """
        df = df.sort(['code', 'date'])
        
        # Institutional flow persistence
        if 'institutional_accumulation' in df.columns:
            # Calculate rolling sign consistency
            df = df.with_columns(
                pl.col('institutional_accumulation').sign().alias('inst_flow_sign')
            )
            
            df = df.with_columns(
                pl.col('inst_flow_sign')
                  .rolling_mean(window_size=self.persistence_window)
                  .over('code')
                  .abs()
                  .alias('institutional_persistence')
            )
        
        # Foreign flow persistence
        if 'foreign_sentiment' in df.columns:
            df = df.with_columns(
                pl.col('foreign_sentiment').sign().alias('foreign_flow_sign')
            )
            
            df = df.with_columns(
                pl.col('foreign_flow_sign')
                  .rolling_mean(window_size=self.persistence_window)
                  .over('code')
                  .abs()
                  .alias('foreign_persistence')
            )
        
        # Smart flow indicator (persistent institutional buying)
        if 'institutional_persistence' in df.columns and 'institutional_accumulation' in df.columns:
            df = df.with_columns(
                (pl.col('institutional_persistence') * 
                 pl.col('institutional_accumulation').sign())
                .alias('smart_flow_indicator')
            )
        
        return df
    
    def _add_concentration_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add flow concentration metrics.
        
        Args:
            df: DataFrame with flow features
        
        Returns:
            DataFrame with concentration metrics added
        """
        # Calculate Herfindahl index for flow concentration
        investor_flow_cols = [col for col in df.columns if col.endswith('_net_flow')]
        
        if len(investor_flow_cols) >= 2:
            # Calculate total absolute flow
            df = df.with_columns(
                sum([pl.col(c).abs() for c in investor_flow_cols]).alias('total_abs_flow')
            )
            
            # Calculate concentration (Herfindahl-like)
            concentration_expr = pl.lit(0.0)
            for col in investor_flow_cols:
                concentration_expr = concentration_expr + \
                    (pl.col(col).abs() / (pl.col('total_abs_flow') + 1e-10)) ** 2
            
            df = df.with_columns(
                concentration_expr.alias('flow_concentration')
            )
            
            # Identify concentrated buying/selling
            df = df.with_columns(
                pl.when(pl.col('flow_concentration') > self.concentration_threshold)
                  .then(pl.col('institutional_accumulation').sign())
                  .otherwise(0)
                  .alias('concentrated_flow_signal')
            )
        
        return df
    
    def _add_null_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add null flow features when no data available.
        
        Args:
            df: Input dataframe
        
        Returns:
            DataFrame with null features
        """
        return df.with_columns([
            pl.lit(None).cast(pl.Float32).alias('institutional_accumulation'),
            pl.lit(None).cast(pl.Float32).alias('foreign_sentiment'),
            pl.lit(None).cast(pl.Float32).alias('retail_institutional_divergence'),
            pl.lit(None).cast(pl.Float32).alias('foreign_domestic_divergence'),
            pl.lit(None).cast(pl.Float32).alias('institutional_persistence'),
            pl.lit(None).cast(pl.Float32).alias('foreign_persistence'),
            pl.lit(None).cast(pl.Float32).alias('smart_flow_indicator'),
            pl.lit(None).cast(pl.Float32).alias('flow_concentration'),
            pl.lit(None).cast(pl.Float32).alias('concentrated_flow_signal'),
        ])


def add_enhanced_flow_features(df_base: pl.DataFrame,
                              fetcher: Any,
                              start_date: str,
                              end_date: str) -> pl.DataFrame:
    """Convenience function to add enhanced flow features.
    
    Args:
        df_base: Base dataset with [code, date] columns
        fetcher: JQuantsAsyncFetcher instance
        start_date: Start date for flow data
        end_date: End date for flow data
    
    Returns:
        DataFrame with enhanced flow features added
    """
    import asyncio
    import aiohttp
    
    async def fetch_flow_data():
        async with aiohttp.ClientSession() as session:
            # Authenticate first
            await fetcher.authenticate(session)
            
            # Fetch trades_spec (weekly investor flow) data
            df_flows = await fetcher.get_trades_spec(
                session,
                from_date=start_date,
                to_date=end_date
            )
            
            return df_flows
    
    # Run async fetch
    try:
        loop = asyncio.get_running_loop()
        try:
            import nest_asyncio  # type: ignore
            nest_asyncio.apply()
        except Exception:
            pass
        df_flows = asyncio.run(fetch_flow_data())
    except RuntimeError:
        df_flows = asyncio.run(fetch_flow_data())
    
    # Extract features
    analyzer = EnhancedFlowAnalyzer()
    
    return analyzer.extract_features(df_base, df_flows)
