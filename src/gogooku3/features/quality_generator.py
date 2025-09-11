"""Quality financial features generator for enhanced ML datasets."""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    logger.warning("Polars not available, using pandas fallback")


class QualityFinancialFeaturesGenerator:
    """
    Quality Financial Features Generator for creating high-quality ML features.
    
    Generates technical indicators, fundamental ratios, and market microstructure
    features with quality validation and normalization.
    """
    
    def __init__(
        self,
        min_coverage_frac: float = 0.98,
        outlier_clip_quantile: float = 0.01,
        technical_indicators: bool = True,
        fundamental_features: bool = True,
        market_features: bool = True,
        **kwargs
    ):
        """
        Initialize Quality Financial Features Generator.
        
        Args:
            min_coverage_frac: Minimum data coverage fraction required
            outlier_clip_quantile: Quantile for outlier clipping
            technical_indicators: Whether to generate technical indicators
            fundamental_features: Whether to generate fundamental features
            market_features: Whether to generate market microstructure features
        """
        self.min_coverage_frac = min_coverage_frac
        self.outlier_clip_quantile = outlier_clip_quantile
        self.technical_indicators = technical_indicators
        self.fundamental_features = fundamental_features
        self.market_features = market_features
        self.kwargs = kwargs
        
        self.feature_names = []
        self.feature_metadata = {}
        
        logger.info(f"Initialized QualityFinancialFeaturesGenerator with "
                   f"min_coverage={min_coverage_frac}")
    
    def generate_features(
        self, 
        df: Union[pd.DataFrame, pl.DataFrame],
        price_cols: Optional[List[str]] = None,
        volume_cols: Optional[List[str]] = None,
        fundamental_cols: Optional[List[str]] = None
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Generate quality financial features from input data.
        
        Args:
            df: Input financial data
            price_cols: Price column names
            volume_cols: Volume column names  
            fundamental_cols: Fundamental data column names
            
        Returns:
            DataFrame with generated features
        """
        logger.info("Generating quality financial features...")
        
        try:
            if price_cols is None:
                price_cols = ['open', 'high', 'low', 'close', 'adj_close']
            if volume_cols is None:
                volume_cols = ['volume']
            if fundamental_cols is None:
                fundamental_cols = []
            
            use_polars = POLARS_AVAILABLE and isinstance(df, pl.DataFrame)
            if use_polars:
                work_df = df.to_pandas()
            else:
                work_df = df.copy()
            
            features_df = work_df.copy()
            
            if self.technical_indicators:
                features_df = self._generate_technical_features(features_df, price_cols, volume_cols)
            
            if self.fundamental_features and fundamental_cols:
                features_df = self._generate_fundamental_features(features_df, fundamental_cols)
            
            if self.market_features:
                features_df = self._generate_market_features(features_df, price_cols, volume_cols)
            
            features_df = self._validate_and_clean_features(features_df)
            
            if use_polars:
                features_df = pl.from_pandas(features_df)
            
            logger.info(f"Generated {len(features_df.columns)} quality features")
            return features_df
            
        except Exception as e:
            logger.error(f"Feature generation failed: {e}")
            raise
    
    def _generate_technical_features(
        self, 
        df: pd.DataFrame, 
        price_cols: List[str], 
        volume_cols: List[str]
    ) -> pd.DataFrame:
        """Generate technical indicator features."""
        logger.info("Generating technical indicator features...")
        
        try:
            if 'close' in df.columns:
                df['return_1d'] = df['close'].pct_change()
                df['return_5d'] = df['close'].pct_change(5)
                df['return_20d'] = df['close'].pct_change(20)
                
                df['sma_5'] = df['close'].rolling(5).mean()
                df['sma_20'] = df['close'].rolling(20).mean()
                df['sma_50'] = df['close'].rolling(50).mean()
                
                df['price_to_sma5'] = df['close'] / df['sma_5']
                df['price_to_sma20'] = df['close'] / df['sma_20']
                
                df['volatility_20d'] = df['return_1d'].rolling(20).std()
                
            if all(col in df.columns for col in ['high', 'low', 'close']):
                df['hl_ratio'] = df['high'] / df['low']
                df['close_to_high'] = df['close'] / df['high']
                df['close_to_low'] = df['close'] / df['low']
            
            if 'volume' in df.columns:
                df['volume_sma_20'] = df['volume'].rolling(20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma_20']
                
            self.feature_names.extend([
                'return_1d', 'return_5d', 'return_20d',
                'sma_5', 'sma_20', 'sma_50',
                'price_to_sma5', 'price_to_sma20',
                'volatility_20d', 'hl_ratio', 'close_to_high', 'close_to_low',
                'volume_sma_20', 'volume_ratio'
            ])
            
            return df
            
        except Exception as e:
            logger.error(f"Technical feature generation failed: {e}")
            return df
    
    def _generate_fundamental_features(
        self, 
        df: pd.DataFrame, 
        fundamental_cols: List[str]
    ) -> pd.DataFrame:
        """Generate fundamental analysis features."""
        logger.info("Generating fundamental features...")
        
        try:
            for col in fundamental_cols:
                if col in df.columns:
                    df[f'{col}_growth_1q'] = df[col].pct_change(1)
                    df[f'{col}_growth_4q'] = df[col].pct_change(4)
                    
                    df[f'{col}_ma_4q'] = df[col].rolling(4).mean()
                    
                    df[f'{col}_to_ma4q'] = df[col] / df[f'{col}_ma_4q']
                    
                    self.feature_names.extend([
                        f'{col}_growth_1q', f'{col}_growth_4q',
                        f'{col}_ma_4q', f'{col}_to_ma4q'
                    ])
            
            return df
            
        except Exception as e:
            logger.error(f"Fundamental feature generation failed: {e}")
            return df
    
    def _generate_market_features(
        self, 
        df: pd.DataFrame, 
        price_cols: List[str], 
        volume_cols: List[str]
    ) -> pd.DataFrame:
        """Generate market microstructure features."""
        logger.info("Generating market microstructure features...")
        
        try:
            if 'close' in df.columns:
                df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
                df['momentum_20d'] = df['close'] / df['close'].shift(20) - 1
                
                df['price_accel'] = df['return_1d'] - df['return_1d'].shift(1)
                
            if all(col in df.columns for col in ['close', 'volume']):
                df['vwap_approx'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
                df['price_to_vwap'] = df['close'] / df['vwap_approx']
                
            self.feature_names.extend([
                'momentum_5d', 'momentum_20d', 'price_accel',
                'vwap_approx', 'price_to_vwap'
            ])
            
            return df
            
        except Exception as e:
            logger.error(f"Market feature generation failed: {e}")
            return df
    
    def _validate_and_clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean generated features."""
        logger.info("Validating and cleaning features...")
        
        try:
            for col in df.columns:
                coverage = df[col].notna().mean()
                if coverage < self.min_coverage_frac:
                    logger.warning(f"Dropping {col} due to low coverage: {coverage:.3f}")
                    df = df.drop(columns=[col])
                    if col in self.feature_names:
                        self.feature_names.remove(col)
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col in self.feature_names:
                    lower = df[col].quantile(self.outlier_clip_quantile)
                    upper = df[col].quantile(1 - self.outlier_clip_quantile)
                    df[col] = df[col].clip(lower, upper)
            
            df = df.fillna(method='ffill').fillna(0)
            
            logger.info(f"Feature validation completed. Final feature count: {len(self.feature_names)}")
            return df
            
        except Exception as e:
            logger.error(f"Feature validation failed: {e}")
            return df
    
    def get_feature_metadata(self) -> Dict[str, Any]:
        """Get metadata about generated features."""
        return {
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "technical_indicators": self.technical_indicators,
            "fundamental_features": self.fundamental_features,
            "market_features": self.market_features,
            "min_coverage_frac": self.min_coverage_frac,
            "outlier_clip_quantile": self.outlier_clip_quantile
        }
    
    def save_features(self, df: pd.DataFrame, output_path: Union[str, Path]) -> None:
        """Save generated features to file."""
        output_path = Path(output_path)
        
        try:
            if output_path.suffix == '.parquet':
                df.to_parquet(output_path, index=False)
            elif output_path.suffix == '.csv':
                df.to_csv(output_path, index=False)
            else:
                raise ValueError(f"Unsupported file format: {output_path.suffix}")
            
            logger.info(f"Features saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save features: {e}")
            raise


def create_quality_features_generator(**kwargs) -> QualityFinancialFeaturesGenerator:
    """Factory function to create quality features generator."""
    return QualityFinancialFeaturesGenerator(**kwargs)
