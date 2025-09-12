"""
Cross-sectional features calculation module
個別銘柄と市場・セクターの相互作用特徴量
"""

import polars as pl
import numpy as np
from typing import Optional, List, Dict, Tuple


class CrossFeatures:
    """クロスセクション特徴量計算（仕様準拠）"""
    
    @staticmethod
    def calculate_beta_alpha(
        df: pl.DataFrame,
        window: int = 60,
        lag: int = 1,
        min_periods: Optional[int] = None
    ) -> pl.DataFrame:
        """
        ベータ・アルファ計算（t-1ラグでlook-ahead bias防止）
        
        Args:
            df: 個別銘柄リターンと市場リターンを含むDataFrame
            window: 計算ウィンドウ（デフォルト60日）
            lag: ラグ日数（デフォルト1日、look-ahead bias防止）
            min_periods: 最小必要データ数（デフォルトはwindowの半分）
        
        Returns:
            ベータ・アルファを追加したDataFrame
        """
        if min_periods is None:
            min_periods = window // 2
        
        # Ensure required columns exist
        required_cols = ["returns_1d", "mkt_ret_1d"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Group by code for stock-specific calculations
        result_dfs = []
        
        for code in df["meta_code"].unique():
            stock_df = df.filter(pl.col("meta_code") == code).sort("meta_date")
            
            # Calculate rolling beta (lagged by 1 day to prevent look-ahead)
            # Beta = Cov(stock, market) / Var(market)
            stock_returns = stock_df["returns_1d"].to_numpy()
            market_returns = stock_df["mkt_ret_1d"].to_numpy()
            
            betas = []
            alphas = []
            
            for i in range(len(stock_returns)):
                if i < window - 1:
                    betas.append(None)
                    alphas.append(None)
                else:
                    # Use data up to i-lag (exclusive) to calculate beta for day i
                    end_idx = i - lag + 1  # Subtract lag to prevent look-ahead
                    start_idx = max(0, end_idx - window)
                    
                    if end_idx - start_idx >= min_periods:
                        stock_window = stock_returns[start_idx:end_idx]
                        market_window = market_returns[start_idx:end_idx]
                        
                        # Remove NaN values
                        valid_mask = ~(np.isnan(stock_window) | np.isnan(market_window))
                        if np.sum(valid_mask) >= min_periods:
                            stock_clean = stock_window[valid_mask]
                            market_clean = market_window[valid_mask]
                            
                            # Calculate beta
                            cov = np.cov(stock_clean, market_clean)[0, 1]
                            var_market = np.var(market_clean)
                            
                            if var_market > 1e-12:
                                beta = cov / var_market
                                # Calculate alpha (Jensen's alpha)
                                alpha = np.mean(stock_clean) - beta * np.mean(market_clean)
                            else:
                                beta = 1.0  # Default beta
                                alpha = 0.0
                            
                            betas.append(beta)
                            alphas.append(alpha)
                        else:
                            betas.append(None)
                            alphas.append(None)
                    else:
                        betas.append(None)
                        alphas.append(None)
            
            # Add results to dataframe
            stock_df = stock_df.with_columns([
                pl.Series(f"x_beta_{window}d", betas),
                pl.Series(f"x_alpha_rolling_{window}d", alphas)
            ])
            
            result_dfs.append(stock_df)
        
        return pl.concat(result_dfs)
    
    @staticmethod
    def add_cross_features(
        df: pl.DataFrame,
        beta_windows: List[int] = [60],
        alpha_horizons: List[int] = [1, 5, 10, 20]
    ) -> pl.DataFrame:
        """
        全クロス特徴量を追加（仕様準拠）
        
        Features added:
        - x_beta_60d: 60日ベータ（t-1ラグ）
        - x_alpha_1d, 5d, 10d, 20d: 各期間アルファ（超過リターン）
        - x_rel_strength_*d: 相対強度
        - x_trend_align_mkt: 市場トレンドアライメント
        - x_idio_vol_ratio: 固有ボラティリティ比率
        - x_beta_stability_60d: ベータ安定性
        """
        
        # Calculate beta with t-1 lag (prevent look-ahead bias)
        for window in beta_windows:
            df = CrossFeatures.calculate_beta_alpha(df, window=window, lag=1)
        
        # Calculate alpha for different horizons (simple excess returns)
        for horizon in alpha_horizons:
            ret_col = f"returns_{horizon}d" if f"returns_{horizon}d" in df.columns else f"px_returns_{horizon}d"
            mkt_col = f"mkt_ret_{horizon}d"
            
            if ret_col in df.columns and mkt_col in df.columns:
                # Alpha (excess return)
                df = df.with_columns(
                    (pl.col(ret_col) - pl.col(mkt_col)).alias(f"x_alpha_{horizon}d")
                )
                
                # Relative strength
                df = df.with_columns(
                    (pl.col(ret_col) / (pl.col(mkt_col).abs() + 1e-12)).alias(f"x_rel_strength_{horizon}d")
                )
        
        # Market trend alignment (correlation with market over rolling window)
        if "returns_1d" in df.columns or "px_returns_1d" in df.columns:
            ret_col = "returns_1d" if "returns_1d" in df.columns else "px_returns_1d"
            
            # Simple rolling correlation proxy
            df = df.with_columns([
                # Sign alignment (1 if same direction, -1 if opposite)
                (pl.col(ret_col).sign() * pl.col("mkt_ret_1d").sign()).alias("x_sign_align"),
                
                # Rolling average of sign alignment
                (pl.col(ret_col).sign() * pl.col("mkt_ret_1d").sign())
                .rolling_mean(20, min_periods=10)
                .alias("x_trend_align_mkt")
            ])
        
        # Idiosyncratic volatility ratio
        if "volatility_20d" in df.columns or "px_volatility_20d" in df.columns:
            vol_col = "volatility_20d" if "volatility_20d" in df.columns else "px_volatility_20d"
            
            if "x_beta_60d" in df.columns and "mkt_vol_20d" in df.columns:
                # Idiosyncratic vol = Total vol - Beta * Market vol
                df = df.with_columns([
                    # Total variance - explained variance
                    (
                        pl.col(vol_col).pow(2) - 
                        (pl.col("x_beta_60d").pow(2) * pl.col("mkt_vol_20d").pow(2))
                    ).sqrt().alias("x_idio_vol"),
                    
                    # Idiosyncratic vol ratio
                    (
                        (pl.col(vol_col).pow(2) - 
                         (pl.col("x_beta_60d").pow(2) * pl.col("mkt_vol_20d").pow(2))).sqrt() /
                        (pl.col(vol_col) + 1e-12)
                    ).alias("x_idio_vol_ratio")
                ])
        
        # Beta stability (rolling std of beta changes)
        if "x_beta_60d" in df.columns:
            df = df.with_columns([
                # Beta change
                pl.col("x_beta_60d").diff().alias("x_beta_change"),
                
                # Beta stability (inverse of beta volatility)
                (1.0 / (pl.col("x_beta_60d").diff().rolling_std(20, min_periods=10) + 1e-12))
                .alias("x_beta_stability_60d")
            ])
        
        # Add validity flags for key features
        validity_cols = ["x_beta_60d", "x_alpha_1d", "x_trend_align_mkt", "x_idio_vol_ratio"]
        for col in validity_cols:
            if col in df.columns:
                df = df.with_columns(
                    pl.col(col).is_not_null().alias(f"is_{col}_valid")
                )
        
        return df
    
    @staticmethod
    def add_sector_relative_features(
        df: pl.DataFrame,
        sector_col: str = "meta_section"
    ) -> pl.DataFrame:
        """
        セクター相対特徴量の追加
        
        Features:
        - x_sector_rel_ret_*d: セクター相対リターン
        - x_sector_rel_vol: セクター相対ボラティリティ
        - x_sector_rank_*: セクター内ランク
        """
        
        # Calculate sector averages
        for horizon in [1, 5, 10, 20]:
            ret_col = f"returns_{horizon}d" if f"returns_{horizon}d" in df.columns else f"px_returns_{horizon}d"
            
            if ret_col in df.columns:
                # Sector mean return
                df = df.with_columns(
                    pl.col(ret_col).mean().over(["meta_date", sector_col])
                    .alias(f"sector_mean_ret_{horizon}d")
                )
                
                # Relative to sector
                df = df.with_columns(
                    (pl.col(ret_col) - pl.col(f"sector_mean_ret_{horizon}d"))
                    .alias(f"x_sector_rel_ret_{horizon}d")
                )
                
                # Rank within sector
                df = df.with_columns(
                    pl.col(ret_col).rank(method="average").over(["meta_date", sector_col])
                    .alias(f"x_sector_rank_ret_{horizon}d")
                )
        
        # Sector relative volatility
        vol_col = "volatility_20d" if "volatility_20d" in df.columns else "px_volatility_20d"
        if vol_col in df.columns:
            df = df.with_columns(
                pl.col(vol_col).mean().over(["meta_date", sector_col])
                .alias("sector_mean_vol")
            )
            df = df.with_columns(
                (pl.col(vol_col) / (pl.col("sector_mean_vol") + 1e-12))
                .alias("x_sector_rel_vol")
            )
        
        return df
    
    @staticmethod
    def add_market_cap_relative_features(
        df: pl.DataFrame,
        mktcap_col: str = "MarketCap",
        n_quantiles: int = 10
    ) -> pl.DataFrame:
        """
        時価総額相対特徴量の追加（仕様準拠: Date × Section × 時価総額デシル）
        
        Features:
        - x_mktcap_decile: 時価総額デシル
        - x_mktcap_rel_ret_*d: 時価総額グループ相対リターン
        - x_mktcap_rank_*: 時価総額グループ内ランク
        """
        
        if mktcap_col not in df.columns:
            # Create dummy market cap if not exists
            df = df.with_columns(
                (pl.col("px_close") * pl.col("Volume")).alias(mktcap_col)
            )
        
        # Calculate market cap deciles per date and section
        df = df.with_columns(
            pl.col(mktcap_col).qcut(n_quantiles, labels=[str(i) for i in range(n_quantiles)])
            .over(["meta_date", "meta_section"])
            .alias("x_mktcap_decile")
        )
        
        # Calculate relative features within market cap groups
        for horizon in [1, 5, 10, 20]:
            ret_col = f"returns_{horizon}d" if f"returns_{horizon}d" in df.columns else f"px_returns_{horizon}d"
            
            if ret_col in df.columns:
                # Mean return per mktcap group
                df = df.with_columns(
                    pl.col(ret_col).mean()
                    .over(["meta_date", "meta_section", "x_mktcap_decile"])
                    .alias(f"mktcap_group_mean_ret_{horizon}d")
                )
                
                # Relative to mktcap group
                df = df.with_columns(
                    (pl.col(ret_col) - pl.col(f"mktcap_group_mean_ret_{horizon}d"))
                    .alias(f"x_mktcap_rel_ret_{horizon}d")
                )
                
                # Rank within mktcap group
                df = df.with_columns(
                    pl.col(ret_col).rank(method="average")
                    .over(["meta_date", "meta_section", "x_mktcap_decile"])
                    .alias(f"x_mktcap_rank_ret_{horizon}d")
                )
        
        return df


class MarketRegimeFeatures:
    """市場レジーム特徴量"""
    
    @staticmethod
    def add_regime_features(
        df: pl.DataFrame,
        lookback_windows: List[int] = [20, 60, 120]
    ) -> pl.DataFrame:
        """
        市場レジーム特徴量の追加
        
        Features:
        - x_regime_bull_*: ブル市場フラグ
        - x_regime_vol_*: ボラティリティレジーム
        - x_regime_trend_strength_*: トレンド強度
        """
        
        for window in lookback_windows:
            # Bull/Bear regime
            if f"mkt_ret_{window}d" in df.columns:
                df = df.with_columns(
                    (pl.col(f"mkt_ret_{window}d") > 0).cast(pl.Int8)
                    .alias(f"x_regime_bull_{window}")
                )
            
            # Volatility regime (high/low)
            if "mkt_vol_20d" in df.columns:
                df = df.with_columns(
                    (pl.col("mkt_vol_20d") > 
                     pl.col("mkt_vol_20d").rolling_median(window, min_periods=window//2))
                    .cast(pl.Int8)
                    .alias(f"x_regime_vol_high_{window}")
                )
            
            # Trend strength (absolute return / volatility)
            if f"mkt_ret_{window}d" in df.columns and "mkt_vol_20d" in df.columns:
                df = df.with_columns(
                    (pl.col(f"mkt_ret_{window}d").abs() / (pl.col("mkt_vol_20d") * np.sqrt(window/20) + 1e-12))
                    .alias(f"x_regime_trend_strength_{window}")
                )
        
        return df