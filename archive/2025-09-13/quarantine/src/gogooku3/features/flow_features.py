"""
Investment Flow Features Module
投資主体別売買動向特徴量
"""

import polars as pl
import numpy as np
from typing import Optional, List, Dict
from datetime import datetime, timedelta


class FlowFeatures:
    """投資フロー特徴量計算（仕様準拠）"""
    
    @staticmethod
    def generate_flow_features(
        df: pl.DataFrame,
        weekly_flow_df: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """
        投資フロー特徴量を生成（仕様準拠）
        
        Features:
        - flow_foreign_net_ratio: 外国人投資家ネット買い比率
        - flow_individual_net_ratio: 個人投資家ネット買い比率
        - flow_institution_net_ratio: 機関投資家ネット買い比率
        - flow_proprietary_net_ratio: 自己売買ネット比率
        - flow_trust_net_ratio: 投資信託ネット買い比率
        - flow_concentration_index: 投資主体集中度
        - flow_momentum_*: 各投資主体のモメンタム
        
        Args:
            df: 株価データ
            weekly_flow_df: 週次投資主体別売買データ（オプション）
        
        Returns:
            投資フロー特徴量を追加したDataFrame
        """
        
        if weekly_flow_df is not None:
            # 実データがある場合の処理
            df = FlowFeatures._merge_weekly_flow_data(df, weekly_flow_df)
        else:
            # シミュレーションデータ生成（テスト用）
            df = FlowFeatures._generate_simulated_flow_data(df)
        
        # フロー派生特徴量の計算
        df = FlowFeatures._calculate_flow_derivatives(df)
        
        # 有効性フラグの追加
        df = FlowFeatures._add_flow_validity_flags(df)
        
        return df
    
    @staticmethod
    def _merge_weekly_flow_data(
        df: pl.DataFrame,
        weekly_flow_df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        週次投資主体別売買データを日次データにマージ
        （週次データを日次にキャリーフォワード）
        
        Args:
            df: 日次株価データ
            weekly_flow_df: 週次投資主体別売買データ
        
        Returns:
            マージ済みDataFrame
        """
        
        # Ensure weekly data has required columns
        required_cols = [
            "Code", "WeekEndDate",
            "foreign_buy", "foreign_sell",
            "individual_buy", "individual_sell",
            "institution_buy", "institution_sell",
            "proprietary_buy", "proprietary_sell",
            "trust_buy", "trust_sell"
        ]
        
        missing_cols = [col for col in required_cols if col not in weekly_flow_df.columns]
        if missing_cols:
            # Create dummy columns if missing
            for col in missing_cols:
                if col not in ["Code", "WeekEndDate"]:
                    weekly_flow_df = weekly_flow_df.with_columns(
                        pl.lit(0.0).alias(col)
                    )
        
        # Calculate net ratios for each investor type
        investor_types = ["foreign", "individual", "institution", "proprietary", "trust"]
        
        for investor in investor_types:
            buy_col = f"{investor}_buy"
            sell_col = f"{investor}_sell"
            
            if buy_col in weekly_flow_df.columns and sell_col in weekly_flow_df.columns:
                weekly_flow_df = weekly_flow_df.with_columns(
                    ((pl.col(buy_col) - pl.col(sell_col)) / 
                     (pl.col(buy_col) + pl.col(sell_col) + 1e-12))
                    .alias(f"flow_{investor}_net_ratio")
                )
        
        # Carry forward weekly data to daily
        # Join on the nearest previous week end date
        df = df.sort(["meta_code", "meta_date"])
        weekly_flow_df = weekly_flow_df.sort(["Code", "WeekEndDate"])
        
        # Use as-of join to carry forward weekly values
        df = df.join_asof(
            weekly_flow_df.select([
                pl.col("Code"),
                pl.col("WeekEndDate").alias("meta_date"),
                *[col for col in weekly_flow_df.columns 
                  if col.startswith("flow_")]
            ]),
            on="meta_date",
            by="Code" if "Code" in weekly_flow_df.columns else None,
            strategy="backward"  # Use most recent weekly data
        )
        
        return df
    
    @staticmethod
    def _generate_simulated_flow_data(df: pl.DataFrame) -> pl.DataFrame:
        """
        シミュレーション用投資フロー データ生成（テスト用）
        
        Args:
            df: 株価データ
        
        Returns:
            シミュレーションフローデータを追加したDataFrame
        """
        
        # Generate correlated flow data based on returns
        ret_col = "returns_1d" if "returns_1d" in df.columns else "px_returns_1d"
        
        if ret_col in df.columns:
            # Foreign investors (trend followers)
            df = df.with_columns(
                (pl.col(ret_col).rolling_mean(5, min_periods=1) * 10 + 
                 np.random.randn() * 0.1).clip(-1, 1)
                .alias("flow_foreign_net_ratio")
            )
            
            # Individual investors (contrarians)
            df = df.with_columns(
                (-pl.col(ret_col).rolling_mean(5, min_periods=1) * 8 + 
                 np.random.randn() * 0.1).clip(-1, 1)
                .alias("flow_individual_net_ratio")
            )
            
            # Institutions (value investors)
            df = df.with_columns(
                (pl.col(ret_col).rolling_mean(20, min_periods=5) * 5 + 
                 np.random.randn() * 0.1).clip(-1, 1)
                .alias("flow_institution_net_ratio")
            )
            
            # Proprietary trading (market makers)
            df = df.with_columns(
                pl.lit(np.clip(np.random.randn() * 0.2, -1, 1))
                .alias("flow_proprietary_net_ratio")
            )
            
            # Trust banks (steady accumulation)
            df = df.with_columns(
                pl.lit(np.clip(0.1 + np.random.randn() * 0.05, -1, 1))
                .alias("flow_trust_net_ratio")
            )
        else:
            # Random flow data if no returns available
            for investor in ["foreign", "individual", "institution", "proprietary", "trust"]:
                df = df.with_columns(
                    pl.lit(0.0).alias(f"flow_{investor}_net_ratio")
                )
        
        return df
    
    @staticmethod
    def _calculate_flow_derivatives(df: pl.DataFrame) -> pl.DataFrame:
        """
        フロー派生特徴量の計算
        
        Features:
        - flow_concentration_index: 投資主体集中度
        - flow_momentum_*: 各投資主体のモメンタム
        - flow_divergence_*: 投資主体間の乖離
        - flow_regime_*: フローレジーム
        """
        
        investor_types = ["foreign", "individual", "institution", "proprietary", "trust"]
        
        # Calculate concentration index (Herfindahl index)
        flow_cols = [f"flow_{inv}_net_ratio" for inv in investor_types]
        existing_flow_cols = [col for col in flow_cols if col in df.columns]
        
        if existing_flow_cols:
            # Sum of squared market shares
            concentration_expr = pl.lit(0.0)
            for col in existing_flow_cols:
                concentration_expr = concentration_expr + pl.col(col).abs().pow(2)
            
            df = df.with_columns(
                concentration_expr.sqrt().alias("flow_concentration_index")
            )
            
            # Calculate momentum for each investor type
            for investor in investor_types:
                flow_col = f"flow_{investor}_net_ratio"
                if flow_col in df.columns:
                    # 5-day momentum
                    df = df.with_columns(
                        pl.col(flow_col).diff(5)
                        .alias(f"flow_momentum_{investor}_5d")
                    )
                    
                    # 20-day momentum
                    df = df.with_columns(
                        pl.col(flow_col).diff(20)
                        .alias(f"flow_momentum_{investor}_20d")
                    )
            
            # Calculate divergence between investor types
            if "flow_foreign_net_ratio" in df.columns and "flow_individual_net_ratio" in df.columns:
                df = df.with_columns(
                    (pl.col("flow_foreign_net_ratio") - pl.col("flow_individual_net_ratio"))
                    .alias("flow_divergence_foreign_individual")
                )
            
            if "flow_institution_net_ratio" in df.columns and "flow_individual_net_ratio" in df.columns:
                df = df.with_columns(
                    (pl.col("flow_institution_net_ratio") - pl.col("flow_individual_net_ratio"))
                    .alias("flow_divergence_institution_individual")
                )
            
            # Flow regime indicators
            for investor in ["foreign", "individual", "institution"]:
                flow_col = f"flow_{investor}_net_ratio"
                if flow_col in df.columns:
                    # Buying regime (positive flow for 5 days)
                    df = df.with_columns(
                        (pl.col(flow_col).rolling_mean(5, min_periods=3) > 0)
                        .cast(pl.Int8)
                        .alias(f"flow_regime_{investor}_buying")
                    )
                    
                    # Strong flow (above 75th percentile)
                    df = df.with_columns(
                        (pl.col(flow_col) > pl.col(flow_col).quantile(0.75))
                        .cast(pl.Int8)
                        .alias(f"flow_regime_{investor}_strong")
                    )
        
        return df
    
    @staticmethod
    def _add_flow_validity_flags(df: pl.DataFrame) -> pl.DataFrame:
        """
        フローデータの有効性フラグを追加
        
        Args:
            df: フローデータを含むDataFrame
        
        Returns:
            有効性フラグを追加したDataFrame
        """
        
        # Main flow features validity
        main_flow_cols = [
            "flow_foreign_net_ratio",
            "flow_individual_net_ratio", 
            "flow_institution_net_ratio"
        ]
        
        for col in main_flow_cols:
            if col in df.columns:
                df = df.with_columns(
                    pl.col(col).is_not_null().alias(f"is_{col}_valid")
                )
        
        # Overall flow data validity
        if any(col in df.columns for col in main_flow_cols):
            validity_expr = pl.lit(True)
            for col in main_flow_cols:
                if col in df.columns:
                    validity_expr = validity_expr & pl.col(col).is_not_null()
            
            df = df.with_columns(
                validity_expr.alias("is_flow_data_valid")
            )
        
        return df
    
    @staticmethod
    def calculate_flow_impact_features(
        df: pl.DataFrame,
        price_col: str = "Close"
    ) -> pl.DataFrame:
        """
        フローの価格インパクト特徴量を計算
        
        Features:
        - flow_price_impact_*: 各投資主体のフローと価格変動の関係
        - flow_lead_lag_*: フローと価格のリード・ラグ関係
        """
        
        ret_col = "returns_1d" if "returns_1d" in df.columns else "px_returns_1d"
        
        if ret_col not in df.columns:
            return df
        
        investor_types = ["foreign", "individual", "institution"]
        
        for investor in investor_types:
            flow_col = f"flow_{investor}_net_ratio"
            
            if flow_col in df.columns:
                # Concurrent impact (same day)
                df = df.with_columns(
                    (pl.col(flow_col) * pl.col(ret_col))
                    .alias(f"flow_price_impact_{investor}_0d")
                )
                
                # Lead impact (flow leads price by 1 day)
                df = df.with_columns(
                    (pl.col(flow_col) * pl.col(ret_col).shift(-1))
                    .alias(f"flow_price_impact_{investor}_lead1d")
                )
                
                # Lag impact (flow lags price by 1 day)
                df = df.with_columns(
                    (pl.col(flow_col) * pl.col(ret_col).shift(1))
                    .alias(f"flow_price_impact_{investor}_lag1d")
                )
                
                # Rolling correlation between flow and returns
                # (Simplified version - actual implementation would use proper rolling correlation)
                df = df.with_columns(
                    (pl.col(flow_col) * pl.col(ret_col))
                    .rolling_mean(20, min_periods=10)
                    .alias(f"flow_return_corr_{investor}_20d")
                )
        
        return df
    
    @staticmethod
    def add_flow_seasonality_features(df: pl.DataFrame) -> pl.DataFrame:
        """
        フローの季節性特徴量を追加
        
        Features:
        - flow_month_end_effect: 月末のフロー効果
        - flow_quarter_end_effect: 四半期末のフロー効果
        - flow_week_pattern: 曜日別フローパターン
        """
        
        # Check for calendar features
        if "cal_is_month_end" in df.columns:
            # Month-end flow effects
            for investor in ["foreign", "individual", "institution"]:
                flow_col = f"flow_{investor}_net_ratio"
                if flow_col in df.columns:
                    df = df.with_columns(
                        (pl.col(flow_col) * pl.col("cal_is_month_end"))
                        .alias(f"flow_month_end_{investor}")
                    )
        
        if "cal_is_quarter_end" in df.columns:
            # Quarter-end flow effects
            for investor in ["foreign", "institution"]:
                flow_col = f"flow_{investor}_net_ratio"
                if flow_col in df.columns:
                    df = df.with_columns(
                        (pl.col(flow_col) * pl.col("cal_is_quarter_end"))
                        .alias(f"flow_quarter_end_{investor}")
                    )
        
        return df