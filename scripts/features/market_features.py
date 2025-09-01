#!/usr/bin/env python3
"""
Market Features Module - TOPIX-based market regime features
市場レジーム特徴量の実装（短期1-3日予測向け最適化）
"""

import polars as pl
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def build_topix_features(topix: pl.DataFrame) -> pl.DataFrame:
    """
    TOPIXから市場レジーム特徴量を生成
    
    Args:
        topix: TOPIXデータ (Date, Open, High, Low, Close)
    
    Returns:
        市場特徴量を含むDataFrame (全銘柄に同一値を付与)
    """
    df = topix.sort("Date")
    
    # ========== リターン・トレンド ==========
    df = df.with_columns([
        pl.col("Close").pct_change().alias("mkt_ret_1d"),
        pl.col("Close").pct_change(n=5).alias("mkt_ret_5d"),
        pl.col("Close").pct_change(n=10).alias("mkt_ret_10d"),
        pl.col("Close").pct_change(n=20).alias("mkt_ret_20d"),
        pl.col("Close").ewm_mean(span=5, adjust=False).alias("mkt_ema_5"),
        pl.col("Close").ewm_mean(span=20, adjust=False).alias("mkt_ema_20"),
        pl.col("Close").ewm_mean(span=60, adjust=False).alias("mkt_ema_60"),
        pl.col("Close").ewm_mean(span=200, adjust=False).alias("mkt_ema_200"),
    ])
    
    df = df.with_columns([
        ((pl.col("Close") - pl.col("mkt_ema_20")) / pl.col("mkt_ema_20")).alias("mkt_dev_20"),
        ((pl.col("mkt_ema_5") - pl.col("mkt_ema_20")) / pl.col("mkt_ema_20")).alias("mkt_gap_5_20"),
        pl.col("mkt_ema_20").pct_change(n=3).alias("mkt_ema20_slope_3"),
    ])
    
    # ========== ボラティリティ・レンジ ==========
    df = df.with_columns([
        (pl.col("mkt_ret_1d").rolling_std(20) * np.sqrt(252)).alias("mkt_vol_20d"),
    ])
    
    # ATR/NATR
    df = df.with_columns([
        # True Range
        pl.max_horizontal(
            pl.col("High") - pl.col("Low"),
            (pl.col("High") - pl.col("Close").shift(1)).abs(),
            (pl.col("Low") - pl.col("Close").shift(1)).abs()
        ).alias("TR")
    ]).with_columns([
        pl.col("TR").rolling_mean(window_size=14).alias("mkt_atr_14")
    ]).with_columns([
        (pl.col("mkt_atr_14") / pl.col("Close")).alias("mkt_natr_14")
    ]).drop("TR")
    
    # Bollinger Bands (20, 2σ)
    df = df.with_columns([
        pl.col("Close").rolling_mean(20).alias("bb_mid"),
        pl.col("Close").rolling_std(20).alias("bb_std")
    ]).with_columns([
        (pl.col("bb_mid") + 2 * pl.col("bb_std")).alias("bb_up"),
        (pl.col("bb_mid") - 2 * pl.col("bb_std")).alias("bb_dn"),
    ]).with_columns([
        ((pl.col("Close") - pl.col("bb_dn")) / (pl.col("bb_up") - pl.col("bb_dn") + 1e-12)).clip(0, 1).alias("mkt_bb_pct_b"),
        ((pl.col("bb_up") - pl.col("bb_dn")) / (pl.col("bb_mid") + 1e-12)).alias("mkt_bb_bw")
    ]).drop(["bb_mid", "bb_std", "bb_up", "bb_dn"])
    
    # ========== ドローダウン・インパルス ==========
    df = df.with_columns([
        pl.col("Close").cum_max().alias("cum_peak")
    ]).with_columns([
        ((pl.col("Close") - pl.col("cum_peak")) / pl.col("cum_peak")).alias("mkt_dd_from_peak")
    ]).drop("cum_peak")
    
    # Big move & Squeeze flags
    df = df.with_columns([
        pl.col("mkt_ret_1d").rolling_std(60).alias("ret_std_60")
    ]).with_columns([
        (pl.col("mkt_ret_1d").abs() >= 2.0 * pl.col("ret_std_60")).cast(pl.Int8).alias("mkt_big_move_flag"),
    ]).drop("ret_std_60")
    
    # ========== 時系列Z-score (252日) ==========
    def z_score(col_name: str, window: int = 252) -> pl.Expr:
        """時系列Z-scoreを計算"""
        mu = pl.col(col_name).rolling_mean(window)
        sd = pl.col(col_name).rolling_std(window) + 1e-12
        return ((pl.col(col_name) - mu) / sd).alias(f"{col_name}_z")
    
    df = df.with_columns([
        z_score("mkt_ret_1d"),
        z_score("mkt_vol_20d"),
        z_score("mkt_bb_bw"),
        z_score("mkt_dd_from_peak")
    ])
    
    # ========== レジームフラグ ==========
    df = df.with_columns([
        (pl.col("Close") > pl.col("mkt_ema_200")).cast(pl.Int8).alias("mkt_bull_200"),
        (pl.col("mkt_gap_5_20") > 0).cast(pl.Int8).alias("mkt_trend_up"),
        (pl.col("mkt_vol_20d_z") > 1.0).cast(pl.Int8).alias("mkt_high_vol"),
        (pl.col("mkt_bb_bw_z") < -1.0).cast(pl.Int8).alias("mkt_squeeze"),
    ])
    
    # 必要な列のみ選択（Open, High, Low, Closeは除外）
    keep_cols = [c for c in df.columns if c not in ["Open", "High", "Low", "Close"]]
    
    logger.info(f"✅ Generated {len(keep_cols) - 1} market features from TOPIX")
    
    return df.select(["Date"] + [c for c in keep_cols if c != "Date"])


def attach_market_and_cross_features(
    stock_df: pl.DataFrame, 
    market_df: pl.DataFrame
) -> pl.DataFrame:
    """
    銘柄データに市場特徴量を結合し、クロス特徴量を生成
    
    Args:
        stock_df: 銘柄データ (Code, Date, returns_1d, returns_5d, etc.)
        market_df: 市場特徴量 (Date, mkt_ret_1d, mkt_ret_5d, etc.)
    
    Returns:
        市場特徴量とクロス特徴量を含むDataFrame
    """
    # 市場特徴量を結合
    df = stock_df.join(market_df, on="Date", how="left")
    
    # ========== β計算 (60日ローリング) ==========
    # Cov(X,Y) = E[XY] - E[X]E[Y]
    # Var(Y) = E[Y^2] - E[Y]^2
    
    # 必要な列が存在するかチェック
    if "returns_1d" in df.columns and "mkt_ret_1d" in df.columns:
        df = df.with_columns([
            # 各銘柄ごとに計算
            (pl.col("returns_1d") * pl.col("mkt_ret_1d")).alias("xy_prod"),
        ])
        
        # 60日ローリング統計
        df = df.with_columns([
            # E[X], E[Y], E[XY], E[X^2], E[Y^2]
            pl.col("returns_1d").rolling_mean(60).over("Code").alias("x_mean"),
            pl.col("mkt_ret_1d").rolling_mean(60).over("Code").alias("y_mean"),
            pl.col("xy_prod").rolling_mean(60).over("Code").alias("xy_mean"),
            (pl.col("returns_1d") ** 2).rolling_mean(60).over("Code").alias("x2_mean"),
            (pl.col("mkt_ret_1d") ** 2).rolling_mean(60).over("Code").alias("y2_mean"),
        ])
        
        # Cov, Var計算
        df = df.with_columns([
            (pl.col("xy_mean") - pl.col("x_mean") * pl.col("y_mean")).alias("cov_xy"),
            (pl.col("y2_mean") - pl.col("y_mean") ** 2).alias("var_y")
        ])
        
        # β = Cov(X,Y) / Var(Y)
        df = df.with_columns([
            (pl.col("cov_xy") / (pl.col("var_y") + 1e-12)).alias("beta_60d")
        ])
        
        # 中間変数を削除
        df = df.drop(["xy_prod", "x_mean", "y_mean", "xy_mean", "x2_mean", "y2_mean", "cov_xy", "var_y"])
        
        # ========== 残差・相対強さ・整合性 ==========
        df = df.with_columns([
            # α (残差リターン)
            (pl.col("returns_1d") - pl.col("beta_60d") * pl.col("mkt_ret_1d")).alias("alpha_1d"),
        ])
        
        # 5日リターンが存在する場合
        if "returns_5d" in df.columns and "mkt_ret_5d" in df.columns:
            df = df.with_columns([
                (pl.col("returns_5d") - pl.col("beta_60d") * pl.col("mkt_ret_5d")).alias("alpha_5d"),
                (pl.col("returns_5d") - pl.col("mkt_ret_5d")).alias("rel_strength_5d"),
            ])
        
        # トレンド整合性
        if "ema_gap_5_20" in df.columns and "mkt_gap_5_20" in df.columns:
            df = df.with_columns([
                (pl.col("ema_gap_5_20") * pl.col("mkt_gap_5_20")).alias("trend_align_mkt"),
                (pl.col("alpha_1d") * pl.col("mkt_gap_5_20").sign()).alias("alpha_vs_regime"),
            ])
        
        # アイディオシンクラティック・ボラティリティ比
        if "volatility_20d" in df.columns and "mkt_vol_20d" in df.columns:
            df = df.with_columns([
                (pl.col("volatility_20d") / 
                 (pl.col("beta_60d").abs() * pl.col("mkt_vol_20d") + 1e-6)).alias("idio_vol_ratio"),
            ])
        
        # β安定性（過去60日のβの標準偏差）
        df = df.with_columns([
            pl.col("beta_60d").rolling_std(60).over("Code").alias("beta_stability_60d")
        ])
        
        logger.info("✅ Generated cross features (beta, alpha, relative strength)")
    else:
        logger.warning("⚠️ Required columns for cross features not found")
    
    return df


def add_market_features_to_dataset(
    stock_df: pl.DataFrame, 
    topix_df: Optional[pl.DataFrame] = None
) -> pl.DataFrame:
    """
    メインのデータセットに市場特徴量を追加する統合関数
    
    Args:
        stock_df: 銘柄データ
        topix_df: TOPIXデータ（オプション）
    
    Returns:
        市場特徴量を含む拡張データセット
    """
    if topix_df is None or topix_df.is_empty():
        logger.warning("No TOPIX data provided, skipping market features")
        return stock_df
    
    logger.info("Adding market features to dataset...")
    
    # 1. TOPIXから市場特徴量を生成
    market_features = build_topix_features(topix_df)
    
    # 2. 銘柄データに市場特徴量を結合し、クロス特徴量を生成
    enhanced_df = attach_market_and_cross_features(stock_df, market_features)
    
    # 特徴量数をログ出力
    new_features = len(enhanced_df.columns) - len(stock_df.columns)
    logger.info(f"✅ Added {new_features} market-related features")
    logger.info(f"  - Market features: {len([c for c in enhanced_df.columns if c.startswith('mkt_')])}")
    logger.info(f"  - Cross features: {len([c for c in enhanced_df.columns if c in ['beta_60d', 'alpha_1d', 'alpha_5d', 'rel_strength_5d', 'trend_align_mkt', 'alpha_vs_regime', 'idio_vol_ratio', 'beta_stability_60d']])}")
    
    return enhanced_df