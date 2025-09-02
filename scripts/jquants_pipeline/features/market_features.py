"""
TOPIX Market Features Generator

このモジュールは、短期1-3日予測を強化するためのTOPIX市場レジーム特徴量を生成します。
26個の市場特徴 + 8個のクロス特徴（計34個）を追加します。

特徴量カテゴリ:
- 市場特徴（mkt_ prefix、26個）: リターン、トレンド、ボラティリティ、レジームフラグ
- クロス特徴（8個）: β、α、相対強さ、整合性指標

Author: gogooku3 team
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class MarketFeaturesGenerator:
    """
    TOPIX単体特徴量生成器

    26個の市場レベル特徴量を生成:
    - リターン系: mkt_ret_1d, mkt_ret_5d, mkt_ret_10d, mkt_ret_20d
    - トレンド系: mkt_ema_5, mkt_ema_20, mkt_ema_60, mkt_ema_200
    - 偏差系: mkt_dev_20, mkt_gap_5_20, mkt_ema20_slope_3
    - ボラ系: mkt_vol_20d, mkt_atr_14, mkt_natr_14, mkt_bb_pct_b, mkt_bb_bw
    - リスク系: mkt_dd_from_peak, mkt_big_move_flag
    - Z正規化: mkt_ret_1d_z, mkt_vol_20d_z, mkt_bb_bw_z, mkt_dd_from_peak_z
    - レジーム: mkt_bull_200, mkt_trend_up, mkt_high_vol, mkt_squeeze
    """

    def __init__(self, z_score_window: int = 252):
        """
        Args:
            z_score_window: Zスコア計算の窓サイズ（デフォルト252営業日）
        """
        self.z_score_window = z_score_window

    def build_topix_features(self, topix_df: pl.DataFrame) -> pl.DataFrame:
        """
        TOPIXデータから市場特徴量を生成

        Args:
            topix_df: TOPIX価格データ（Date, Open, High, Low, Close列必須）

        Returns:
            市場特徴量を追加したDataFrame
        """
        if topix_df is None or topix_df.is_empty():
            logger.warning("No TOPIX data provided")
            return pl.DataFrame()
            
        df = topix_df.sort("Date")
        
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
        
        # ATR/NATR (Open, High, Low必須)
        if all(col in df.columns for col in ["Open", "High", "Low"]):
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
        else:
            # Open/High/Lowがない場合はCloseのみから簡易版
            df = df.with_columns([
                pl.col("Close").pct_change().abs().rolling_mean(14).alias("mkt_atr_14"),
                pl.col("Close").pct_change().abs().rolling_mean(14).alias("mkt_natr_14"),
            ])
        
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
        
        # Big move flags
        df = df.with_columns([
            pl.col("mkt_ret_1d").rolling_std(60).alias("ret_std_60")
        ]).with_columns([
            (pl.col("mkt_ret_1d").abs() >= 2.0 * pl.col("ret_std_60")).cast(pl.Int8).alias("mkt_big_move_flag"),
        ]).drop("ret_std_60")
        
        # ========== 時系列Z-score (252日、min_periods=252) ==========
        def z_score(col_name: str, window: int = 252, min_periods: int = 252) -> pl.Expr:
            """時系列Z-scoreを計算"""
            mu = pl.col(col_name).rolling_mean(window, min_periods=min_periods)
            sd = pl.col(col_name).rolling_std(window, min_periods=min_periods) + 1e-12
            return ((pl.col(col_name) - mu) / sd).alias(f"{col_name}_z")
        
        df = df.with_columns([
            z_score("mkt_ret_1d", self.z_score_window, 252),
            z_score("mkt_vol_20d", self.z_score_window, 252),
            z_score("mkt_bb_bw", self.z_score_window, 252),
            z_score("mkt_dd_from_peak", self.z_score_window, 252)
        ])
        
        # ========== レジームフラグ ==========
        df = df.with_columns([
            (pl.col("Close") > pl.col("mkt_ema_200")).cast(pl.Int8).alias("mkt_bull_200"),
            (pl.col("mkt_gap_5_20") > 0).cast(pl.Int8).alias("mkt_trend_up"),
            (pl.col("mkt_vol_20d_z") > 1.0).cast(pl.Int8).alias("mkt_high_vol"),
            (pl.col("mkt_bb_bw_z") < -1.0).cast(pl.Int8).alias("mkt_squeeze"),
        ])
        
        # 必要な列のみ選択（Open, High, Low, Closeは除外）
        keep_cols = [c for c in df.columns if c not in ["Open", "High", "Low", "Close", "Volume"]]
        
        logger.info(f"✅ Generated {len(keep_cols) - 1} market features from TOPIX")
        
        return df.select([pl.col("Date")] + [pl.col(c) for c in keep_cols if c != "Date"])


class CrossMarketFeaturesGenerator:
    """
    銘柄×市場のクロス特徴量生成器
    
    8個のクロス特徴量を生成:
    - beta_60d: 60日ベータ
    - alpha_1d: 1日アルファ（残差リターン）
    - alpha_5d: 5日アルファ
    - rel_strength_5d: 5日相対強度
    - trend_align_mkt: トレンド整合性
    - alpha_vs_regime: アルファ×レジーム
    - idio_vol_ratio: アイディオシンクラティック・ボラ比
    - beta_stability_60d: ベータ安定性
    """
    
    def __init__(self):
        pass
    
    def attach_market_and_cross(
        self, 
        stock_df: pl.DataFrame,
        market_df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        銘柄データに市場特徴量を結合し、クロス特徴量を生成
        
        Args:
            stock_df: 銘柄データ (Code, Date, returns_1d, returns_5d等が必要)
            market_df: 市場特徴量 (Date, mkt_ret_1d, mkt_ret_5d等)
        
        Returns:
            市場特徴量とクロス特徴量を含むDataFrame
        """
        # Date列の型を統一（両方をDatetime[ms]に変換）
        if "Date" in stock_df.columns:
            stock_df = stock_df.with_columns(
                pl.col("Date").cast(pl.Datetime("ms"))
            )
        
        if "Date" in market_df.columns:
            market_df = market_df.with_columns(
                pl.col("Date").cast(pl.Datetime("ms"))
            )
        
        # 市場特徴量を結合
        df = stock_df.join(market_df, on="Date", how="left")
        
        # P1: β計算 with t-1 lag (60日ローリング)
        # 必要な列が存在するかチェック
        if "returns_1d" in df.columns and "mkt_ret_1d" in df.columns:
            # P1: 市場リターンをt-1にラグ（1日前の市場と今日の個別銘柄）
            df = df.with_columns([
                pl.col("mkt_ret_1d").shift(1).over("Code").alias("mkt_ret_lag1")
            ])
            
            df = df.with_columns([
                # 各銘柄ごとに計算（t-1ラグ付き）
                (pl.col("returns_1d") * pl.col("mkt_ret_lag1")).alias("xy_prod"),
            ])
            
            # 60日ローリング統計（min_periods適用）
            df = df.with_columns([
                # E[X], E[Y], E[XY], E[X^2], E[Y^2]
                pl.col("returns_1d").rolling_mean(60, min_periods=60).over("Code").alias("x_mean"),
                pl.col("mkt_ret_lag1").rolling_mean(60, min_periods=60).over("Code").alias("y_mean"),
                pl.col("xy_prod").rolling_mean(60, min_periods=60).over("Code").alias("xy_mean"),
                (pl.col("returns_1d") ** 2).rolling_mean(60, min_periods=60).over("Code").alias("x2_mean"),
                (pl.col("mkt_ret_lag1") ** 2).rolling_mean(60, min_periods=60).over("Code").alias("y2_mean"),
            ])
            
            # Cov, Var計算
            df = df.with_columns([
                (pl.col("xy_mean") - pl.col("x_mean") * pl.col("y_mean")).alias("cov_xy"),
                (pl.col("y2_mean") - pl.col("y_mean") ** 2).alias("var_y")
            ])
            
            # β = Cov(X,Y) / Var(Y) with t-1 lag
            df = df.with_columns([
                (pl.col("cov_xy") / (pl.col("var_y") + 1e-12)).alias("beta_60d_raw")
            ])
            # フォールバック(beta_20d)を併用しcoalesce列を提供
            df = df.with_columns([
                pl.col("returns_1d").rolling_mean(20, min_periods=20).over("Code").alias("x_mean20"),
                pl.col("mkt_ret_lag1").rolling_mean(20, min_periods=20).over("Code").alias("y_mean20"),
                (pl.col("returns_1d") * pl.col("mkt_ret_lag1")).rolling_mean(20, min_periods=20).over("Code").alias("xy_mean20"),
                (pl.col("returns_1d")**2).rolling_mean(20, min_periods=20).over("Code").alias("x2_mean20"),
                (pl.col("mkt_ret_lag1")**2).rolling_mean(20, min_periods=20).over("Code").alias("y2_mean20"),
            ]).with_columns([
                (pl.col("xy_mean20") - pl.col("x_mean20") * pl.col("y_mean20")).alias("cov_xy20"),
                (pl.col("y2_mean20") - pl.col("y_mean20")**2).alias("var_y20")
            ]).with_columns([
                (pl.col("cov_xy20") / (pl.col("var_y20") + 1e-12)).alias("beta_20d_raw")
            ]).with_columns([
                pl.coalesce([pl.col("beta_60d_raw"), pl.col("beta_20d_raw")]).alias("beta_rolling")
            ]).drop(["x_mean20","y_mean20","xy_mean20","x2_mean20","y2_mean20","cov_xy20","var_y20"])
            
            # 中間変数を削除
            df = df.drop(["mkt_ret_lag1", "xy_prod", "x_mean", "y_mean", "xy_mean", "x2_mean", "y2_mean", "cov_xy", "var_y"])
            
            # ========== 残差・相対強さ・整合性 ==========
            df = df.with_columns([
                # α (残差リターン)
                (pl.col("returns_1d") - pl.col("beta_rolling") * pl.col("mkt_ret_1d")).alias("alpha_1d"),
            ])
            
            # 5日リターンが存在する場合
            if "returns_5d" in df.columns and "mkt_ret_5d" in df.columns:
                df = df.with_columns([
                    (pl.col("returns_5d") - pl.col("beta_rolling") * pl.col("mkt_ret_5d")).alias("alpha_5d"),
                    (pl.col("returns_5d") - pl.col("mkt_ret_5d")).alias("rel_strength_5d"),
                ])
            
            # トレンド整合性（ema_gap_5_20が銘柄側の特徴）
            if "ma_gap_5_20" in df.columns and "mkt_gap_5_20" in df.columns:
                df = df.with_columns([
                    (pl.col("ma_gap_5_20") * pl.col("mkt_gap_5_20")).alias("trend_align_mkt"),
                    (pl.col("alpha_1d") * pl.col("mkt_gap_5_20").sign()).alias("alpha_vs_regime"),
                ])
            
            # アイディオシンクラティック・ボラティリティ比
            if "volatility_20d" in df.columns and "mkt_vol_20d" in df.columns:
                df = df.with_columns([
                    (pl.col("volatility_20d") / 
                     (pl.col("beta_rolling").abs() * pl.col("mkt_vol_20d") + 1e-6)).alias("idio_vol_ratio"),
                ])
            
            # β安定性（過去60日のβの標準偏差）
            df = df.with_columns([
                pl.col("beta_rolling").rolling_std(60).over("Code").alias("beta_stability_60d")
            ])
            
            logger.info("✅ Generated cross features (beta, alpha, relative strength)")
        else:
            logger.warning("⚠️ Required columns for cross features not found (returns_1d, mkt_ret_1d)")
        
        return df