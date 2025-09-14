"""
Feature Validator - 特徴量の有効性管理とmin_periods制御
本番品質のための窓集計バリデーション
"""

import logging

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


class FeatureValidator:
    """
    特徴量の有効性を管理し、min_periodsを一貫して適用
    """

    # 各特徴量に必要な最小期間の定義
    MIN_PERIODS_CONFIG = {
        # ボラティリティ
        "volatility_5d": 5,
        "volatility_10d": 10,
        "volatility_20d": 20,
        "volatility_60d": 60,
        "realized_vol_20": 20,

        # 移動平均
        "sma_5": 5,
        "sma_10": 10,
        "sma_20": 20,
        "sma_60": 60,
        "sma_120": 120,

        # EMA（通常span×3）
        "ema_5": 15,
        "ema_10": 30,
        "ema_20": 60,
        "ema_60": 180,
        "ema_200": 600,

        # 技術指標
        "rsi_14": 14,
        "rsi_2": 2,
        "macd_signal": 26,  # slow period
        "bb_position": 20,
        "atr_14": 14,
        "adx": 14,

        # ベータ・相関
        "beta_60d": 60,
        "beta_stability_60d": 80,  # 60 + 20

        # TOPIX関連
        "mkt_vol_20d": 20,
        "mkt_big_move_flag": 60,
        "mkt_ret_1d_z": 252,
        "mkt_vol_20d_z": 252,

        # フロー（週次）
        "foreign_net_z": 52,
        "individual_net_z": 52,
        "activity_z": 52,
        "smart_money_mom4": 56,  # 52 + 4
    }

    @classmethod
    def add_validity_flags(cls, df: pl.DataFrame) -> pl.DataFrame:
        """
        各特徴量に対応する有効性フラグを追加
        
        Args:
            df: 特徴量を含むDataFrame
            
        Returns:
            有効性フラグを追加したDataFrame
        """
        logger.info("Adding validity flags for all features...")

        # 各銘柄のrow_indexを追加
        df = df.with_columns([
            pl.int_range(0, pl.len()).over("Code").alias("row_idx")
        ])

        # 有効性フラグの生成
        validity_expressions = []

        for feature, min_periods in cls.MIN_PERIODS_CONFIG.items():
            if feature in df.columns:
                # NULL以外かつ最小期間を満たす
                validity_expr = (
                    (pl.col(feature).is_not_null()) &
                    (pl.col("row_idx") >= min_periods - 1)
                ).cast(pl.Int8).alias(f"is_{feature}_valid")

                validity_expressions.append(validity_expr)

        if validity_expressions:
            df = df.with_columns(validity_expressions)

        # 統合有効性フラグ（主要指標がすべて有効）
        core_features = ["volatility_20d", "ema_20", "beta_60d"]
        core_valid_flags = [f"is_{f}_valid" for f in core_features if f in df.columns]

        if core_valid_flags:
            df = df.with_columns([
                pl.min_horizontal(core_valid_flags).alias("is_core_valid")
            ])

        # Add specific flags for DATASET.md compatibility
        df = df.with_columns([
            (pl.col("row_idx") >= 2).cast(pl.Int8).alias("is_rsi2_valid"),
            (pl.col("row_idx") >= 60).cast(pl.Int8).alias("is_valid_ma"),
        ])

        logger.info(f"Added {len(validity_expressions) + 2} validity flags")
        return df

    @classmethod
    def apply_min_periods_to_rolling(
        cls,
        df: pl.DataFrame,
        column: str,
        window: int,
        func: str = "mean",
        min_periods: int | None = None
    ) -> pl.Expr:
        """
        min_periodsを適用したローリング集計
        
        Args:
            df: DataFrame
            column: 対象列名
            window: 窓サイズ
            func: 集計関数（mean, std, sum等）
            min_periods: 最小期間（省略時はwindowと同じ）
            
        Returns:
            min_periods適用済みの集計式
        """
        if min_periods is None:
            min_periods = window

        if func == "mean":
            return pl.col(column).rolling_mean(window, min_periods=min_periods)
        elif func == "std":
            return pl.col(column).rolling_std(window, min_periods=min_periods)
        elif func == "sum":
            return pl.col(column).rolling_sum(window, min_periods=min_periods)
        elif func == "min":
            return pl.col(column).rolling_min(window, min_periods=min_periods)
        elif func == "max":
            return pl.col(column).rolling_max(window, min_periods=min_periods)
        else:
            raise ValueError(f"Unsupported function: {func}")

    @classmethod
    def calculate_realized_volatility(
        cls,
        df: pl.DataFrame,
        window: int = 20,
        annualize: bool = True
    ) -> pl.DataFrame:
        """
        Parkinson実現ボラティリティの正しい計算
        
        Args:
            df: High, Low列を含むDataFrame
            window: ローリング窓サイズ
            annualize: 年率換算するか
            
        Returns:
            realized_vol列を追加したDataFrame
        """
        # Parkinson estimator: 日次
        df = df.with_columns([
            ((pl.col("High") / pl.col("Low")).log() ** 2 / (4 * np.log(2)))
            .alias("pk_daily")
        ])

        # ローリング平均→平方根（min_periods適用）
        df = df.with_columns([
            cls.apply_min_periods_to_rolling(
                df, "pk_daily", window, "mean", min_periods=window
            ).alias("pk_mean")
        ])

        # 年率換算
        if annualize:
            df = df.with_columns([
                (pl.col("pk_mean") * 252).sqrt().alias(f"realized_vol_{window}")
            ])
        else:
            df = df.with_columns([
                pl.col("pk_mean").sqrt().alias(f"realized_vol_{window}")
            ])

        # 中間列を削除
        df = df.drop(["pk_daily", "pk_mean"])

        return df

    @classmethod
    def calculate_beta_with_lag(
        cls,
        stock_df: pl.DataFrame,
        market_df: pl.DataFrame,
        window: int = 60,
        lag: int = 1
    ) -> pl.DataFrame:
        """
        t-1固定のベータ計算
        
        Args:
            stock_df: 個別銘柄データ（returns_1d列必須）
            market_df: 市場データ（mkt_ret_1d列必須）
            window: ローリング窓サイズ
            lag: 市場リターンのラグ（デフォルト1）
            
        Returns:
            beta_60d列を追加したDataFrame
        """
        # 市場リターンをラグ
        market_lagged = market_df.with_columns([
            pl.col("mkt_ret_1d").shift(lag).alias("mkt_ret_lag")
        ]).select(["Date", "mkt_ret_lag"])

        # 結合
        df = stock_df.join(market_lagged, on="Date", how="left")

        # 共分散と分散の計算
        df = df.with_columns([
            # E[XY]
            (pl.col("returns_1d") * pl.col("mkt_ret_lag")).alias("xy"),
            # E[X]
            cls.apply_min_periods_to_rolling(
                df, "returns_1d", window, "mean", min_periods=window
            ).over("Code").alias("x_mean"),
            # E[Y]
            cls.apply_min_periods_to_rolling(
                df, "mkt_ret_lag", window, "mean", min_periods=window
            ).over("Code").alias("y_mean"),
        ])

        df = df.with_columns([
            # E[XY]
            cls.apply_min_periods_to_rolling(
                df, "xy", window, "mean", min_periods=window
            ).over("Code").alias("xy_mean"),
            # E[Y^2]
            cls.apply_min_periods_to_rolling(
                df,
                "mkt_ret_lag",
                window,
                "std",
                min_periods=window
            ).over("Code").pow(2).alias("y_var")
        ])

        # Beta = Cov(X,Y) / Var(Y)
        df = df.with_columns([
            ((pl.col("xy_mean") - pl.col("x_mean") * pl.col("y_mean")) /
             (pl.col("y_var") + 1e-12))
            .alias(f"beta_{window}d")
        ])

        # 中間列を削除
        df = df.drop(["mkt_ret_lag", "xy", "x_mean", "y_mean", "xy_mean", "y_var"])

        return df

    @classmethod
    def validate_no_leakage(cls, df: pl.DataFrame) -> dict[str, bool]:
        """
        リーク検査
        
        Args:
            df: 検証対象のDataFrame
            
        Returns:
            検査結果の辞書
        """
        results = {}

        # days_since_* が負でないか
        for col in df.columns:
            if "days_since" in col:
                negative_count = df.filter(pl.col(col) < 0).height
                results[f"{col}_no_leak"] = (negative_count == 0)
                if negative_count > 0:
                    logger.warning(f"Found {negative_count} negative values in {col}")

        # インパルスとdays_sinceの整合性
        if "stmt_imp_statement" in df.columns and "stmt_days_since_statement" in df.columns:
            inconsistent = df.filter(
                (pl.col("stmt_imp_statement") == 1) &
                (pl.col("stmt_days_since_statement") != 0)
            ).height
            results["stmt_impulse_consistent"] = (inconsistent == 0)

        if "flow_impulse" in df.columns and "days_since_flow" in df.columns:
            inconsistent = df.filter(
                (pl.col("flow_impulse") == 1) &
                (pl.col("days_since_flow") != 0)
            ).height
            results["flow_impulse_consistent"] = (inconsistent == 0)

        # (Code, Date)の一意性
        unique_count = df.select(["Code", "Date"]).n_unique()
        total_count = len(df)
        results["code_date_unique"] = (unique_count == total_count)

        return results

    @classmethod
    def remove_redundant_features(cls, df: pl.DataFrame) -> pl.DataFrame:
        """
        冗長な特徴量を削除（P1対応）
        
        Args:
            df: 特徴量を含むDataFrame
            
        Returns:
            冗長特徴量を削除したDataFrame
        """
        redundant_pairs = [
            # close_to_highとclose_to_lowは和が1だが、DATASET.mdでは両方必要なので削除しない
            # ("close_to_low", None),  # コメントアウト - 両方保持

            # SMAは削除してEMAに統一
            ("sma_5", None),
            ("sma_10", None),
            ("sma_20", None),
            ("sma_60", None),
            ("sma_120", None),
            ("price_to_sma5", None),
            ("price_to_sma20", None),
            ("price_to_sma60", None),

            # ターンオーバー率（発行株式数が不明な場合）
            ("turnover_rate", None),
        ]

        cols_to_drop = []
        for col, _ in redundant_pairs:
            if col in df.columns:
                cols_to_drop.append(col)

        if cols_to_drop:
            logger.info(f"Removing {len(cols_to_drop)} redundant features: {cols_to_drop}")
            df = df.drop(cols_to_drop)

        return df
