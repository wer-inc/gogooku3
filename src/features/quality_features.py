"""
Quality Financial Features Generator
高品質金融特徴量生成器

主な機能:
- Sharpe分母の明示化（daily_vol列）
- Flow指標（SMI、±2σフラグ）
- 高低ボラflag（CS分位・ローリング分位）
- 命名統一とクリーンアップ
- 指数残差リターン
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)


class QualityFinancialFeaturesGenerator:
    """
    高品質金融特徴量生成器

    生成する特徴量:
    1. 明示的Sharpe分母（daily_vol）
    2. Flow指標（SMI、デルタ、±2σフラグ）
    3. 高低ボラフラグ（分位ベース）
    4. 指数残差リターン
    5. 統一命名とクリーンアップ
    """

    def __init__(
        self,
        volatility_column: str = "volatility_20d",
        volume_columns: list[str] | None = None,
        index_return_column: str | None = None,
        use_cross_sectional_quantiles: bool = True,
        rolling_window: int = 252,
        flow_window: int = 60,
        sigma_threshold: float = 2.0,
        verbose: bool = True,
    ):
        """
        Args:
            volatility_column: ボラティリティ列名
            volume_columns: 出来高関連列名
            index_return_column: 指数リターン列名
            use_cross_sectional_quantiles: CS分位を使うか
            rolling_window: ローリング窓サイズ
            flow_window: フロー計算窓サイズ
            sigma_threshold: シグマ閾値
            verbose: 詳細ログ
        """
        self.volatility_column = volatility_column
        self.volume_columns = volume_columns or ["adjustment_volume", "turnover_value"]
        self.index_return_column = index_return_column
        self.use_cross_sectional_quantiles = use_cross_sectional_quantiles
        self.rolling_window = rolling_window
        self.flow_window = flow_window
        self.sigma_threshold = sigma_threshold
        self.verbose = verbose

        # 特徴量名の統一マッピング
        self.feature_name_mapping = {
            "ema_gap_5": "ema_gap_5d",
            "ema_gap_10": "ema_gap_10d",
            "ema_gap_20": "ema_gap_20d",
            "ma_gap_5": "ema_gap_5d",
            "ma_gap_10": "ema_gap_10d",
            "ma_gap_20": "ema_gap_20d",
            "macd_histogram": "macd_hist",
            "bb_width": "bb_width_20",
            "bb_position": "bb_pos_20",
        }

        if self.verbose:
            logger.info("QualityFinancialFeaturesGenerator initialized")

    def _convert_to_polars(self, df: pd.DataFrame | pl.DataFrame) -> pl.DataFrame:
        """DataFrameをPolarsに変換"""
        if isinstance(df, pd.DataFrame):
            return pl.from_pandas(df)
        return df

    def _add_daily_volatility(self, df: pl.DataFrame) -> pl.DataFrame:
        """日次ボラティリティを明示化"""
        try:
            if self.volatility_column in df.columns:
                # 年率ボラティリティを日次に変換
                df = df.with_columns(
                    [(pl.col(self.volatility_column) / np.sqrt(252)).alias("daily_vol")]
                )

                if self.verbose:
                    logger.debug("Added daily_vol column from volatility_20d")
            else:
                # フォールバック: リターンの移動標準偏差
                if "return_1d" in df.columns:
                    df = df.with_columns(
                        [
                            pl.col("return_1d")
                            .rolling_std(window_size=20, min_periods=5)
                            .alias("daily_vol")
                        ]
                    )
                    logger.debug("Added daily_vol from return_1d rolling std")
                else:
                    # デフォルト値
                    df = df.with_columns([pl.lit(0.02).alias("daily_vol")])
                    logger.warning("Used default daily_vol=0.02")

            return df

        except Exception as e:
            logger.warning(f"Failed to add daily_vol: {e}")
            return df.with_columns([pl.lit(0.02).alias("daily_vol")])

    def _add_explicit_sharpe_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """明示的Sharpe分母を使った特徴量"""
        try:
            # 各ホライズンでのSharpe比率
            horizons = [1, 5, 10, 20]

            sharpe_exprs = []
            for h in horizons:
                return_col = f"return_{h}d"
                if return_col in df.columns:
                    # Sharpe = リターン / (日次vol * √h)
                    sharpe_expr = (
                        pl.col(return_col) / (pl.col("daily_vol") * np.sqrt(h) + 1e-12)
                    ).alias(f"sharpe_{h}d")
                    sharpe_exprs.append(sharpe_expr)

            if sharpe_exprs:
                df = df.with_columns(sharpe_exprs)
                if self.verbose:
                    logger.debug(f"Added {len(sharpe_exprs)} explicit Sharpe features")

            return df

        except Exception as e:
            logger.warning(f"Failed to add Sharpe features: {e}")
            return df

    def _add_flow_indicators(self, df: pl.DataFrame) -> pl.DataFrame:
        """Flow指標を追加"""
        try:
            flow_exprs = []

            # 出来高ベースFlow（仮想的な外国人・個人フロー）
            if "adjustment_volume" in df.columns and "turnover_value" in df.columns:
                # SMI = (外国人流入 - 個人流入) / 総流入の概算
                # 簡易実装: volume_spikeとprice_actionの組み合わせ

                # Z-score計算（ローリングまたはCS分位）
                if self.use_cross_sectional_quantiles:
                    # GPU-ETL（任意）: USE_GPU_ETL=1 ならcuDFでz-scoreを計算
                    use_gpu = os.getenv("USE_GPU_ETL", "0") == "1"
                    if use_gpu:
                        try:
                            from src.utils.gpu_etl import cs_z  # type: ignore

                            df = cs_z(
                                df,
                                value_col="adjustment_volume",
                                group_keys=("date",),
                                out_name="__vol_cs_z",
                            )
                            df = cs_z(
                                df,
                                value_col="turnover_value",
                                group_keys=("date",),
                                out_name="__tov_cs_z",
                            )
                            volume_z = pl.col("__vol_cs_z")
                            turnover_z = pl.col("__tov_cs_z")
                        except Exception:
                            use_gpu = False
                    if not use_gpu:
                        # 日次クロスセクション分位（CPU）
                        volume_z = (
                            pl.col("adjustment_volume")
                            - pl.col("adjustment_volume").mean().over("date")
                        ) / (pl.col("adjustment_volume").std().over("date") + 1e-12)

                        turnover_z = (
                            pl.col("turnover_value")
                            - pl.col("turnover_value").mean().over("date")
                        ) / (pl.col("turnover_value").std().over("date") + 1e-12)
                else:
                    # ローリングZ-score
                    volume_z = (
                        pl.col("adjustment_volume")
                        - pl.col("adjustment_volume").rolling_mean(
                            window_size=self.flow_window, min_periods=10
                        )
                    ) / (
                        pl.col("adjustment_volume").rolling_std(
                            window_size=self.flow_window, min_periods=10
                        )
                        + 1e-12
                    )

                    turnover_z = (
                        pl.col("turnover_value")
                        - pl.col("turnover_value").rolling_mean(
                            window_size=self.flow_window, min_periods=10
                        )
                    ) / (
                        pl.col("turnover_value").rolling_std(
                            window_size=self.flow_window, min_periods=10
                        )
                        + 1e-12
                    )

                # SMI（仮想フロー指標）
                smi = volume_z - turnover_z
                flow_exprs.append(smi.alias("smi"))

                # ±2σフラグ
                flow_exprs.extend(
                    [
                        (volume_z > self.sigma_threshold).alias("volume_spike_2sigma"),
                        (volume_z < -self.sigma_threshold).alias("volume_drop_2sigma"),
                        (turnover_z > self.sigma_threshold).alias(
                            "turnover_spike_2sigma"
                        ),
                        (smi > self.sigma_threshold).alias("flow_positive_2sigma"),
                        (smi < -self.sigma_threshold).alias("flow_negative_2sigma"),
                    ]
                )

                # Δ5d（5日変化）
                if "return_5d" in df.columns:
                    smi_delta = smi - smi.shift(5, fill_value=0)
                    flow_exprs.append(smi_delta.alias("smi_delta_5d"))

            if flow_exprs:
                df = df.with_columns(flow_exprs)
                if self.verbose:
                    logger.debug(f"Added {len(flow_exprs)} Flow indicators")

            # GPUで追加した一時列はクリーンアップ
            drop_tmp = [c for c in ("__vol_cs_z", "__tov_cs_z") if c in df.columns]
            if drop_tmp:
                df = df.drop(drop_tmp)
            return df

        except Exception as e:
            logger.warning(f"Failed to add Flow indicators: {e}")
            return df

    def _add_volatility_flags(self, df: pl.DataFrame) -> pl.DataFrame:
        """高低ボラflagを追加"""
        try:
            if self.volatility_column not in df.columns:
                return df

            vol_flag_exprs = []

            if self.use_cross_sectional_quantiles:
                # 同日クロスセクション分位（GPU-ETL対応）
                use_gpu = os.getenv("USE_GPU_ETL", "0") == "1"
                vol_quantile = None
                if use_gpu:
                    try:
                        from src.utils.gpu_etl import cs_rank_and_z  # type: ignore

                        # cs_rank_and_z でランク正規化（0..1）を取得
                        tmp = cs_rank_and_z(
                            df.select(["date", self.volatility_column]).with_row_count(
                                "__rid__"
                            ),
                            rank_col=self.volatility_column,
                            z_col=self.volatility_column,
                            group_keys=("date",),
                            out_rank_name="__vol_q",
                            out_z_name="__dummy",
                        )
                        tmp = tmp.sort("__rid__").select(["__vol_q"])
                        df = df.hstack([tmp])
                        vol_quantile = pl.col("__vol_q")
                    except Exception:
                        use_gpu = False
                if not use_gpu:
                    vol_quantile = pl.col(self.volatility_column).rank(
                        method="average"
                    ).over("date") / pl.col(self.volatility_column).count().over("date")

                vol_flag_exprs.extend(
                    [
                        (vol_quantile >= 0.8).alias("high_vol_flag"),
                        (vol_quantile <= 0.2).alias("low_vol_flag"),
                        (vol_quantile).alias("vol_cs_quantile"),
                    ]
                )
            else:
                # ローリング分位
                vol_rolling_mean = pl.col(self.volatility_column).rolling_mean(
                    window_size=self.rolling_window, min_periods=60
                )
                vol_rolling_std = pl.col(self.volatility_column).rolling_std(
                    window_size=self.rolling_window, min_periods=60
                )

                vol_z_score = (pl.col(self.volatility_column) - vol_rolling_mean) / (
                    vol_rolling_std + 1e-12
                )

                vol_flag_exprs.extend(
                    [
                        (vol_z_score > 1.5).alias("high_vol_flag"),
                        (vol_z_score < -1.5).alias("low_vol_flag"),
                        vol_z_score.alias("vol_rolling_zscore"),
                    ]
                )

            if vol_flag_exprs:
                df = df.with_columns(vol_flag_exprs)
                if self.verbose:
                    logger.debug(f"Added {len(vol_flag_exprs)} volatility flags")

            # GPU一時列の掃除
            if "__vol_q" in df.columns:
                df = df.drop(["__vol_q"])
            return df

        except Exception as e:
            logger.warning(f"Failed to add volatility flags: {e}")
            return df

    def _add_index_excess_returns(self, df: pl.DataFrame) -> pl.DataFrame:
        """指数残差リターンを追加"""
        try:
            if (
                self.index_return_column is None
                or self.index_return_column not in df.columns
            ):
                return df

            excess_exprs = []
            horizons = [1, 5, 10, 20]

            for h in horizons:
                return_col = f"return_{h}d"
                if return_col in df.columns:
                    # 簡易ベータ = 1.0と仮定（実際はローリング回帰で計算すべき）
                    beta = 1.0

                    excess_return = pl.col(return_col) - beta * pl.col(
                        self.index_return_column
                    )

                    excess_exprs.append(excess_return.alias(f"excess_return_{h}d"))

            if excess_exprs:
                df = df.with_columns(excess_exprs)
                if self.verbose:
                    logger.debug(f"Added {len(excess_exprs)} excess return features")

            return df

        except Exception as e:
            logger.warning(f"Failed to add excess returns: {e}")
            return df

    def _standardize_feature_names(self, df: pl.DataFrame) -> pl.DataFrame:
        """特徴量名を統一"""
        try:
            # 名前変更
            rename_dict = {}
            for old_name, new_name in self.feature_name_mapping.items():
                if old_name in df.columns:
                    rename_dict[old_name] = new_name

            if rename_dict:
                df = df.rename(rename_dict)
                if self.verbose:
                    logger.debug(f"Renamed {len(rename_dict)} columns: {rename_dict}")

            return df

        except Exception as e:
            logger.warning(f"Failed to standardize names: {e}")
            return df

    def _cleanup_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """不要な特徴量をクリーンアップ"""
        try:
            # 除外パターン
            drop_patterns = [
                "row_idx",  # インデックス列
                "_isna",  # 欠損フラグ（0埋めで対応）
                "_1d_1d",  # 重複パターン
                "_temp_",  # 一時列
            ]

            # Flow系の0埋め列をチェック
            flow_zero_cols = []
            if hasattr(df, "columns"):
                for col in df.columns:
                    # Flow系で全て0の列を特定
                    if (
                        any(pattern in col.lower() for pattern in ["flow", "smi"])
                        and col in df.columns
                    ):
                        try:
                            if df[col].sum() == 0 and df[col].std() == 0:
                                flow_zero_cols.append(col)
                        except:
                            pass

            # 削除対象列を決定
            drop_cols = []
            for col in df.columns:
                if any(pattern in col for pattern in drop_patterns):
                    drop_cols.append(col)

            drop_cols.extend(flow_zero_cols)

            # 重複除去
            drop_cols = list(set(drop_cols))

            # 実際に存在する列のみ削除
            drop_cols = [col for col in drop_cols if col in df.columns]

            if drop_cols:
                df = df.drop(drop_cols)
                if self.verbose:
                    logger.debug(f"Dropped {len(drop_cols)} columns: {drop_cols}")

            return df

        except Exception as e:
            logger.warning(f"Failed to cleanup features: {e}")
            return df

    def generate_quality_features(
        self, data: pd.DataFrame | pl.DataFrame
    ) -> pl.DataFrame:
        """
        高品質特徴量を生成

        Args:
            data: 入力データ

        Returns:
            特徴量拡張されたデータ
        """
        if self.verbose:
            logger.info("Generating quality financial features")

        # Polarsに変換
        df = self._convert_to_polars(data)

        original_cols = len(df.columns)

        try:
            # 1. 日次ボラティリティの明示化
            df = self._add_daily_volatility(df)

            # 2. 明示的Sharpe比率特徴量
            df = self._add_explicit_sharpe_features(df)

            # 3. Flow指標
            df = self._add_flow_indicators(df)

            # 4. 高低ボラフラグ
            df = self._add_volatility_flags(df)

            # 5. 指数残差リターン
            df = self._add_index_excess_returns(df)

            # 6. 特徴量名の統一
            df = self._standardize_feature_names(df)

            # 7. クリーンアップ
            df = self._cleanup_features(df)

            # 8. TOPIX市場特徴量の統合
            df = self._integrate_topix_features(df)

            final_cols = len(df.columns)
            added_cols = final_cols - original_cols

            if self.verbose:
                logger.info(
                    f"Quality features generation completed: "
                    f"{original_cols} → {final_cols} columns (+{added_cols})"
                )

            return df

        except Exception as e:
            logger.error(f"Quality features generation failed: {e}")
            return df

    def _integrate_topix_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        TOPIX市場特徴量を統合

        Args:
            df: 入力データフレーム

        Returns:
            TOPIX特徴量統合済みのデータフレーム
        """
        try:
            # TOPIX特徴量が既に存在するか確認
            existing_topix_cols = [
                col
                for col in df.columns
                if col.startswith(("mkt_", "beta_", "alpha_", "rel_"))
            ]

            if existing_topix_cols:
                if self.verbose:
                    logger.info(
                        f"TOPIX features already exist: {len(existing_topix_cols)} features"
                    )
                return df

            # TOPIX特徴量がなければ、MLDatasetBuilderを使って追加
            if self.verbose:
                logger.info("Integrating TOPIX market features...")

            try:
                from scripts.data.ml_dataset_builder import MLDatasetBuilder

                builder = MLDatasetBuilder()
                df_with_topix = builder.add_topix_features(df)

                topix_cols = [
                    col
                    for col in df_with_topix.columns
                    if col.startswith(("mkt_", "beta_", "alpha_", "rel_"))
                ]
                if self.verbose:
                    logger.info(f"Added {len(topix_cols)} TOPIX market features")

                return df_with_topix

            except ImportError:
                if self.verbose:
                    logger.warning(
                        "MLDatasetBuilder not available, skipping TOPIX integration"
                    )
                return df

        except Exception as e:
            logger.error(f"Error integrating TOPIX features: {e}")
            return df

    def get_feature_categories(self, columns: list[str]) -> dict[str, list[str]]:
        """特徴量をカテゴリ別に分類"""
        categories = {
            "price_features": [],
            "volume_features": [],
            "technical_indicators": [],
            "volatility_features": [],
            "flow_features": [],
            "sharpe_features": [],
            "flag_features": [],
            "excess_returns": [],
            "market_features": [],  # TOPIX市場特徴量
            "cross_features": [],  # 銘柄×市場クロス特徴量
            "other": [],
        }

        for col in columns:
            col_lower = col.lower()

            if any(
                pattern in col_lower
                for pattern in ["price", "close", "open", "high", "low"]
            ):
                categories["price_features"].append(col)
            elif any(pattern in col_lower for pattern in ["volume", "turnover"]):
                categories["volume_features"].append(col)
            elif any(
                pattern in col_lower for pattern in ["rsi", "ema", "macd", "bb_", "atr"]
            ):
                categories["technical_indicators"].append(col)
            elif any(pattern in col_lower for pattern in ["vol", "volatility"]):
                categories["volatility_features"].append(col)
            elif any(pattern in col_lower for pattern in ["flow", "smi"]):
                categories["flow_features"].append(col)
            elif "sharpe" in col_lower:
                categories["sharpe_features"].append(col)
            elif any(pattern in col_lower for pattern in ["flag", "spike", "_2sigma"]):
                categories["flag_features"].append(col)
            elif "excess" in col_lower:
                categories["excess_returns"].append(col)
            elif col.startswith("mkt_"):
                categories["market_features"].append(col)
            elif any(pattern in col_lower for pattern in ["beta_", "alpha_", "rel_"]):
                categories["cross_features"].append(col)
            else:
                categories["other"].append(col)

        # 空のカテゴリを除外
        return {k: v for k, v in categories.items() if v}

    def validate_features(self, df: pl.DataFrame) -> dict[str, Any]:
        """特徴量品質を検証"""
        validation_results = {
            "total_features": len(df.columns),
            "missing_daily_vol": "daily_vol" not in df.columns,
            "zero_variance_features": [],
            "high_missing_features": [],
            "infinite_features": [],
            "feature_categories": {},
        }

        try:
            # 各特徴量をチェック
            for col in df.columns:
                if col in ["date", "code"]:
                    continue

                try:
                    series = df[col]

                    # ゼロ分散チェック
                    if series.std() == 0:
                        validation_results["zero_variance_features"].append(col)

                    # 高欠損率チェック
                    null_rate = series.null_count() / len(series)
                    if null_rate > 0.5:
                        validation_results["high_missing_features"].append(col)

                    # 無限値チェック
                    if not series.is_finite().all():
                        validation_results["infinite_features"].append(col)

                except Exception:
                    continue

            # 特徴量カテゴリ分析
            validation_results["feature_categories"] = self.get_feature_categories(
                df.columns
            )

        except Exception as e:
            logger.warning(f"Feature validation failed: {e}")

        return validation_results
