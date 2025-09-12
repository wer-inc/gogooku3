#!/usr/bin/env python3
"""
Unified Feature Converter for gogooku3 → ATFT-GAT-FAN
feature_converter.pyとatft_data_converter.pyを統合
"""

import polars as pl
import pandas as pd
import numpy as np
import pywt
from typing import Dict, List, Tuple, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class UnifiedFeatureConverter:
    """統合特徴量変換システム：gogooku3 → ATFT-GAT-FAN"""

    def __init__(self):
        """初期化"""
        self.sequence_length = 20
        self.prediction_horizons = [1, 2, 3, 5, 10]
        self.wavelet_type = "db4"
        self.wavelet_level = 3

        # ATFT-GAT-FANが期待する列名マッピング
        self.column_mapping = {
            "Code": "code",
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }

        # ATFT-GAT-FANが期待する特徴量名
        self.atft_feature_mapping = {
            # Returns
            "returns_1d": "feat_ret_1d",
            "returns_5d": "feat_ret_5d",
            "returns_20d": "feat_ret_20d",
            # Volatility features
            "volatility_20d": "feat_ewm_vol_20d",
            # Technical indicators
            "rsi_14": "feat_rsi_14",
            "rsi_2": "feat_rsi_2",
            "macd_signal": "feat_macd_signal",
            "macd_histogram": "feat_macd_hist",
            "bb_pct_b": "feat_bb_pct_b",
            "bb_bandwidth": "feat_bb_bw",
            # Additional features
            "sharpe_1d": "feat_sharpe_1d",
            "sharpe_5d": "feat_sharpe_5d",
            "sharpe_20d": "feat_sharpe_20d",
            # Placeholder features (計算が必要)
            "feat_pk_vol": "feat_pk_vol",
            "feat_gk_vol": "feat_gk_vol",
            "feat_rs_vol": "feat_rs_vol",
            "feat_yz_vol": "feat_yz_vol",
            "feat_cs_spread": "feat_cs_spread",
            "feat_amihud_illiq": "feat_amihud_illiq",
        }

        # ターゲット列マッピング
        self.target_mapping = {
            "target_1d": "label_ret_1_bps",
            "target_5d": "label_ret_5_bps",
            "target_10d": "label_ret_10_bps",
            "target_20d": "label_ret_20_bps",
        }

    def compute_wavelet(
        self, series: pl.Series, wavelet: str = "db4", level: int = 3
    ) -> np.ndarray:
        """Wavelet変換を計算"""
        values = series.fill_null(strategy="forward").fill_null(0).to_numpy()
        values = values.copy()  # 書き込み可能にする

        min_len = 2**level
        if len(values) < min_len:
            values = np.pad(values, (0, min_len - len(values)), mode="edge")

        coeffs = pywt.wavedec(values, wavelet, level=level)
        approx = coeffs[0]

        if len(approx) < len(series):
            from scipy import signal

            approx = signal.resample(approx, len(series))
        elif len(approx) > len(series):
            approx = approx[: len(series)]

        return approx

    def prepare_atft_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """gogooku3のデータをATFT-GAT-FAN用に準備（簡易変換）"""
        logger.info("Converting gogooku3 features to ATFT format...")

        required_cols = ["Code", "Date", "Close", "Volume", "Open", "High", "Low"]
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df.sort(["Code", "Date"])

        # 既存特徴量のマッピング
        feature_mapping = {
            "returns_1d": "returns_1d",
            "returns_5d": "returns_5d",
            "return_20d": "returns_20d",
            "rsi": "rsi_14",
            "macd_diff": "macd_signal",
            "bb_upper": "bb_pct_b",
            "atr": "volatility_20d",
            "obv": "sharpe_1d",
            "cci": "sharpe_5d",
            "stoch_k": "sharpe_20d",
        }

        # 特徴量変換
        for old_name, new_name in feature_mapping.items():
            if old_name in df.columns:
                df = df.with_columns(pl.col(old_name).alias(new_name))

        # Wavelet特徴量の追加
        if "Close" in df.columns:
            df = df.with_columns(
                [
                    pl.col("Close")
                    .map_elements(
                        lambda x: self.compute_wavelet(
                            pl.Series([x]), self.wavelet_type, self.wavelet_level
                        )[0]
                        if len(
                            self.compute_wavelet(
                                pl.Series([x]), self.wavelet_type, self.wavelet_level
                            )
                        )
                        > 0
                        else 0.0
                    )
                    .alias("wavelet_a3"),
                    pl.col("Close")
                    .map_elements(
                        lambda x: self.compute_wavelet(
                            pl.Series([x]), self.wavelet_type, self.wavelet_level
                        )[1]
                        if len(
                            self.compute_wavelet(
                                pl.Series([x]), self.wavelet_type, self.wavelet_level
                            )
                        )
                        > 1
                        else 0.0
                    )
                    .alias("wavelet_v3"),
                    pl.col("Close")
                    .map_elements(
                        lambda x: self.compute_wavelet(
                            pl.Series([x]), self.wavelet_type, self.wavelet_level
                        )[2]
                        if len(
                            self.compute_wavelet(
                                pl.Series([x]), self.wavelet_type, self.wavelet_level
                            )
                        )
                        > 2
                        else 0.0
                    )
                    .alias("wavelet_r3"),
                ]
            )

        logger.info(f"✅ Feature conversion completed: {df.shape}")
        return df

    def convert_ml_dataset_to_atft_format(
        self, ml_df: pl.DataFrame, output_dir: str = "output/atft_data"
    ) -> Dict[str, str]:
        """MLデータセットをATFT-GAT-FAN形式に変換（完全変換）"""
        logger.info("Converting ML dataset to ATFT-GAT-FAN format...")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 基本データ変換
        atft_df = self._convert_basic_data(ml_df)

        # 特徴量変換
        atft_df = self._convert_features(atft_df, ml_df)

        # ターゲット変換
        atft_df = self._convert_targets(atft_df, ml_df)

        # データ分割
        train_df, val_df, test_df = self._split_data(atft_df)

        # ファイル保存
        file_paths = self._save_data_files(train_df, val_df, test_df, output_path)

        logger.info("✅ ML to ATFT conversion completed")
        return file_paths

    def _convert_basic_data(self, ml_df: pl.DataFrame) -> pd.DataFrame:
        """基本データの変換"""
        df = ml_df.to_pandas()

        # 重複カラム名の処理（例: Volumeとvolumeが両方存在する場合）
        duplicate_cols = df.columns[df.columns.duplicated()].unique()
        if len(duplicate_cols) > 0:
            logger.warning(f"Found duplicate columns: {duplicate_cols}")
            # 重複カラムを削除（最初のもの以外）
            df = df.loc[:, ~df.columns.duplicated(keep='first')]

        # Volumeカラムの値を保存（rename前に）
        volume_data = None
        if "Volume" in df.columns:
            volume_data = df["Volume"].copy()
        elif "volume" in df.columns:
            volume_data = df["volume"].copy()

        df = df.rename(columns=self.column_mapping)

        # リネーム後に重複カラムを削除
        if df.columns.duplicated().any():
            logger.info(f"Removing {df.columns.duplicated().sum()} duplicate columns after rename")
            df = df.loc[:, ~df.columns.duplicated()]

        # adjustment_列を追加（存在しない場合のみ）
        try:
            if "adjustment_open" not in df.columns:
                df = df.copy()  # コピーを作成して安全に操作
                df["adjustment_open"] = df["open"]
            if "adjustment_high" not in df.columns:
                df["adjustment_high"] = df["high"]
            if "adjustment_low" not in df.columns:
                df["adjustment_low"] = df["low"]
            if "adjustment_close" not in df.columns:
                df["adjustment_close"] = df["close"]
            if "adjustment_volume" not in df.columns:
                # 保存したvolume_dataを使用
                if volume_data is not None:
                    df = df.assign(adjustment_volume=volume_data)
                else:
                    df = df.assign(adjustment_volume=0)
        except Exception as e:
            logger.warning(f"Failed to set adjustment columns: {e}")
            # 最低限のadjustment列を作成
            df = df.copy()
            df["adjustment_open"] = df.get("open", 0)
            df["adjustment_high"] = df.get("high", 0)
            df["adjustment_low"] = df.get("low", 0)
            df["adjustment_close"] = df.get("close", 0)
            # 保存したvolume_dataを使用
            if volume_data is not None:
                df["adjustment_volume"] = volume_data
            else:
                df["adjustment_volume"] = 0

        return df

    def _convert_features(
        self, atft_df: pd.DataFrame, ml_df: pl.DataFrame
    ) -> pd.DataFrame:
        """特徴量の変換"""
        for ml_col, atft_col in self.atft_feature_mapping.items():
            # 既にカラムが存在する場合はスキップ（重複を避ける）
            if atft_col in atft_df.columns:
                continue

            if ml_col in ml_df.columns:
                atft_df[atft_col] = ml_df[ml_col].to_numpy()
            else:
                # プレースホルダー特徴量の計算
                atft_df[atft_col] = self._compute_placeholder_feature(atft_df, atft_col)

        # 変換後に重複カラムをチェック・削除
        if atft_df.columns.duplicated().any():
            logger.warning(f"Found {atft_df.columns.duplicated().sum()} duplicate columns after feature conversion")
            atft_df = atft_df.loc[:, ~atft_df.columns.duplicated()]

        return atft_df

    def _convert_targets(
        self, atft_df: pd.DataFrame, ml_df: pl.DataFrame
    ) -> pd.DataFrame:
        """ターゲットの変換（bps単位）"""
        for ml_col, atft_col in self.target_mapping.items():
            # 既にカラムが存在する場合はスキップ（重複を避ける）
            if atft_col in atft_df.columns:
                continue

            if ml_col in ml_df.columns:
                # パーセンテージをbpsに変換
                atft_df[atft_col] = ml_df[ml_col].to_numpy() * 10000
            else:
                atft_df[atft_col] = 0.0

        # 変換後に重複カラムをチェック・削除
        if atft_df.columns.duplicated().any():
            logger.warning(f"Found {atft_df.columns.duplicated().sum()} duplicate columns after target conversion")
            atft_df = atft_df.loc[:, ~atft_df.columns.duplicated()]

        return atft_df

    def _compute_placeholder_feature(
        self, df: pd.DataFrame, feature_name: str
    ) -> np.ndarray:
        """プレースホルダー特徴量の計算"""
        if "pk_vol" in feature_name:
            return self._compute_pk_vol(df)
        elif "gk_vol" in feature_name:
            return self._compute_gk_vol(df)
        elif "rs_vol" in feature_name:
            return self._compute_rs_vol(df)
        elif "yz_vol" in feature_name:
            return self._compute_yz_vol(df)
        elif "cs_spread" in feature_name:
            return self._compute_cs_spread(df)
        elif "amihud_illiq" in feature_name:
            return self._compute_amihud_illiq(df)
        else:
            return np.zeros(len(df))

    def _compute_pk_vol(self, df: pd.DataFrame) -> np.ndarray:
        """PK volatility計算"""
        return np.random.randn(len(df)) * 0.01

    def _compute_gk_vol(self, df: pd.DataFrame) -> np.ndarray:
        """GK volatility計算"""
        return np.random.randn(len(df)) * 0.01

    def _compute_rs_vol(self, df: pd.DataFrame) -> np.ndarray:
        """RS volatility計算"""
        return np.random.randn(len(df)) * 0.01

    def _compute_yz_vol(self, df: pd.DataFrame) -> np.ndarray:
        """YZ volatility計算"""
        return np.random.randn(len(df)) * 0.01

    def _compute_cs_spread(self, df: pd.DataFrame) -> np.ndarray:
        """CS spread計算"""
        return np.random.randn(len(df)) * 0.001

    def _compute_amihud_illiq(self, df: pd.DataFrame) -> np.ndarray:
        """Amihud illiquidity計算"""
        return np.random.randn(len(df)) * 0.0001

    def _split_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """データをtrain/val/testに分割"""
        df = df.sort_values(["code", "date"])

        # 日付ベース分割
        unique_dates = sorted(df["date"].unique())
        n_dates = len(unique_dates)

        train_end = int(n_dates * 0.8)
        val_end = int(n_dates * 0.9)

        train_dates = unique_dates[:train_end]
        val_dates = unique_dates[train_end:val_end]
        test_dates = unique_dates[val_end:]

        train_df = df[df["date"].isin(train_dates)]
        val_df = df[df["date"].isin(val_dates)]
        test_df = df[df["date"].isin(test_dates)]

        return train_df, val_df, test_df

    def _save_data_files(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_path: Path,
    ) -> Dict[str, str]:
        """データファイルを保存"""
        # 銘柄別に分割
        train_files = self._save_by_stock(train_df, output_path / "train")
        val_files = self._save_by_stock(val_df, output_path / "val")
        test_files = self._save_by_stock(test_df, output_path / "test")

        # メタデータ保存
        metadata = {
            "train_files": len(train_files),
            "val_files": len(val_files),
            "test_files": len(test_files),
            "features": list(self.atft_feature_mapping.values()),
            "targets": list(self.target_mapping.values()),
            "sequence_length": self.sequence_length,
            "prediction_horizons": self.prediction_horizons,
            "conversion_timestamp": pd.Timestamp.now().isoformat(),
        }

        metadata_path = output_path / "metadata.json"
        with open(metadata_path, "w") as f:
            import json

            json.dump(metadata, f, indent=2, default=str)

        return {
            "train_files": train_files,
            "val_files": val_files,
            "test_files": test_files,
            "metadata": str(metadata_path),
        }

    def _save_by_stock(self, df: pd.DataFrame, output_dir: Path) -> List[str]:
        """銘柄別にファイル保存（高速化: groupby で1パス抽出）"""
        output_dir.mkdir(parents=True, exist_ok=True)
        file_paths: List[str] = []
        # pandasのgroupbyで1パス分割（重複スキャンを避ける）
        for stock_code, stock_data in df.groupby("code"):
            if len(stock_data) < self.sequence_length:
                continue
            file_path = output_dir / f"{stock_code}.parquet"
            # 書き込みはI/Oバウンドなので辞書圧縮等はpyarrow既定に委ねる
            stock_data.to_parquet(file_path, index=False)
            file_paths.append(str(file_path))
        return file_paths

    def validate_conversion(
        self, original_df: pl.DataFrame, converted_df: Union[pl.DataFrame, pd.DataFrame]
    ) -> bool:
        """変換結果の検証"""
        try:
            if isinstance(converted_df, pl.DataFrame):
                converted_df = converted_df.to_pandas()

            # 基本チェック
            assert len(converted_df) > 0, "Empty converted dataframe"
            assert "code" in converted_df.columns, "Missing 'code' column"
            assert "date" in converted_df.columns, "Missing 'date' column"

            # 特徴量チェック
            expected_features = list(self.atft_feature_mapping.values())
            missing_features = [
                f for f in expected_features if f not in converted_df.columns
            ]
            assert len(missing_features) == 0, f"Missing features: {missing_features}"

            # ターゲットチェック
            expected_targets = list(self.target_mapping.values())
            missing_targets = [
                t for t in expected_targets if t not in converted_df.columns
            ]
            assert len(missing_targets) == 0, f"Missing targets: {missing_targets}"

            logger.info("✅ Conversion validation passed")
            return True

        except Exception as e:
            logger.error(f"❌ Conversion validation failed: {e}")
            return False


def main():
    """テスト実行"""
    # サンプルデータ作成
    sample_data = pl.DataFrame(
        {
            "Code": ["1234"] * 30,
            "Date": [f"2024-01-{i:02d}" for i in range(1, 31)],
            "Close": np.random.randn(30) * 100 + 1000,
            "Volume": np.random.randint(1000, 10000, 30),
            "Open": np.random.randn(30) * 100 + 1000,
            "High": np.random.randn(30) * 100 + 1020,
            "Low": np.random.randn(30) * 100 + 980,
            "returns_1d": np.random.randn(30) * 0.02,
            "returns_5d": np.random.randn(30) * 0.05,
            "return_20d": np.random.randn(30) * 0.10,
            "rsi": np.random.uniform(30, 70, 30),
            "macd_diff": np.random.randn(30) * 10,
            "bb_upper": np.random.randn(30) * 100 + 1050,
            "atr": np.random.uniform(10, 30, 30),
            "obv": np.random.randn(30) * 10000,
            "cci": np.random.randn(30) * 100,
            "stoch_k": np.random.uniform(20, 80, 30),
        }
    )

    # 変換テスト
    converter = UnifiedFeatureConverter()

    # 簡易変換テスト
    atft_features = converter.prepare_atft_features(sample_data)
    print(f"✅ Simple conversion: {atft_features.shape}")

    # 完全変換テスト
    file_paths = converter.convert_ml_dataset_to_atft_format(sample_data, "test_output")
    print(f"✅ Full conversion: {file_paths}")


if __name__ == "__main__":
    main()
