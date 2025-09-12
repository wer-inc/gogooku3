#!/usr/bin/env python3
"""
ATFT-GAT-FAN Model Inference Wrapper for gogooku3
Sharpe比 0.849の高性能モデルを gogooku3 から利用
"""

import sys
from pathlib import Path
import torch
import numpy as np
import polars as pl
from typing import Dict, Optional, Union, List
import logging
from omegaconf import OmegaConf

# ATFT-GAT-FANのパスを追加
ATFT_PATH = Path("/home/ubuntu/gogooku2/apps/ATFT-GAT-FAN")
sys.path.insert(0, str(ATFT_PATH))

# ATFT-GAT-FANのインポート
try:
    from src.models.architectures.atft_gat_fan import ATFT_GAT_FAN
except ImportError:
    ATFT_GAT_FAN = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ATFTInference:
    """ATFT-GAT-FANモデルの推論ラッパー"""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize ATFT-GAT-FAN model for inference

        Args:
            checkpoint_path: Path to model checkpoint (default: best model)
            config_path: Path to model config (default: v1 config)
            device: Device to use (cuda/cpu/auto)
        """
        # デフォルトパス設定
        if checkpoint_path is None:
            checkpoint_path = ATFT_PATH / "models/checkpoints/atft_gat_fan_final.pt"
        # 設定ファイルのパス
        config_path = (
            Path(__file__).parent.parent.parent / "configs/atft/model/atft_gat_fan.yaml"
        )

        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)

        # デバイス設定
        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # モデル読み込み
        self.model = None
        self.config = None
        self.load_model()

    def load_model(self):
        """Load ATFT-GAT-FAN model and config"""
        # 設定読み込み
        logger.info(f"Loading config from: {self.config_path}")
        model_config = OmegaConf.load(self.config_path)

        # 完全なconfig構造を作成（train.pyと同じ形式）
        self.config = OmegaConf.create(
            {
                "model": model_config,
                "data": {
                    "features": {
                        "input_dim": 8  # チェックポイントと一致する値
                    },
                    "time_series": {"prediction_horizons": [1, 2, 3, 5, 10]},
                    "sequence_length": 20,
                    "prediction_horizons": [1, 2, 3, 5, 10],
                    "num_features": 13,
                },
                "train": {"batch": {"train_batch_size": 512, "val_batch_size": 1024}},
            }
        )

        # モデル初期化
        logger.info("Initializing ATFT-GAT-FAN model...")
        self.model = ATFT_GAT_FAN(self.config)

        # チェックポイント読み込み
        logger.info(f"Loading checkpoint from: {self.checkpoint_path}")
        checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )

        # モデルウェイト読み込み
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        # モデル情報表示
        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info("✅ Model loaded successfully")
        logger.info(f"   Parameters: {param_count:,}")
        logger.info("   Expected Sharpe Ratio: 0.849")

    @torch.no_grad()
    def predict(
        self,
        features: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        horizon: Union[int, List[int]] = [1, 2, 3, 5, 10],
        return_confidence: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Run inference with ATFT-GAT-FAN

        Args:
            features: Input features tensor [batch, seq_len, features]
            edge_index: Graph edge indices [2, num_edges]
            edge_attr: Graph edge attributes [num_edges, edge_features]
            horizon: Prediction horizons (1, 2, 3, 5, or 10)
            return_confidence: Return confidence intervals

        Returns:
            Dictionary with predictions and optionally confidence
        """
        self.model.eval()

        # バッチ準備 - ATFT-GAT-FANはdynamic_featuresを直接テンソルとして期待
        dynamic_features = features.to(self.device)

        # 推論実行
        outputs = self.model(
            dynamic_features=dynamic_features,
            edge_index=edge_index.to(self.device) if edge_index is not None else None,
            edge_attr=edge_attr.to(self.device) if edge_attr is not None else None,
        )

        # 結果処理
        results = {}

        # 予測値取得 - ATFT-GAT-FANはpoint_horizon_X形式で返す
        predictions_list = []
        for h in [1, 2, 3, 5, 10]:
            key = f"point_horizon_{h}"
            if key in outputs:
                predictions_list.append(outputs[key].cpu().numpy())

        if predictions_list:
            predictions = np.column_stack(predictions_list)

            # horizon指定に応じて結果を抽出
            if isinstance(horizon, int):
                horizon_idx = [1, 2, 3, 5, 10].index(horizon)
                results["predictions"] = predictions[:, horizon_idx]
            else:
                results["predictions"] = predictions

        # 信頼度取得（Student-t分布のσから計算）
        if return_confidence:
            confidence_list = []
            for h in [1, 2, 3, 5, 10]:
                key = f"t_params_horizon_{h}"
                if key in outputs:
                    # Student-tパラメータから信頼度を計算
                    t_params = outputs[key].cpu().numpy()
                    if t_params.shape[-1] >= 2:  # df, loc, scale
                        scale = t_params[:, 2]  # scale parameter
                        confidence = 1.0 / (scale + 1e-8)
                        confidence_list.append(confidence)

            if confidence_list:
                confidence = np.column_stack(confidence_list)

                if isinstance(horizon, int):
                    horizon_idx = [1, 2, 3, 5, 10].index(horizon)
                    results["confidence"] = confidence[:, horizon_idx]
                else:
                    results["confidence"] = confidence

        return results

    def predict_from_dataframe(
        self,
        df: pl.DataFrame,
        sequence_length: int = 20,
        horizon: Union[int, List[int]] = [1, 2, 3, 5, 10],
        batch_size: int = 256,
    ) -> pl.DataFrame:
        """
        Predict from Polars DataFrame

        Args:
            df: Input dataframe with required features
            sequence_length: Sequence length for time series
            horizon: Prediction horizons
            batch_size: Batch size for inference

        Returns:
            DataFrame with predictions
        """
        # 必要な特徴量の確認（8次元に制限）
        required_features = [
            "return_1d",
            "return_5d",
            "return_20d",
            "rsi",
            "macd",
            "bb_upper",
            "atr",
            "obv",
        ]

        missing = set(required_features) - set(df.columns)
        if missing:
            logger.warning(f"Missing features: {missing}")
            logger.warning("Using unified feature converter to create missing features")
            from .unified_feature_converter import UnifiedFeatureConverter

            converter = UnifiedFeatureConverter()
            df = converter.prepare_atft_features(df)

        # データをテンソルに変換
        features_list = []
        stock_codes = df.get_column("Code").unique().to_list()

        for code in stock_codes:
            stock_data = df.filter(pl.col("Code") == code).sort("Date")

            # シーケンス作成
            if len(stock_data) >= sequence_length:
                # 特徴量抽出
                feature_cols = [
                    col for col in required_features if col in stock_data.columns
                ]
                if len(feature_cols) != len(required_features):
                    missing = set(required_features) - set(feature_cols)
                    logger.warning(f"Missing features for stock {code}: {missing}")

                    # 利用可能な特徴量のみを使用
                    available_features = stock_data.select(feature_cols).to_numpy()

                    # 不足分をゼロで埋める
                    padded_features = np.zeros(
                        (len(stock_data), len(required_features))
                    )
                    for i, feature in enumerate(required_features):
                        if feature in feature_cols:
                            col_idx = feature_cols.index(feature)
                            padded_features[:, i] = available_features[:, col_idx]

                    features = padded_features
                else:
                    features = stock_data.select(feature_cols).to_numpy()

                # シーケンス分割
                for i in range(len(features) - sequence_length + 1):
                    seq = features[i : i + sequence_length]
                    features_list.append(seq)

        if not features_list:
            logger.warning("No valid sequences found")
            return pl.DataFrame()

        # バッチ処理
        features_tensor = torch.FloatTensor(np.array(features_list))
        predictions_list = []

        for i in range(0, len(features_tensor), batch_size):
            batch = features_tensor[i : i + batch_size]
            results = self.predict(batch, horizon=horizon)
            predictions_list.append(results["predictions"])

        # 結果をDataFrameに変換
        all_predictions = np.vstack(predictions_list)

        # 結果DataFrame作成
        result_data = []
        pred_idx = 0

        for code in stock_codes:
            stock_data = df.filter(pl.col("Code") == code).sort("Date")

            if len(stock_data) >= sequence_length:
                dates = stock_data.get_column("Date").to_list()[sequence_length - 1 :]

                for date in dates:
                    if pred_idx < len(all_predictions):
                        if isinstance(horizon, int):
                            pred_value = float(all_predictions[pred_idx])
                            result_data.append(
                                {
                                    "Code": code,
                                    "Date": date,
                                    f"prediction_{horizon}d": pred_value,
                                }
                            )
                        else:
                            row = {"Code": code, "Date": date}
                            for h_idx, h in enumerate([1, 2, 3, 5, 10]):
                                row[f"prediction_{h}d"] = float(
                                    all_predictions[pred_idx, h_idx]
                                )
                            result_data.append(row)
                        pred_idx += 1

        return pl.DataFrame(result_data)

    def calculate_sharpe_ratio(
        self,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
        risk_free_rate: float = 0.0,
    ) -> float:
        """
        Calculate Sharpe ratio for predictions

        Args:
            predictions: Predicted returns
            actual_returns: Actual returns
            risk_free_rate: Risk-free rate (annualized)

        Returns:
            Sharpe ratio
        """
        # ポートフォリオリターン計算（予測に基づく重み付け）
        weights = np.sign(predictions)  # Simple long/short based on prediction
        portfolio_returns = weights * actual_returns

        # Sharpe比計算
        excess_returns = (
            portfolio_returns - risk_free_rate / 252
        )  # Daily risk-free rate
        sharpe = (
            np.mean(excess_returns) / (np.std(excess_returns) + 1e-8) * np.sqrt(252)
        )

        return sharpe


def main():
    """Test ATFT inference with sample data"""
    # モデル初期化
    logger.info("=== ATFT-GAT-FAN Inference Test ===")
    atft = ATFTInference()

    # サンプルデータ作成
    logger.info("Creating sample data...")
    sample_size = 100
    seq_len = 20
    n_features = 10  # ATFT expects specific features

    # ランダムテストデータ
    sample_features = torch.randn(sample_size, seq_len, n_features)

    # 推論実行
    logger.info("Running inference...")
    results = atft.predict(sample_features, horizon=1)

    # 結果表示
    logger.info("=== Results ===")
    logger.info(f"Predictions shape: {results['predictions'].shape}")
    logger.info(f"Predictions mean: {results['predictions'].mean():.6f}")
    logger.info(f"Predictions std: {results['predictions'].std():.6f}")

    if "confidence" in results:
        logger.info(f"Confidence mean: {results['confidence'].mean():.6f}")

    logger.info("✅ Inference test completed successfully!")


if __name__ == "__main__":
    main()
