#!/usr/bin/env python3
"""
一気に性能を上げるための改善スクリプト
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ImprovedMultiHorizonLoss(nn.Module):
    """改善された損失関数"""

    def __init__(self, horizons: list[int], weights: dict[str, float] = None):
        super().__init__()
        self.horizons = horizons

        # デフォルトの重み
        default_weights = {
            "huber": 0.4,      # 減らす
            "rank_ic": 0.3,    # 増やす
            "sharpe": 0.2,     # 増やす
            "mae": 0.1
        }
        self.weights = weights or default_weights

    def forward(self, predictions: dict[str, torch.Tensor],
                targets: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        """
        改善された損失計算
        - ターゲットの正規化
        - RankICとSharpe比の重視
        """
        total_loss = 0.0
        loss_components = {}

        for h in self.horizons:
            pred_key = f"point_horizon_{h}"
            targ_key = f"horizon_{h}"

            if pred_key not in predictions or targ_key not in targets:
                continue

            pred = predictions[pred_key]
            targ = targets[targ_key]

            # ターゲットの正規化（重要！）
            targ_mean = targ.mean()
            targ_std = targ.std() + 1e-8
            targ_norm = (targ - targ_mean) / targ_std

            # Huber損失（外れ値に強い）
            huber = nn.functional.huber_loss(pred, targ_norm, delta=1.0)

            # Rank IC損失（順位相関を最大化）
            rank_ic_loss = -self.compute_rank_ic(pred, targ_norm)

            # Sharpe損失（リスク調整リターンを最大化）
            sharpe_loss = -self.compute_sharpe(pred, targ_norm)

            # MAE（絶対誤差）
            mae = torch.abs(pred - targ_norm).mean()

            # 加重和
            horizon_loss = (
                self.weights["huber"] * huber +
                self.weights["rank_ic"] * rank_ic_loss +
                self.weights["sharpe"] * sharpe_loss +
                self.weights["mae"] * mae
            )

            total_loss = total_loss + horizon_loss

            loss_components[f"h{h}_huber"] = huber.item()
            loss_components[f"h{h}_rank_ic"] = -rank_ic_loss.item()
            loss_components[f"h{h}_sharpe"] = -sharpe_loss.item()
            loss_components[f"h{h}_mae"] = mae.item()

        return total_loss / len(self.horizons), loss_components

    def compute_rank_ic(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        """Rank IC (順位相関)"""
        # ランクに変換
        pred_rank = pred.argsort().argsort().float()
        targ_rank = targ.argsort().argsort().float()

        # 相関係数
        pred_mean = pred_rank.mean()
        targ_mean = targ_rank.mean()

        cov = ((pred_rank - pred_mean) * (targ_rank - targ_mean)).mean()
        pred_std = (pred_rank - pred_mean).std() + 1e-8
        targ_std = (targ_rank - targ_mean).std() + 1e-8

        return cov / (pred_std * targ_std)

    def compute_sharpe(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        """Sharpe比（符号反転版）"""
        # ポジション決定（符号反転）
        positions = -torch.sign(pred)

        # ポートフォリオリターン
        returns = positions * targ

        # Sharpe比
        mean_return = returns.mean()
        std_return = returns.std() + 1e-8

        return mean_return / std_return


class EnhancedFeatureEngineering:
    """特徴量エンジニアリングの強化"""

    @staticmethod
    def add_interaction_features(features: torch.Tensor) -> torch.Tensor:
        """相互作用特徴量の追加"""
        # 上位特徴量の相互作用
        # 例: momentum × volatility
        # 例: volume × price_change
        # 実装は既存の特徴量に依存
        return features

    @staticmethod
    def add_market_regime_features(features: torch.Tensor,
                                  market_data: dict) -> torch.Tensor:
        """市場レジーム特徴量"""
        # ボラティリティレジーム
        # トレンドレジーム
        # 流動性レジーム
        return features


def quick_improvements_config():
    """即効性のある改善設定"""
    return {
        # 1. 損失関数の改善
        "loss": {
            "weights": {
                "huber": 0.3,
                "rank_ic": 0.35,
                "sharpe": 0.25,
                "mae": 0.1
            }
        },

        # 2. 学習率スケジュール
        "scheduler": {
            "type": "cosine_annealing_warm_restarts",
            "T_0": 10,
            "T_mult": 2,
            "eta_min": 1e-6
        },

        # 3. データ拡張
        "augmentation": {
            "noise_level": 0.01,
            "dropout_rate": 0.1,
            "mixup_alpha": 0.2
        },

        # 4. アンサンブル
        "ensemble": {
            "n_models": 3,
            "weight_decay_diff": 0.1,
            "dropout_diff": 0.05
        }
    }


def main():
    """メイン実行"""
    config = quick_improvements_config()
    logger.info("Quick improvements configuration:")
    logger.info(config)

    # 改善された損失関数のテスト
    loss_fn = ImprovedMultiHorizonLoss(
        horizons=[1, 5, 10, 20],
        weights=config["loss"]["weights"]
    )

    # ダミーデータでテスト
    batch_size = 32
    predictions = {
        f"point_horizon_{h}": torch.randn(batch_size)
        for h in [1, 5, 10, 20]
    }
    targets = {
        f"horizon_{h}": torch.randn(batch_size) * 0.01  # 実際のリターンスケール
        for h in [1, 5, 10, 20]
    }

    loss, components = loss_fn(predictions, targets)
    logger.info(f"Test loss: {loss.item():.4f}")
    logger.info(f"Components: {components}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
