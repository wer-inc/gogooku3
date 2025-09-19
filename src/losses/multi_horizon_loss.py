"""
Multi-Horizon Loss Functions for ATFT-GAT-FAN
Huber損失ベースの多ホライズン予測最適化
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
# config = get_settings()  # 遅延初期化に変更


class HuberLoss(nn.Module):
    """Huber損失（Robust L1/L2 loss）"""

    def __init__(self, delta: float = 0.01):
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 予測値 [batch, ...]
            target: 目標値 [batch, ...]

        Returns:
            Huber損失
        """
        diff = pred - target
        abs_diff = torch.abs(diff)

        # Huber関数: |x| <= δ の場合 L2、|x| > δ の場合 L1
        quadratic = 0.5 * (diff ** 2)
        linear = self.delta * (abs_diff - 0.5 * self.delta)

        loss = torch.where(abs_diff < self.delta, quadratic, linear)
        return loss.mean()


class QuantileLoss(nn.Module):
    """分位点損失（Quantile Loss）"""

    def __init__(self, quantiles: list[float] = [0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = torch.tensor(quantiles)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 予測分位点 [batch, n_quantiles]
            target: 目標値 [batch]

        Returns:
            分位点損失
        """
        target = target.unsqueeze(-1)  # [batch, 1]
        diff = target - pred  # [batch, n_quantiles]

        # 分位点損失: ρ_τ(u) = u * (τ - I(u < 0))
        quantile_loss = torch.maximum(
            self.quantiles * diff,
            (self.quantiles - 1) * diff
        )

        return quantile_loss.mean()


class CoveragePenalty(nn.Module):
    """カバレッジ正則化（Coverage Regularization）"""

    def __init__(self, target_quantiles: list[float] = [0.1, 0.5, 0.9], alpha: float = 0.01):
        super().__init__()
        self.target_quantiles = torch.tensor(target_quantiles)
        self.alpha = alpha

    def forward(self, pred_quantiles: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_quantiles: 予測分位点 [batch, n_quantiles]
            target: 目標値 [batch]

        Returns:
            カバレッジ誤差のペナルティ
        """
        target = target.unsqueeze(-1)  # [batch, 1]

        # カバレッジ計算: P(real <= predicted_quantile)
        coverage = (target <= pred_quantiles).float().mean(dim=0)  # [n_quantiles]

        # 目標カバレッジからの誤差
        coverage_error = coverage - self.target_quantiles.to(coverage.device)

        # L2ペナルティ
        penalty = self.alpha * torch.sum(coverage_error ** 2)

        return penalty


class MultiHorizonLoss(nn.Module):
    """
    多ホライズン損失（Multi-Horizon Loss）

    特徴:
    - 短期重視の重み付け
    - Huber損失ベース
    - カバレッジ正則化オプション
    """

    def __init__(
        self,
        horizons: list[int] = [1, 2, 3, 5, 10],
        weights: dict[int, float] | None = None,
        huber_delta: float = 0.01,
        use_coverage_penalty: bool = False,
        coverage_alpha: float = 0.01,
        quantiles: list[float] | None = None,
        **kwargs
    ):
        super().__init__()

        self.horizons = horizons
        self.huber_delta = huber_delta
        self.use_coverage_penalty = use_coverage_penalty
        self.coverage_alpha = coverage_alpha

        # デフォルト重み: 短期重視
        if weights is None:
            weights = {1: 1.0, 2: 0.8, 3: 0.7, 5: 0.5, 10: 0.3}
        self.weights = {h: weights.get(h, 1.0) for h in horizons}

        # Huber損失
        self.huber_loss = HuberLoss(delta=huber_delta)

        # 分位点関連
        if quantiles:
            self.quantile_loss = QuantileLoss(quantiles)
            self.coverage_penalty = CoveragePenalty(quantiles, coverage_alpha)
        else:
            self.quantile_loss = None
            self.coverage_penalty = None

        logger.info(f"MultiHorizonLoss initialized with horizons: {horizons}")
        logger.info(f"Weights: {self.weights}")
        logger.info(f"Huber delta: {huber_delta}")

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Args:
            predictions: 予測値 {'point_horizon_1': tensor, 'point_horizon_5': tensor, ...}
            targets: 目標値 {'horizon_1': tensor, 'horizon_5': tensor, ...}
            return_components: 各ホライズンの損失を返す

        Returns:
            総合損失または損失コンポーネント
        """
        # デバイスを取得（空の場合はCPU）
        if predictions:
            device = next(iter(predictions.values())).device
        elif targets:
            device = next(iter(targets.values())).device
        else:
            device = torch.device('cpu')

        losses = []  # 勾配を保持するためリストに収集
        loss_components = {}

        for horizon in self.horizons:
            # 予測/正解のキー候補を列挙し、存在するものを利用
            pred_candidates = [
                f'point_horizon_{horizon}',
                f'horizon_{horizon}',
                f'horizon_{horizon}d',
                f'h{horizon}',
            ]
            target_candidates = [
                f'horizon_{horizon}',
                f'horizon_{horizon}d',
                f'point_horizon_{horizon}',
                f'h{horizon}',
            ]

            pred_key = next((k for k in pred_candidates if k in predictions), None)
            target_key = next((k for k in target_candidates if k in targets), None)

            if pred_key is None or target_key is None:
                continue

            pred = predictions[pred_key]
            target = targets[target_key]
            weight = self.weights[horizon]

            # Huber損失
            huber_loss = self.huber_loss(pred, target)
            weighted_loss = weight * huber_loss

            losses.append(weighted_loss)  # リストに追加
            loss_components[f'huber_{horizon}'] = huber_loss.item()
            loss_components[f'weighted_huber_{horizon}'] = weighted_loss.item()

        # 分位点損失（オプション）
        if self.quantile_loss is not None:
            for horizon in self.horizons:
                # 実際のキー名に合わせる
                q_key = f'quantile_horizon_{horizon}'
                target_key = f'horizon_{horizon}'

                # 古いキー名もサポート
                if q_key not in predictions and f'h{horizon}_quantiles' in predictions:
                    q_key = f'h{horizon}_quantiles'
                if target_key not in targets and f'h{horizon}' in targets:
                    target_key = f'h{horizon}'

                if q_key in predictions and target_key in targets:
                    pred_q = predictions[q_key]
                    target = targets[target_key]

                    q_loss = self.quantile_loss(pred_q, target)
                    losses.append(q_loss)  # リストに追加

                    loss_components[f'quantile_{horizon}'] = q_loss.item()

                    # カバレッジペナルティ
                    if self.use_coverage_penalty and self.coverage_penalty is not None:
                        cov_penalty = self.coverage_penalty(pred_q, target)
                        losses.append(cov_penalty)  # リストに追加
                        loss_components[f'coverage_penalty_{horizon}'] = cov_penalty.item()

        # 勾配を保持したまま合計を計算
        if losses:
            total_loss = torch.stack(losses).sum()
        else:
            # 損失がない場合のフォールバック
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        loss_components['total_loss'] = total_loss.item()

        if return_components:
            return total_loss, loss_components
        return total_loss


class RankICLoss(nn.Module):
    """RankIC最大化損失（ランキング相関）"""

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 予測値 [batch]
            target: 目標値 [batch]

        Returns:
            RankIC損失（負の相関係数）
        """
        # Spearman相関係数の近似計算
        pred_rank = pred.argsort().argsort().float()
        target_rank = target.argsort().argsort().float()

        # 相関係数
        pred_mean = pred_rank.mean()
        target_mean = target_rank.mean()

        pred_std = pred_rank.std()
        target_std = target_rank.std()

        if pred_std > 0 and target_std > 0:
            correlation = ((pred_rank - pred_mean) * (target_rank - target_mean)).mean() / (pred_std * target_std)
        else:
            correlation = torch.tensor(0.0)

        # RankICを最大化するための負の損失
        return -self.weight * correlation


class SharpeLoss(nn.Module):
    """Sharpe比最大化損失"""

    def __init__(self, weight: float = 0.1, min_periods: int = 20):
        super().__init__()
        self.weight = weight
        self.min_periods = min_periods

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 予測値 [batch]
            target: 目標値 [batch]

        Returns:
            Sharpe損失（負のSharpe比）
        """
        if len(pred) < self.min_periods:
            return torch.tensor(0.0)

        # リターン計算（簡易）
        returns = target

        # Sharpe比 = mean(return) / std(return)
        if returns.std() > 0:
            sharpe = returns.mean() / returns.std()
        else:
            sharpe = torch.tensor(0.0)

        # Sharpe比を最大化するための負の損失
        return -self.weight * sharpe


class ComprehensiveLoss(nn.Module):
    """
    包括的損失関数
    多ホライズン + RankIC + Sharpe + 正則化項
    """

    def __init__(
        self,
        horizons: list[int] = [1, 2, 3, 5, 10],
        horizon_weights: dict[int, float] | None = None,
        huber_delta: float = 0.01,
        rankic_weight: float = 0.1,
        sharpe_weight: float = 0.05,
        l2_lambda: float = 1e-5,
        **kwargs
    ):
        super().__init__()

        # 多ホライズン損失
        self.multi_horizon_loss = MultiHorizonLoss(
            horizons=horizons,
            weights=horizon_weights,
            huber_delta=huber_delta,
            **kwargs
        )

        # 補助損失
        self.rankic_loss = RankICLoss(weight=rankic_weight) if rankic_weight > 0 else None
        self.sharpe_loss = SharpeLoss(weight=sharpe_weight) if sharpe_weight > 0 else None

        # 正則化
        self.l2_lambda = l2_lambda

        logger.info(f"ComprehensiveLoss initialized with {len(horizons)} horizons")
        logger.info(f"Huber delta: {huber_delta}, RankIC weight: {rankic_weight}, Sharpe weight: {sharpe_weight}")

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        model: nn.Module | None = None,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Args:
            predictions: 予測値
            targets: 目標値
            model: モデル（L2正則化用）
            return_components: 各損失コンポーネントを返す

        Returns:
            総合損失または損失コンポーネント
        """
        total_loss = 0.0
        loss_components = {}

        # 多ホライズン損失
        mh_loss, mh_components = self.multi_horizon_loss(predictions, targets, return_components=True)
        total_loss += mh_loss
        loss_components.update(mh_components)

        # RankIC損失（h1のみ）
        if self.rankic_loss is not None and 'h1' in predictions and 'h1' in targets:
            ric_loss = self.rankic_loss(predictions['h1'], targets['h1'])
            total_loss += ric_loss
            loss_components['rankic'] = ric_loss.item()

        # Sharpe損失（h1のみ）
        if self.sharpe_loss is not None and 'h1' in predictions and 'h1' in targets:
            sharpe_loss = self.sharpe_loss(predictions['h1'], targets['h1'])
            total_loss += sharpe_loss
            loss_components['sharpe'] = sharpe_loss.item()

        # L2正則化
        if self.l2_lambda > 0 and model is not None:
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters() if p.requires_grad)
            l2_loss = self.l2_lambda * l2_norm
            total_loss += l2_loss
            loss_components['l2_regularization'] = l2_loss.item()

        loss_components['total_loss'] = total_loss.item()

        if return_components:
            return total_loss, loss_components
        return total_loss


# 後方互換性のためのエイリアス
HuberMultiHorizonLoss = MultiHorizonLoss
