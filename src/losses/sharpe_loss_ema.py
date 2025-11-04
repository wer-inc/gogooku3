"""
P0-7: Sharpe Loss with EMA Smoothing

バッチノイズを抑制したEMA平滑化Sharpe Ratio Loss。

Key improvements:
- decay 0.9 → 0.95 (よりスムーズ)
- 初期化の安定化
- warm-up期間の追加

Usage:
    from src.losses.sharpe_loss_ema import SharpeLossEMA

    sharpe_loss = SharpeLossEMA(decay=0.95, eps=1e-6)
    loss = sharpe_loss(predictions, targets)
"""
import torch
import torch.nn as nn


class SharpeLossEMA(nn.Module):
    """
    Sharpe Ratio Loss with Exponential Moving Average

    Args:
        decay: EMA decay rate (推奨: 0.92-0.95, default: 0.95)
        eps: 数値安定性定数 (default: 1e-6)
        risk_free_rate: リスクフリーレート (年率, default: 0.0)
        annualization_factor: 年換算係数 (default: 252)
        warmup_steps: Warm-up期間 (default: 10)

    P0-7 改善点:
    - decay 0.9 → 0.95: バッチノイズ抑制
    - warm-up: 初期不安定性の緩和
    - eps安定化: 分母ゼロ対策強化
    """
    def __init__(
        self,
        decay: float = 0.95,  # P0-7: 0.9 → 0.95
        eps: float = 1e-6,
        risk_free_rate: float = 0.0,
        annualization_factor: float = 252.0,
        warmup_steps: int = 10
    ):
        super().__init__()
        self.decay = decay
        self.eps = eps
        self.risk_free_rate = risk_free_rate / annualization_factor  # Daily
        self.annualization_factor = annualization_factor
        self.warmup_steps = warmup_steps

        # EMA state variables
        self.register_buffer("ema_mean", torch.tensor(0.0))
        self.register_buffer("ema_var", torch.tensor(1.0))
        self.register_buffer("step_count", torch.tensor(0))

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: 予測値 [B, H] or [B, H, 1]
            targets: 真値 [B, H] or [B, H, 1]

        Returns:
            loss: -Sharpe ratio (minimization)
        """
        # Shape normalization
        if predictions.dim() == 3:
            predictions = predictions.squeeze(-1)
        if targets.dim() == 3:
            targets = targets.squeeze(-1)

        # Returns computation
        returns = predictions - targets  # 簡易版（実際にはpredictionsが収益率予測）
        # または: returns = predictions（もしpredictionsが既に収益率なら）

        # Batch statistics
        batch_mean = returns.mean()
        batch_var = returns.var(unbiased=False) + self.eps

        # EMA update
        if self.training:
            with torch.no_grad():
                self.step_count += 1

                # Warm-up: 初期はバッチ統計を直接使用
                if self.step_count <= self.warmup_steps:
                    warmup_weight = self.step_count.float() / self.warmup_steps
                    self.ema_mean = warmup_weight * batch_mean + (1 - warmup_weight) * self.ema_mean
                    self.ema_var = warmup_weight * batch_var + (1 - warmup_weight) * self.ema_var
                else:
                    # Normal EMA update
                    self.ema_mean = self.decay * self.ema_mean + (1 - self.decay) * batch_mean
                    self.ema_var = self.decay * self.ema_var + (1 - self.decay) * batch_var

        # Sharpe ratio computation
        mean = self.ema_mean if self.training else batch_mean
        std = torch.sqrt(self.ema_var) if self.training else torch.sqrt(batch_var)

        # Excess return
        excess_return = mean - self.risk_free_rate

        # Sharpe ratio (annualized)
        sharpe = excess_return / (std + self.eps) * torch.sqrt(torch.tensor(self.annualization_factor))

        # Loss: negative Sharpe (maximization → minimization)
        loss = -sharpe

        return loss

    def get_sharpe_ratio(self) -> float:
        """
        現在のSharpe ratioを取得（診断用）

        Returns:
            float: Annualized Sharpe ratio
        """
        std = torch.sqrt(self.ema_var)
        excess_return = self.ema_mean - self.risk_free_rate
        sharpe = excess_return / (std + self.eps) * torch.sqrt(torch.tensor(self.annualization_factor))

        return float(sharpe.item())

    def reset_ema(self):
        """
        EMA状態をリセット（epoch間やvalidation時）
        """
        self.ema_mean.zero_()
        self.ema_var.fill_(1.0)
        self.step_count.zero_()
