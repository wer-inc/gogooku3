"""
P0-6: Quantile Crossing Penalty

分位点予測の単調性制約を強制するペナルティ。
y(q_i) <= y(q_{i+1}) を満たさない箇所にペナルティ。

Usage:
    from src.losses.quantile_crossing import quantile_crossing_penalty

    loss = base_loss + quantile_crossing_penalty(y_quantiles, lambda_qc=1e-3)
"""
import torch
import torch.nn as nn


def quantile_crossing_penalty(
    yhat_q: torch.Tensor,
    lambda_qc: float = 1e-3
) -> torch.Tensor:
    """
    Quantile Crossing Penalty (P0-6)

    分位点予測が単調増加でない箇所にペナルティ。

    Args:
        yhat_q: 分位点予測 [B, H, Q] (Q: number of quantiles)
        lambda_qc: ペナルティ係数 (推奨: 1e-3 ~ 5e-3)

    Returns:
        penalty: スカラーペナルティ (lower is better)

    Example:
        >>> y_q = model(x)  # [64, 20, 5]
        >>> penalty = quantile_crossing_penalty(y_q, lambda_qc=1e-3)
        >>> loss = base_loss + penalty
    """
    if yhat_q.size(-1) < 2:
        # 分位点が1つしかない場合はペナルティなし
        return torch.tensor(0.0, device=yhat_q.device, dtype=yhat_q.dtype)

    # 隣接分位点の差: y(q_{i+1}) - y(q_i)
    diffs = yhat_q[:, :, 1:] - yhat_q[:, :, :-1]

    # 違反（負の差）をペナルティ
    # ReLU(-diffs) = max(0, -diffs) = max(0, y(q_i) - y(q_{i+1}))
    violations = torch.relu(-diffs)

    # 平均ペナルティ
    penalty = lambda_qc * violations.mean()

    return penalty


class QuantileCrossingLoss(nn.Module):
    """
    Quantile Crossing Loss Module (P0-6)

    分位点交差ペナルティのモジュール版。
    既存のloss functionと組み合わせて使用。

    Args:
        lambda_qc: ペナルティ係数 (default: 1e-3)

    Example:
        >>> qc_loss = QuantileCrossingLoss(lambda_qc=1e-3)
        >>> penalty = qc_loss(y_quantiles)
    """
    def __init__(self, lambda_qc: float = 1e-3):
        super().__init__()
        self.lambda_qc = lambda_qc

    def forward(self, yhat_q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            yhat_q: 分位点予測 [B, H, Q]

        Returns:
            penalty: スカラーペナルティ
        """
        return quantile_crossing_penalty(yhat_q, self.lambda_qc)


def quantile_crossing_rate(yhat_q: torch.Tensor) -> float:
    """
    分位点交差率の計算（診断用）

    Args:
        yhat_q: 分位点予測 [B, H, Q]

    Returns:
        rate: 交差発生率 (0.0-1.0, target < 0.05)
    """
    if yhat_q.size(-1) < 2:
        return 0.0

    diffs = yhat_q[:, :, 1:] - yhat_q[:, :, :-1]
    violations = (diffs < 0).float().mean().item()

    return float(violations)
