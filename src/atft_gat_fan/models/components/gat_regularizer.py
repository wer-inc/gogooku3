"""
P0-3: GAT Attention Entropy Regularization
注意の尖り/平坦化を抑制
"""
import torch


def attn_entropy_penalty(alpha_list, coef: float = 1e-4):
    """
    Attention entropy regularization to prevent over-sharpening or flattening.

    alpha_list: GATv2Conv が吐く attention weights（必要ならフックで取得）
    ここでは簡易に最後の層の α を想定し、-Σ p log p を最大化（=最小化に負号）
    取り回し簡素化のため、alpha_list が None の時は 0 を返す。

    Args:
        alpha_list: List of attention weight tensors
        coef: Regularization coefficient

    Returns:
        Entropy penalty term
    """
    if not alpha_list:
        return torch.tensor(0.0)

    loss = 0.0
    for a in alpha_list:
        p = a.clamp_min(1e-12)
        loss = loss - (p * p.log()).mean()

    return coef * loss
