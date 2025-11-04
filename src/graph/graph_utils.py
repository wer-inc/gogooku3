"""
P0-3: Graph Utilities for Edge Attribute Processing
edge_attr標準化、エッジDropout
"""
import torch


def standardize_edge_attr(edge_attr: torch.Tensor) -> torch.Tensor:
    """
    edge_attr: [E, Fe] を列ごとに標準化（均一スケールに）
    例: [corr, same_market, same_sector, delta_corr]

    Args:
        edge_attr: Edge attributes tensor [E, Fe]

    Returns:
        Standardized edge attributes [E, Fe]
    """
    if edge_attr.numel() == 0:
        return edge_attr

    mean = edge_attr.mean(dim=0, keepdim=True)
    std = edge_attr.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (edge_attr - mean) / std


def apply_edge_dropout(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    p: float,
    training: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    エッジDropout（訓練時のみ）

    Args:
        edge_index: [2, E] edge indices
        edge_attr: [E, Fe] edge attributes
        p: dropout probability
        training: training mode flag

    Returns:
        (edge_index, edge_attr) with dropout applied
    """
    if (not training) or p <= 0 or edge_index.numel() == 0:
        return edge_index, edge_attr

    E = edge_index.size(1)
    keep = torch.rand(E, device=edge_index.device) > p

    # 全滅回避
    if keep.sum() == 0:
        keep[torch.randint(0, E, (1,), device=edge_index.device)] = True

    return edge_index[:, keep], edge_attr[keep]
