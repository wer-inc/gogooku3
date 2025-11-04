from __future__ import annotations

import torch
from torch import nn


class ListNetLoss(nn.Module):
    """ListNet loss with optional Top-K reweighting."""

    def __init__(self, tau: float = 0.5, topk: int | None = None, eps: float = 1e-12) -> None:
        super().__init__()
        self.tau = tau
        self.topk = topk
        self.eps = eps

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor | None:
        # GRADIENT-SAFE: Return None instead of requires_grad=False tensor
        if scores.numel() <= 1:
            return None

        if torch.std(labels) < 1e-8:
            return None

        p = torch.softmax(scores / self.tau, dim=0)
        q = torch.softmax(labels / self.tau, dim=0)

        if self.topk is not None and self.topk < scores.numel():
            _, top_idx = torch.topk(q, self.topk, largest=True, sorted=False)
            mask = torch.zeros_like(q)
            mask[top_idx] = 1.0
            mask = mask / (mask.sum() + self.eps)
            loss = -(mask * torch.log(p + self.eps)).sum()
        else:
            loss = -(q * torch.log(p + self.eps)).sum()

        return loss


class RankNetLoss(nn.Module):
    """Pairwise RankNet loss with optional negative sampling."""

    def __init__(self, neg_sample: int | None = None) -> None:
        super().__init__()
        self.neg_sample = neg_sample

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor | None:
        n = scores.numel()
        # GRADIENT-SAFE: Return None instead of requires_grad=False tensor
        if n <= 1:
            return None

        if torch.std(labels) < 1e-8:
            return None

        if self.neg_sample is None or n * (n - 1) // 2 <= self.neg_sample:
            idx_i, idx_j = torch.triu_indices(n, n, offset=1, device=scores.device)
        else:
            idx_i = torch.randint(0, n, (self.neg_sample,), device=scores.device)
            idx_j = torch.randint(0, n, (self.neg_sample,), device=scores.device)
            mask = idx_i != idx_j
            idx_i, idx_j = idx_i[mask], idx_j[mask]
            if idx_i.numel() == 0:
                return None

        s_diff = scores[idx_i] - scores[idx_j]
        y_diff = labels[idx_i] - labels[idx_j]
        target = (y_diff > 0).float()
        return nn.functional.binary_cross_entropy_with_logits(s_diff, target)


class CompositeLoss(nn.Module):
    """Weighted combination of ListNet / RankNet / MSE losses."""

    def __init__(
        self,
        *,
        listnet_weight: float = 1.0,
        ranknet_weight: float = 0.5,
        mse_weight: float = 0.1,
        listnet_tau: float = 0.5,
        listnet_topk: int | None = None,
        ranknet_neg_sample: int | None = None,
    ) -> None:
        super().__init__()
        self.listnet_weight = listnet_weight
        self.ranknet_weight = ranknet_weight
        self.mse_weight = mse_weight

        self.listnet = ListNetLoss(tau=listnet_tau, topk=listnet_topk)
        self.ranknet = RankNetLoss(neg_sample=ranknet_neg_sample)
        self.mse = nn.MSELoss()

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor | None:
        # GRADIENT-SAFE: Collect loss terms as tensors, then stack
        # Avoids leaf tensor in-place update (torch.tensor(0.0) + ...)
        loss_terms: list[torch.Tensor] = []

        if self.listnet_weight > 0:
            l_listnet = self.listnet(scores, labels)
            if l_listnet is not None:
                loss_terms.append(self.listnet_weight * l_listnet)

        if self.ranknet_weight > 0:
            l_ranknet = self.ranknet(scores, labels)
            if l_ranknet is not None:
                loss_terms.append(self.ranknet_weight * l_ranknet)

        if self.mse_weight > 0:
            l_mse = self.mse(scores, labels)
            loss_terms.append(self.mse_weight * l_mse)

        # Empty aggregation → return None (caller should skip)
        if not loss_terms:
            return None

        # Stack and reduce → maintains gradient graph
        return torch.stack(loss_terms).sum()
