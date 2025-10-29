"""Adaptive normalization components (FAN and SAN)."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class FrequencyAdaptiveNorm(nn.Module):
    """Frequency Adaptive Normalization (FAN).

    Multi-scale正規化を学習済み重みで統合し、時系列のスケール変化を吸収する。
    """

    def __init__(
        self,
        num_features: int,
        window_sizes: Iterable[int] = (5, 10, 20),
        aggregation: str = "weighted_mean",
        learn_weights: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__()
        window_sizes = list(window_sizes)
        if len(window_sizes) == 0:
            raise ValueError("window_sizes must contain at least one element.")

        self.num_features = int(num_features)
        self.window_sizes = window_sizes
        self.aggregation = aggregation
        self.eps = float(eps)

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(self.num_features, eps=self.eps) for _ in window_sizes]
        )

        if aggregation not in {"weighted_mean", "mean", "max"}:
            raise ValueError(f"Invalid aggregation '{aggregation}'.")

        if learn_weights and aggregation == "weighted_mean":
            self.weights = nn.Parameter(torch.ones(len(window_sizes)))
        else:
            self.register_buffer("weights", torch.ones(len(window_sizes)))
        self._last_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, features)
        """
        if x.dim() != 3:
            raise ValueError(f"FAN expects 3D input (B, L, F); got {x.shape}")

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        batch_size, seq_len, _ = x.shape

        normalized_outputs = []
        for i, window_size in enumerate(self.window_sizes):
            if seq_len >= window_size:
                # (B, L, F) -> unfold -> (B, L-window+1, window, F)
                unfolded = x.unfold(dimension=1, size=window_size, step=1)
                unfolded = unfolded.permute(0, 1, 3, 2)  # (B, L-window+1, window, F)
                mean = unfolded.mean(dim=2, keepdim=True)
                var = unfolded.var(dim=2, unbiased=False, keepdim=True)
                std = torch.sqrt(var + self.eps)
                normalized = (unfolded - mean) / std

                center_idx = window_size // 2
                normalized = normalized[:, :, center_idx, :]

                pad_left = center_idx
                pad_right = seq_len - normalized.shape[1] - pad_left
                normalized = F.pad(normalized, (0, 0, pad_left, pad_right))
                normalized_outputs.append(normalized)
            else:
                normalized_outputs.append(self.layer_norms[i](x))

        if self.aggregation == "weighted_mean":
            weights = F.softmax(self.weights, dim=0)
            self._last_weights = weights.detach()
            output = sum(
                weight * normalized
                for weight, normalized in zip(weights, normalized_outputs, strict=False)
            )
        elif self.aggregation == "mean":
            output = torch.stack(normalized_outputs, dim=0).mean(dim=0)
        else:  # "max"
            output = torch.stack(normalized_outputs, dim=0).max(dim=0)[0]

        return output


class SliceAdaptiveNorm(nn.Module):
    """Slice Adaptive Normalization (SAN).

    時系列窓を重なりありのスライスに分割し、スライスごとに正規化。
    """

    def __init__(
        self,
        num_features: int,
        num_slices: int = 4,
        overlap: float = 0.5,
        slice_aggregation: str = "learned",
        eps: float = 1e-5,
    ):
        super().__init__()
        if num_slices < 1:
            raise ValueError("num_slices must be >= 1.")
        if not (0.0 <= overlap < 1.0):
            raise ValueError("overlap must be in [0, 1).")

        self.num_features = int(num_features)
        self.num_slices = int(num_slices)
        self.overlap = float(overlap)
        self.slice_aggregation = slice_aggregation
        self.eps = float(eps)

        self.instance_norms = nn.ModuleList(
            [
                nn.InstanceNorm1d(self.num_features, affine=True, eps=self.eps)
                for _ in range(self.num_slices)
            ]
        )

        if slice_aggregation not in {"learned", "mean"}:
            raise ValueError(f"Invalid slice_aggregation '{slice_aggregation}'.")

        if slice_aggregation == "learned":
            self.slice_weights = nn.Linear(self.num_features, self.num_slices)
            nn.init.zeros_(self.slice_weights.bias)
        else:
            self.register_buffer("slice_weights", torch.ones(self.num_slices))
        self._last_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, features)
        """
        if x.dim() != 3:
            raise ValueError(f"SAN expects 3D input (B, L, F); got {x.shape}")

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        batch_size, seq_len, _ = x.shape

        slice_size = max(1, seq_len // self.num_slices)
        step = max(1, int(slice_size * (1 - self.overlap)))

        normalized_slices: list[tuple[torch.Tensor, int, int]] = []
        for i in range(self.num_slices):
            start = i * step
            end = min(start + slice_size, seq_len)
            if start >= seq_len:
                break
            slice_data = x[:, start:end, :]
            if slice_data.size(1) <= 1:
                normalized = F.layer_norm(slice_data, (slice_data.size(-1),), eps=self.eps)
            else:
                slice_data_t = slice_data.transpose(1, 2)
                normalized = self.instance_norms[i](slice_data_t).transpose(1, 2)
            normalized_slices.append((normalized, start, end))

        if not normalized_slices:
            return F.layer_norm(x, (self.num_features,), eps=self.eps)

        if self.slice_aggregation == "learned":
            pooled = x.mean(dim=1)
            weights = F.softmax(self.slice_weights(pooled), dim=1)  # (B, num_slices)
            self._last_weights = weights.detach()
            weights = weights.unsqueeze(1).unsqueeze(3)  # (B, 1, num_slices, 1)

            stacked = []
            for idx, (normalized, start, end) in enumerate(normalized_slices):
                padded = torch.zeros_like(x)
                padded[:, start:end, :] = normalized[:, : end - start, :]
                stacked.append(padded)
            stacked = torch.stack(stacked, dim=2)  # (B, L, num_slices, F)
            output = (stacked * weights).sum(dim=2)
        else:
            output = torch.zeros_like(x)
            norm_count = torch.zeros(batch_size, seq_len, 1, device=x.device)
            for normalized, start, end in normalized_slices:
                length = end - start
                output[:, start:end, :] += normalized[:, :length, :]
                norm_count[:, start:end, :] += 1
            output = torch.where(
                norm_count > 0, output / norm_count.clamp_min(1.0), x
            )

        return output
