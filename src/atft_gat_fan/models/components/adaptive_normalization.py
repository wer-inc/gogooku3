"""Adaptive normalization components (FAN and SAN)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class FrequencyAdaptiveNorm(nn.Module):
    """Frequency Adaptive Normalization (FAN)."""

    def __init__(
        self,
        num_features: int,
        window_sizes: list = [5, 10, 20],
        aggregation: str = "weighted_mean",
        learn_weights: bool = True,
    ):
        """Initialize FAN.

        Args:
            num_features: Number of input features
            window_sizes: List of window sizes for multi-scale normalization
            aggregation: How to aggregate multi-scale features
            learn_weights: Whether to learn aggregation weights
        """
        super().__init__()
        self.num_features = num_features
        self.window_sizes = window_sizes
        self.aggregation = aggregation

        # Layer normalization for each window size
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(num_features) for _ in window_sizes]
        )

        # Learnable aggregation weights
        if learn_weights and aggregation == "weighted_mean":
            self.weights = nn.Parameter(torch.ones(len(window_sizes)))
        else:
            self.register_buffer("weights", torch.ones(len(window_sizes)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, features)

        Returns:
            Normalized tensor
        """
        batch_size, seq_len, num_features = x.shape
        normalized_outputs = []

        for i, window_size in enumerate(self.window_sizes):
            if seq_len >= window_size:
                # Apply layer norm with different receptive fields
                # Reshape to apply norm over windows
                unfolded = x.unfold(1, window_size, 1)  # (B, L-W+1, F, W)
                unfolded = rearrange(unfolded, "b l f w -> (b l) w f")

                # Apply layer norm
                normalized = self.layer_norms[i](unfolded)
                normalized = rearrange(normalized, "(b l) w f -> b l w f", b=batch_size)

                # Take the center value for each window
                center_idx = window_size // 2
                normalized = normalized[:, :, center_idx, :]

                # Pad to match original sequence length
                pad_left = center_idx
                pad_right = seq_len - normalized.shape[1] - pad_left
                normalized = F.pad(normalized, (0, 0, pad_left, pad_right))

                normalized_outputs.append(normalized)
            else:
                # If sequence is shorter than window, use global norm
                normalized = self.layer_norms[i](x)
                normalized_outputs.append(normalized)

        # Aggregate multi-scale normalized features
        if self.aggregation == "weighted_mean":
            weights = F.softmax(self.weights, dim=0)
            output = sum(w * out for w, out in zip(weights, normalized_outputs))
        elif self.aggregation == "mean":
            output = torch.stack(normalized_outputs).mean(dim=0)
        else:
            output = torch.stack(normalized_outputs).max(dim=0)[0]

        return output


class SliceAdaptiveNorm(nn.Module):
    """Slice Adaptive Normalization (SAN)."""

    def __init__(
        self,
        num_features: int,
        num_slices: int = 4,
        overlap: float = 0.5,
        slice_aggregation: str = "learned",
    ):
        """Initialize SAN.

        Args:
            num_features: Number of input features
            num_slices: Number of slices to divide the sequence
            overlap: Overlap ratio between slices
            slice_aggregation: How to aggregate slice features
        """
        super().__init__()
        self.num_features = num_features
        self.num_slices = num_slices
        self.overlap = overlap
        self.slice_aggregation = slice_aggregation

        # Instance normalization for each slice
        self.instance_norms = nn.ModuleList(
            [nn.InstanceNorm1d(num_features, affine=True) for _ in range(num_slices)]
        )

        # Learned aggregation
        if slice_aggregation == "learned":
            self.slice_weights = nn.Linear(num_features, num_slices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, features)

        Returns:
            Normalized tensor
        """
        batch_size, seq_len, num_features = x.shape

        # Calculate slice size and step
        slice_size = max(1, seq_len // self.num_slices)
        step = max(1, int(slice_size * (1 - self.overlap)))

        normalized_slices = []
        slice_masks = []

        for i in range(self.num_slices):
            start = i * step
            end = min(start + slice_size, seq_len)

            if start < seq_len:
                # Extract slice
                slice_data = x[:, start:end, :]

                # Apply instance norm (need to transpose for 1D norm)
                slice_data = slice_data.transpose(1, 2)  # (B, F, L)
                normalized = self.instance_norms[i](slice_data)
                normalized = normalized.transpose(1, 2)  # (B, L, F)

                # Create mask for this slice
                mask = torch.zeros(batch_size, seq_len, 1, device=x.device)
                mask[:, start:end, :] = 1.0

                normalized_slices.append(normalized)
                slice_masks.append(mask)

        # Aggregate slices
        if self.slice_aggregation == "learned":
            # Compute slice weights based on input features
            weights = F.softmax(
                self.slice_weights(x.mean(dim=1)), dim=1
            )  # (B, num_slices)
            weights = weights.unsqueeze(1).unsqueeze(3)  # (B, 1, num_slices, 1)

            # Pad slices to full sequence length and stack
            padded_slices = []
            for i, (normalized, mask) in enumerate(zip(normalized_slices, slice_masks)):
                padded = torch.zeros_like(x)
                start = i * step
                end = min(start + slice_size, seq_len)
                padded[:, start:end, :] = normalized
                padded_slices.append(padded)

            stacked = torch.stack(padded_slices, dim=2)  # (B, L, num_slices, F)
            output = (stacked * weights).sum(dim=2)

        else:
            # Simple averaging with overlap handling
            output = torch.zeros_like(x)
            norm_count = torch.zeros(batch_size, seq_len, 1, device=x.device)

            for i, (normalized, mask) in enumerate(zip(normalized_slices, slice_masks)):
                start = i * step
                end = min(start + slice_size, seq_len)
                output[:, start:end, :] += normalized
                norm_count[:, start:end, :] += 1

            # Average where we have overlaps, but keep original where no slices
            mask = norm_count > 0
            output = torch.where(mask, output / norm_count, x)

        return output
