from __future__ import annotations

import math

import torch
from torch import nn


class TransformerBlock(nn.Module):
    """Standard Transformer block with Pre-LN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        residual = tokens
        tokens = self.ln1(tokens)
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = residual + attn_out

        residual = tokens
        tokens = self.ln2(tokens)
        tokens = residual + self.ffn(tokens)
        return tokens


class PatchEmbedding(nn.Module):
    """Patch embedding layer with optional channel independence."""

    def __init__(
        self,
        in_feats: int,
        d_model: int,
        patch_len: int,
        stride: int,
        channel_independent: bool,
        patch_multiplier: int,
    ) -> None:
        super().__init__()
        self.channel_independent = channel_independent

        if channel_independent:
            self.conv = nn.Conv1d(
                in_feats,
                in_feats * patch_multiplier,
                kernel_size=patch_len,
                stride=stride,
                groups=in_feats,
                bias=False,
            )
            self.proj = nn.Linear(in_feats * patch_multiplier, d_model)
        else:
            self.conv = nn.Conv1d(
                in_feats,
                d_model,
                kernel_size=patch_len,
                stride=stride,
                bias=False,
            )
            self.proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F, L]
        x = self.conv(x)  # [B, C, Np]
        x = x.transpose(1, 2)  # [B, Np, C]
        if self.proj is not None:
            x = self.proj(x)
        return x


class PatchTSTEncoder(nn.Module):
    """PatchTST encoder used as the v0 backbone."""

    def __init__(
        self,
        in_feats: int,
        d_model: int = 192,
        depth: int = 3,
        patch_len: int = 16,
        stride: int = 8,
        n_heads: int = 8,
        dropout: float = 0.1,
        channel_independent: bool = True,
        patch_multiplier: int | None = None,
    ) -> None:
        super().__init__()

        if patch_multiplier is None:
            patch_multiplier = max(2, d_model // max(1, in_feats))

        self.patch_embed = PatchEmbedding(
            in_feats=in_feats,
            d_model=d_model,
            patch_len=patch_len,
            stride=stride,
            channel_independent=channel_independent,
            patch_multiplier=patch_multiplier,
        )

        self.positional = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_model * 4, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Time series tensor ``[B, L, F]``.

        Returns:
            Tuple of pooled representation ``[B, d_model]`` and token matrix ``[B, Np, d_model]``.
        """
        x = x.transpose(1, 2)  # [B, F, L]
        tokens = self.patch_embed(x)  # [B, Np, d_model]
        tokens = self.positional(tokens)

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)
        return pooled, tokens


class PositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.size(1)
        return tokens + self.pe[:, :length]
