"""Adaptive normalization components (FAN and SAN)."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        import logging
        import os

        logger = logging.getLogger(__name__)

        if x.dim() != 3:
            raise ValueError(f"FAN expects 3D input (B, L, F); got {x.shape}")

        # Debug gradient flow through FAN
        if self.training and os.getenv("ENABLE_FAN_GRAD_DEBUG", "0") == "1":
            x.retain_grad()

            def _log_fan_input_grad(grad):
                logger.info(f"[FAN-GRAD] INPUT grad_norm={float(grad.norm()):.3e}")
                return grad

            x.register_hook(_log_fan_input_grad)

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # Check if nan_to_num broke gradient flow
        if self.training and os.getenv("ENABLE_FAN_GRAD_DEBUG", "0") == "1":

            def _log_fan_post_nan_grad(grad):
                logger.info(f"[FAN-GRAD] POST-NAN grad_norm={float(grad.norm()):.3e}")
                return grad

            x.register_hook(_log_fan_post_nan_grad)

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

        # Debug gradient flow at FAN output
        if self.training and os.getenv("ENABLE_FAN_GRAD_DEBUG", "0") == "1":

            def _log_fan_output_grad(grad):
                logger.info(f"[FAN-GRAD] OUTPUT grad_norm={float(grad.norm()):.3e}")
                return grad

            output.register_hook(_log_fan_output_grad)

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
        import logging
        import os

        logger = logging.getLogger(__name__)

        if x.dim() != 3:
            raise ValueError(f"SAN expects 3D input (B, L, F); got {x.shape}")

        # Debug gradient flow through SAN
        if self.training and os.getenv("ENABLE_SAN_GRAD_DEBUG", "0") == "1":
            x.retain_grad()

            def _log_san_input_grad(grad):
                logger.info(f"[SAN-GRAD] INPUT grad_norm={float(grad.norm()):.3e}")
                return grad

            x.register_hook(_log_san_input_grad)

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
                normalized = F.layer_norm(
                    slice_data, (slice_data.size(-1),), eps=self.eps
                )
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
                # FIX: Use F.pad instead of zeros_like to preserve gradient flow
                length = end - start
                pad_left = start
                pad_right = seq_len - end
                # Pad normalized data (which has gradients) instead of copying into disconnected tensor
                padded = F.pad(
                    normalized[:, :length, :], (0, 0, pad_left, pad_right), value=0.0
                )
                stacked.append(padded)
            stacked = torch.stack(stacked, dim=2)  # (B, L, num_slices, F)
            output = (stacked * weights).sum(dim=2)
        else:
            # FIX: Build output from normalized slices with proper gradient flow
            # Instead of accumulating into zeros_like(x), use scatter operations
            output_list = []
            for normalized, start, end in normalized_slices:
                length = end - start
                pad_left = start
                pad_right = seq_len - end
                padded = F.pad(
                    normalized[:, :length, :], (0, 0, pad_left, pad_right), value=0.0
                )
                output_list.append(padded)

            # Average overlapping regions
            if output_list:
                stacked = torch.stack(output_list, dim=0)  # (num_slices, B, L, F)
                output = stacked.mean(dim=0)  # (B, L, F)
            else:
                output = F.layer_norm(x, (self.num_features,), eps=self.eps)

        # Debug gradient flow at SAN output
        if self.training and os.getenv("ENABLE_SAN_GRAD_DEBUG", "0") == "1":

            def _log_san_output_grad(grad):
                logger.info(f"[SAN-GRAD] OUTPUT grad_norm={float(grad.norm()):.3e}")
                return grad

            output.register_hook(_log_san_output_grad)

        return output


# ============================================================================
# Stable FAN/SAN (P0-1 修復版) - TFT入力直前 [B,T,H] 用
# ============================================================================


class FrequencyAdaptiveNormSimple(nn.Module):
    """周波数適応正規化（安定版）

    複数の時間窓で統計を計算し、学習可能な重みでブレンド。
    数値安定性のため σ の下限を設定し、NaN除去を徹底。
    """
    def __init__(self, feat_dim: int, windows=(5, 10, 20), eps=1e-4):
        super().__init__()
        self.windows = tuple(sorted(windows))
        self.eps = eps
        # [H, W] 各特徴量×各窓の重み
        self.alpha = nn.Parameter(torch.zeros(feat_dim, len(self.windows)))
        # エントロピー追跡用
        self._entropy = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, H] バッチ×時系列×特徴量
        Returns:
            正規化された tensor [B, T, H]
        """
        B, T, H = x.shape
        # Softmax で重み正規化
        al = torch.softmax(self.alpha, dim=-1)  # [H, W]

        # エントロピー計算（正則化用）
        self._entropy = -(al.clamp_min(1e-12) * al.clamp_min(1e-12).log()).sum()

        out = torch.zeros_like(x)
        for wi, w in enumerate(self.windows):
            if w > T:  # 窓が長い場合はスキップ（自動降格）
                continue
            # スライディングウィンドウ展開
            u = x.unfold(1, w, 1)  # [B, T-w+1, H, w]
            mu = u.mean(dim=3)
            sd = u.std(dim=3).clamp_min(self.eps)

            # パディング（先頭を複製）
            mu = F.pad(mu, (0, 0, w-1, 0), mode='replicate')
            sd = F.pad(sd, (0, 0, w-1, 0), mode='replicate')

            # 正規化（分母にフロア設定）
            z = (x - mu) / sd
            # NaN/Inf除去
            z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

            # 窓別の重みで加算
            out = out + z * al[:, wi].view(1, 1, H)

        return out


class SliceAdaptiveNormSimple(nn.Module):
    """スライス適応正規化（安定版）

    時系列を重複するスライスに分割し、各スライスで統計を計算。
    学習可能なアフィン変換を適用。
    """
    def __init__(self, feat_dim: int, num_slices=3, overlap=0.5, eps=1e-4):
        super().__init__()
        assert 0 <= overlap < 1, "overlap must be in [0, 1)"
        self.K = int(num_slices)
        self.overlap = float(overlap)
        self.eps = eps

        # スライスごとのアフィン変換パラメータ
        self.gamma = nn.Parameter(torch.ones(self.K, feat_dim))
        self.beta = nn.Parameter(torch.zeros(self.K, feat_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, H]
        Returns:
            [B, T, H]
        """
        B, T, H = x.shape

        # スライスサイズ計算
        step = max(1, int(round(T / (self.K - (self.K - 1) * self.overlap))))
        win = min(T, step + int(round(step * self.overlap)))

        # 重複加算用バッファ
        y = torch.zeros_like(x)
        cnt = torch.zeros(B, T, 1, device=x.device)

        s = 0
        for k in range(self.K):
            e = min(T, s + win)
            if e <= s:
                break

            xk = x[:, s:e, :]

            # スライス内統計（勾配を流さない）
            with torch.no_grad():
                mu = xk.mean(dim=(0, 1), keepdim=True)
                sd = xk.std(dim=(0, 1), keepdim=True).clamp_min(1e-4)

            # 正規化
            z = (xk - mu) / sd
            z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

            # アフィン変換（学習可能）
            z = z * self.gamma[k].view(1, 1, H) + self.beta[k].view(1, 1, H)

            # 重複領域は加算
            y[:, s:e, :] += z
            cnt[:, s:e, :] += 1.0

            s += step

        # 平均化（重複カウントで割る）
        return y / (cnt + self.eps)


class StableFANSAN(nn.Module):
    """FAN→SAN統合モジュール（Pre-Norm + Residual）

    TFT入力直前（[B,T,H]）に適用する安定版。
    内部で Pre-Norm + Residual（出力は x + SAN(FAN(x))）。

    特徴:
    - Pre-Normalization: 入力直後に適用
    - Residual connection: 元のスケールを保持
    - エントロピー正則化: αの一極集中を防止
    - 数値安定性: すべての分母にフロア、NaN自動除去
    """
    def __init__(
        self,
        feat_dim: int,
        windows=(5, 10, 20),
        num_slices=3,
        overlap=0.5,
        eps=1e-4,
        entropy_coef=1e-4
    ):
        super().__init__()
        self.fan = FrequencyAdaptiveNormSimple(feat_dim, windows, eps)
        self.san = SliceAdaptiveNormSimple(feat_dim, num_slices, overlap, eps)
        self.entropy_coef = float(entropy_coef)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, H] 入力特徴量（TFT入力直前）
        Returns:
            [B, T, H] 正規化済み特徴量（Residual付き）
        """
        xin = x
        # FAN → SAN の順で適用
        x = self.fan(x)
        x = self.san(x)
        # Residual: 元のスケールを保持
        return xin + x

    def regularizer(self) -> torch.Tensor:
        """正則化項（訓練ループで loss に加算）

        Returns:
            エントロピー正則化項
        """
        return self.entropy_coef * self.fan._entropy.abs()
