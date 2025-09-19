"""
Frequency Adaptive Normalization (FAN) with FreqDropout
周波数適応正規化と帯域マスクドロップアウト
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math
import logging

logger = logging.getLogger(__name__)


class FreqDropout1D(nn.Module):
    """
    周波数ドメイン帯域マスクドロップアウト
    学習時のみランダム帯域をゼロマスク
    """

    def __init__(
        self,
        p: float = 0.1,
        min_width: float = 0.05,
        max_width: float = 0.2,
        dim: int = -2  # 時間次元
    ):
        """
        Args:
            p: ドロップアウト確率
            min_width: 最小帯域幅（全長に対する割合）
            max_width: 最大帯域幅（全長に対する割合）
            dim: FFTを適用する次元
        """
        super().__init__()
        self.p = p
        self.min_width = min_width
        self.max_width = max_width
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 入力テンソル [..., seq_len, features]

        Returns:
            帯域マスク適用後のテンソル
        """
        if not self.training or torch.rand(1) > self.p:
            return x

        # FFT適用
        x_fft = torch.fft.rfft(x, dim=self.dim)

        # 周波数次元のサイズ
        freq_dim = x_fft.size(self.dim)
        if freq_dim <= 1:
            return x

        # ランダム帯域選択
        width_ratio = torch.empty(1).uniform_(self.min_width, self.max_width)
        width = max(1, int(freq_dim * width_ratio))

        # 開始位置（低周波数側を避ける）
        start = torch.randint(1, freq_dim - width + 1, (1,)).item()

        # マスク適用
        mask = torch.ones_like(x_fft)
        indices = [slice(None)] * x_fft.ndim
        indices[self.dim] = slice(start, start + width)
        mask[indices] = 0

        x_fft_masked = x_fft * mask

        # IFFTで時間領域に戻す
        x_masked = torch.fft.irfft(x_fft_masked, n=x.size(self.dim), dim=self.dim)

        return x_masked


class FrequencyAdaptiveNormalization(nn.Module):
    """
    周波数適応正規化 (FAN)
    帯域ごとに適応的なゲイン制御
    """

    def __init__(
        self,
        num_features: int,
        seq_len: int,
        num_bands: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            num_features: 特徴量数
            seq_len: シーケンス長
            num_bands: 周波数帯域数
            dropout: ドロップアウト率
        """
        super().__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.num_bands = num_bands

        # 周波数帯域ごとのゲイン
        self.band_gains = nn.Parameter(torch.ones(num_bands, num_features))

        # 帯域の境界
        self.register_buffer('band_edges', self._create_band_edges())

        # ドロップアウト
        self.dropout = nn.Dropout(dropout)

    def _create_band_edges(self) -> torch.Tensor:
        """周波数帯域の境界を作成"""
        # 対数スケールで帯域分割
        freq_bins = torch.logspace(0, math.log10(self.seq_len // 2), self.num_bands + 1)
        return freq_bins.long()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 入力テンソル [batch, seq_len, features]

        Returns:
            FAN適用後のテンソル
        """
        if not self.training:
            # 推論時はシンプルな処理
            return self.dropout(x)

        # FFT
        x_fft = torch.fft.rfft(x, dim=-2)  # [batch, freq, features]

        # 周波数帯域ごとの処理
        batch_size, freq_len, num_features = x_fft.shape
        output_fft = torch.zeros_like(x_fft)

        for i in range(self.num_bands):
            start_freq = self.band_edges[i]
            end_freq = self.band_edges[i + 1]

            if start_freq >= freq_len:
                continue

            end_freq = min(end_freq, freq_len)

            # 帯域抽出
            band_fft = x_fft[:, start_freq:end_freq, :]  # [batch, band_freq, features]

            # ゲイン適用
            band_gain = self.band_gains[i].unsqueeze(0).unsqueeze(1)  # [1, 1, features]
            band_fft = band_fft * band_gain

            # 結果格納
            output_fft[:, start_freq:end_freq, :] = band_fft

        # IFFT
        output = torch.fft.irfft(output_fft, n=self.seq_len, dim=-2)

        return self.dropout(output)


class AdaptiveNormalization(nn.Module):
    """
    適応正規化統合モジュール
    LayerNorm + FAN + Slice Adaptive Norm
    """

    def __init__(
        self,
        hidden_size: int,
        config: Optional[any] = None,
        fan_enabled: bool = True,
        san_enabled: bool = False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.fan_enabled = fan_enabled
        self.san_enabled = san_enabled

        # 基本的なLayerNorm
        self.layer_norm = nn.LayerNorm(hidden_size)

        # FAN（オプション）
        if fan_enabled:
            self.fan = FrequencyAdaptiveNormalization(
                num_features=hidden_size,
                seq_len=60,  # デフォルト値
                num_bands=8
            )
        else:
            self.fan = None

        # Slice Adaptive Norm（オプション）
        if san_enabled:
            self.san_weights = nn.Parameter(torch.ones(4))  # 4スライス
        else:
            self.san_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 入力テンソル [batch, seq_len, hidden_size]

        Returns:
            正規化適用後のテンソル
        """
        # LayerNorm
        x = self.layer_norm(x)

        # FAN（オプション）
        if self.fan is not None and self.fan_enabled:
            x = self.fan(x)

        # Slice Adaptive Norm（オプション）
        if self.san_weights is not None and self.san_enabled:
            # シーケンスを4スライスに分割
            seq_len = x.size(1)
            slice_size = seq_len // 4

            outputs = []
            for i in range(4):
                start = i * slice_size
                end = (i + 1) * slice_size if i < 3 else seq_len

                slice_x = x[:, start:end, :]
                slice_norm = self.layer_norm(slice_x)
                slice_weighted = slice_norm * self.san_weights[i]

                outputs.append(slice_weighted)

            x = torch.cat(outputs, dim=1)

        return x


# 後方互換性のためのエイリアス
FAN = FrequencyAdaptiveNormalization
FreqDropout = FreqDropout1D
