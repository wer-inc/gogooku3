"""
ATFT-GAT-FAN アーキテクチャ実装
Adaptive Temporal Fusion Transformer with Graph Attention and Frequency Adaptive Normalization
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class ATFT_GAT_FAN(pl.LightningModule):
    """
    ATFT-GAT-FAN: Adaptive Temporal Fusion Transformer with Graph Attention and Frequency Adaptive Normalization

    特徴:
    - Temporal Fusion Transformer (TFT): 時系列特徴量の融合
    - Graph Attention Network (GAT): 銘柄間関係のモデリング
    - Frequency Adaptive Normalization (FAN): 周波数適応正規化
    - Slice Adaptive Normalization (SAN): スライス適応正規化
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        # self.save_hyperparameters()  # 一時的に無効化（テスト用）

        # 特徴量次元の計算
        self._calculate_feature_dims()

        # モデルコンポーネントの構築
        self._build_model()

        # 損失関数
        self._setup_loss_functions()

        logger.info(f"ATFT-GAT-FAN initialized with {self.n_dynamic_features} dynamic features")

    def _calculate_feature_dims(self):
        """特徴量次元の計算（ML_DATASET_COLUMNS.md準拠）"""
        data_config = self.config.data.features

        # 基本特徴量
        n_basic = len(data_config.basic.price_volume) + len(data_config.basic.flags)

        # テクニカル指標（拡張）
        n_technical = (
            len(data_config.technical.momentum) +
            len(data_config.technical.volatility) +
            len(data_config.technical.trend) +
            len(data_config.technical.moving_averages) +
            len(data_config.technical.macd) +
            len(getattr(data_config.technical, 'bollinger_bands', []))
        )

        # MA派生特徴量
        n_ma_derived = 0
        if hasattr(data_config, 'ma_derived') and data_config.ma_derived is not None:
            for category in ['price_deviations', 'ma_gaps', 'ma_slopes', 'ma_crosses', 'ma_ribbon']:
                if hasattr(data_config.ma_derived, category):
                    category_data = getattr(data_config.ma_derived, category)
                    if category_data is not None:
                        n_ma_derived += len(category_data)

        # リターン×MA相互作用特徴量
        n_interaction = 0
        if hasattr(data_config, 'returns_ma_interaction') and data_config.returns_ma_interaction is not None:
            for category in ['momentum', 'interactions']:
                if hasattr(data_config.returns_ma_interaction, category):
                    category_data = getattr(data_config.returns_ma_interaction, category)
                    if category_data is not None:
                        n_interaction += len(category_data)

        # フロー特徴量
        flow_data = getattr(data_config, 'flow', [])
        if flow_data is not None:
            if hasattr(flow_data, '__len__') and not isinstance(flow_data, str):
                try:
                    n_flow = len(flow_data)
                except TypeError:
                    n_flow = 0
            else:
                n_flow = 0
        else:
            n_flow = 0

        # リターン特徴量
        n_returns = len(data_config.returns.columns)

        # 現在値特徴量の合計
        self.n_current_features = n_basic + n_technical + n_ma_derived + n_interaction + n_flow + n_returns

        # 履歴特徴量の計算
        n_historical = 0
        if hasattr(data_config, 'historical') and data_config.historical is not None:
            try:
                for hist_name, hist_config in data_config.historical.items():
                    if hasattr(hist_config, 'range'):
                        n_historical += (hist_config.range[1] - hist_config.range[0] + 1)
            except (AttributeError, TypeError):
                # historicalが空辞書やNoneの場合
                n_historical = 0

        self.n_historical_features = n_historical

        # 合計特徴量数
        self.n_dynamic_features = self.n_current_features + self.n_historical_features

        # 静的特徴量（market_code_nameのエンコーディング後の次元）
        self.n_static_features = 10  # 仮の値（実際はエンコーディング方法による）

        logger.info(f"Feature dimensions - Basic: {n_basic}, Technical: {n_technical}, "
                   f"MA-derived: {n_ma_derived}, Interaction: {n_interaction}, "
                   f"Flow: {n_flow}, Returns: {n_returns}")
        logger.info(f"Total current features: {self.n_current_features}, "
                   f"Historical: {self.n_historical_features}, Total: {self.n_dynamic_features}")

        # ML_DATASET_COLUMNS.md準拠チェック
        expected_features = 59  # ML_DATASET_COLUMNS.mdより
        if abs(self.n_current_features - expected_features) > 10:  # 許容誤差
            logger.warning(f"Feature count mismatch! Expected ~{expected_features}, got {self.n_current_features}")
            logger.warning("Please verify data configuration matches ML_DATASET_COLUMNS.md")

    def _build_model(self):
        """モデルアーキテクチャの構築"""

        # 入力投影層
        self.input_projection = self._build_input_projection()

        # Temporal Fusion Transformer
        self.tft = self._build_tft()

        # Graph Attention Network
        if self.config.model.gat.enabled:
            self.gat = self._build_gat()

        # 適応正規化
        self.adaptive_norm = self._build_adaptive_normalization()

        # FreqDropout（オプション）
        freq_dropout_p = getattr(self.config, 'freq_dropout_p', 0.0)
        if freq_dropout_p is not None and isinstance(freq_dropout_p, (int, float)) and freq_dropout_p > 0:
            from ...components.freq_dropout import FreqDropout1D
            self.freq_dropout = FreqDropout1D(
                p=freq_dropout_p,
                min_width=getattr(self.config, 'freq_dropout_min_width', 0.05),
                max_width=getattr(self.config, 'freq_dropout_max_width', 0.2)
            )
        else:
            self.freq_dropout = None

        # 予測ヘッド（改善版）
        self.prediction_head = self._build_prediction_head()

    def _build_input_projection(self):
        """入力投影層"""
        return nn.Sequential(
            nn.Linear(self.n_dynamic_features, self.config.model.hidden_size),
            nn.LayerNorm(self.config.model.hidden_size) if self.config.model.input_projection.use_layer_norm else nn.Identity(),
            nn.Dropout(self.config.model.input_projection.dropout)
        )

    def _build_tft(self):
        """Temporal Fusion Transformerの構築"""
        # 入力投影後の次元を使用
        projected_features = self.config.model.hidden_size
        return TemporalFusionTransformer(
            hidden_size=self.config.model.hidden_size,
            n_dynamic_features=projected_features,  # 投影後の次元を使用
            n_static_features=self.n_static_features,
            config=self.config.model.tft
        )

    def _build_gat(self):
        """Graph Attention Networkの構築"""
        return GraphAttentionNetwork(
            hidden_size=self.config.model.hidden_size,
            config=self.config.model.gat
        )

    def _build_adaptive_normalization(self):
        """適応正規化層の構築"""
        return AdaptiveNormalization(
            hidden_size=self.config.model.hidden_size,
            config=self.config.model.adaptive_normalization
        )

    def _build_prediction_head(self):
        """予測ヘッドの構築"""
        return PredictionHead(
            hidden_size=self.config.model.hidden_size,
            config=self.config.model.prediction_head
        )

    def _setup_loss_functions(self):
        """損失関数の設定"""
        # Quantile Loss（メイン）
        quantiles = self.config.model.prediction_head.output.quantile_prediction.quantiles
        self.quantile_loss = QuantileLoss(quantiles)

        # 補助損失
        if self.config.train.loss.auxiliary.sharpe_loss.enabled:
            self.sharpe_loss = SharpeLoss(
                weight=self.config.train.loss.auxiliary.sharpe_loss.weight,
                min_periods=self.config.train.loss.auxiliary.sharpe_loss.min_periods
            )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        フォワードパス
        """
        # 入力データの展開
        dynamic_features = batch['dynamic_features']  # [batch, seq_len, n_features]
        static_features = batch.get('static_features', None)  # [batch, n_static]
        edge_index = batch.get('edge_index', None)
        edge_attr = batch.get('edge_attr', None)

        # 入力投影
        x = self.input_projection(dynamic_features)

        # Temporal Fusion Transformer
        tft_output = self.tft(x, static_features)

        # Graph Attention Network（有効な場合）
        if self.config.model.gat.enabled and edge_index is not None:
            graph_output = self.gat(tft_output, edge_index, edge_attr)
            combined_features = torch.cat([tft_output, graph_output], dim=-1)
        else:
            combined_features = tft_output

        # FreqDropout適用（オプション）
        if self.freq_dropout is not None:
            combined_features = self.freq_dropout(combined_features)

        # 適応正規化
        normalized_features = self.adaptive_norm(combined_features)

        # 予測
        predictions = self.prediction_head(normalized_features)

        return {
            'predictions': predictions,
            'features': normalized_features
        }

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """学習ステップ"""
        outputs = self.forward(batch)
        targets = batch['targets']

        # メイン損失（Quantile Loss）
        main_loss = self.quantile_loss(outputs['predictions'], targets)

        # 補助損失
        total_loss = main_loss

        if hasattr(self, 'sharpe_loss'):
            sharpe_loss = self.sharpe_loss(outputs['predictions'], targets)
            total_loss = total_loss + sharpe_loss

        # ログ記録
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_main_loss', main_loss, prog_bar=True)

        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """検証ステップ"""
        outputs = self.forward(batch)
        targets = batch['targets']

        # 損失計算
        loss = self.quantile_loss(outputs['predictions'], targets)

        # 金融指標計算
        metrics = self._calculate_financial_metrics(outputs['predictions'], targets)

        # ログ記録
        self.log('val_loss', loss, prog_bar=True)
        for metric_name, metric_value in metrics.items():
            self.log(f'val_{metric_name}', metric_value, prog_bar=True)

        return loss

    def _calculate_financial_metrics(self, predictions: torch.Tensor, targets: torch.Tensor):
        """金融指標の計算"""
        # Sharpe比
        returns = targets.mean(dim=-1)  # 各ホライズンの平均
        if returns.std() > 0:
            sharpe = returns.mean() / returns.std() * torch.sqrt(torch.tensor(252.0))
        else:
            sharpe = torch.tensor(0.0)

        # 最大ドローダウン（簡易計算）
        cumulative_returns = torch.cumprod(1 + returns, dim=0)
        running_max = torch.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # 勝率
        pred_direction = (predictions > 0).float()
        true_direction = (targets > 0).float()
        hit_rate = (pred_direction == true_direction).float().mean()

        return {
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'hit_rate': hit_rate
        }

    def configure_optimizers(self):
        """オプティマイザーの設定"""
        optimizer_config = self.config.train.optimizer

        if optimizer_config.type == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=optimizer_config.lr,
                weight_decay=optimizer_config.weight_decay,
                betas=tuple(optimizer_config.betas),
                eps=optimizer_config.eps
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_config.type}")

        # スケジューラー設定
        scheduler_config = self.config.train.scheduler

        if scheduler_config.type == 'CosineAnnealingWarmRestarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=scheduler_config.T_0,
                T_mult=scheduler_config.T_mult,
                eta_min=scheduler_config.eta_min
            )
        else:
            scheduler = None

        if scheduler:
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch'
                }
            }
        else:
            return optimizer


# 基本コンポーネント（簡易実装）
class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer (簡易実装)"""
    def __init__(self, hidden_size: int, n_dynamic_features: int, n_static_features: int, config: DictConfig):
        super().__init__()
        self.hidden_size = hidden_size
        self.config = config

        # LSTMエンコーダー
        self.lstm = nn.LSTM(
            input_size=n_dynamic_features,
            hidden_size=hidden_size,
            num_layers=config.lstm.layers,
            dropout=config.lstm.dropout,
            batch_first=True
        )

        # 静的特徴量エンコーダー
        if n_static_features > 0:
            self.static_encoder = nn.Linear(n_static_features, hidden_size)
        else:
            self.static_encoder = None

    def forward(self, x: torch.Tensor, static_features: Optional[torch.Tensor] = None):
        # LSTM処理
        lstm_out, _ = self.lstm(x)

        # 静的特徴量の統合
        if static_features is not None and self.static_encoder is not None:
            static_encoded = self.static_encoder(static_features)
            # 簡易的な統合（実際にはより複雑な処理が必要）
            combined = lstm_out + static_encoded.unsqueeze(1)
        else:
            combined = lstm_out

        return combined


class GraphAttentionNetwork(nn.Module):
    """Graph Attention Network (簡易実装)"""
    def __init__(self, hidden_size: int, config: DictConfig):
        super().__init__()
        self.config = config

        # 簡易的なGAT層
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=config.architecture.heads[0],
            dropout=config.layer_config.dropout,
            batch_first=True
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None):
        # 簡易的なグラフ注意処理
        attn_output, _ = self.attention(x, x, x)
        return attn_output


class AdaptiveNormalization(nn.Module):
    """適応正規化 (簡易実装)"""
    def __init__(self, hidden_size: int, config: DictConfig):
        super().__init__()
        self.config = config

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

        # 適応パラメータ（学習可能）
        if config.fan.enabled:
            self.fan_weights = nn.Parameter(torch.ones(len(config.fan.window_sizes)))
        if config.san.enabled:
            self.san_weights = nn.Parameter(torch.ones(config.san.num_slices))

    def forward(self, x: torch.Tensor):
        # 基本的なLayer Normalization
        return self.layer_norm(x)


class PredictionHead(nn.Module):
    """改善版予測ヘッド（small-init + LayerScale）"""
    def __init__(self, hidden_size: int, config: DictConfig, output_std: float = 0.01, layer_scale_gamma: float = 0.1):
        super().__init__()
        self.config = config

        # 隠れ層
        layers = []
        for hidden_dim in config.architecture.hidden_layers:
            layers.extend([
                nn.Linear(hidden_size, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.architecture.dropout)
            ])
            hidden_size = hidden_dim

        # 出力層（改善版初期化）
        quantiles = config.output.quantile_prediction.quantiles
        self.output_layer = nn.Linear(hidden_size, len(quantiles))

        # small-init + zero bias
        nn.init.trunc_normal_(self.output_layer.weight, std=output_std)
        nn.init.zeros_(self.output_layer.bias)

        # LayerScale
        self.layer_scale = nn.Parameter(torch.ones(len(quantiles)) * layer_scale_gamma)

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        # 最後のタイムステップを使用
        x = x[:, -1, :]  # [batch, hidden_size]

        # 隠れ層処理
        x = self.layers(x)

        # 出力 + LayerScale
        output = self.output_layer(x)
        return output * self.layer_scale


class QuantileLoss(nn.Module):
    """Quantile Loss"""
    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.quantiles = torch.tensor(quantiles)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        # 簡易的なQuantile Loss実装
        errors = targets.unsqueeze(-1) - predictions
        quantile_loss = torch.maximum(
            self.quantiles * errors,
            (self.quantiles - 1) * errors
        )
        return quantile_loss.mean()


class SharpeLoss(nn.Module):
    """Sharpe比最大化損失"""
    def __init__(self, weight: float = 0.1, min_periods: int = 20):
        super().__init__()
        self.weight = weight
        self.min_periods = min_periods

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        # 簡易的なSharpe比計算
        returns = targets.mean(dim=-1)
        if len(returns) >= self.min_periods and returns.std() > 0:
            sharpe = returns.mean() / returns.std()
            # Sharpe比を最大化するための損失（負のSharpe比）
            return -self.weight * sharpe
        else:
            return torch.tensor(0.0)
