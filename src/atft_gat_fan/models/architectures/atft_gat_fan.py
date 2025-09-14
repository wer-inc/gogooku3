"""
ATFT-GAT-FAN アーキテクチャ実装
Adaptive Temporal Fusion Transformer with Graph Attention and Frequency Adaptive Normalization
"""
import logging

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig

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
        """予測ヘッドの構築（Multi-horizon / RegimeMoE 対応）"""
        # Detect requested head type; default to multi-horizon for backward compatibility
        head_type = getattr(self.config.model.prediction_head, 'type', 'multi_horizon')

        if head_type == 'regime_moe':
            # Lazy import to avoid circulars
            from .regime_moe import RegimeMoEPredictionHeads

            return RegimeMoEPredictionHeads(
                hidden_size=self.config.model.hidden_size,
                config=self.config.model
            )

        # Multi-horizon prediction head（従来）
        use_multi_horizon = getattr(self.config.training, 'use_multi_horizon_heads', True)
        if use_multi_horizon:
            return MultiHorizonPredictionHeads(
                hidden_size=self.config.model.hidden_size,
                config=self.config.model.prediction_head
            )
        else:
            # Backward compatibility: single horizon head
            return PredictionHead(
                hidden_size=self.config.model.hidden_size,
                config=self.config.model.prediction_head
            )

    def _setup_loss_functions(self):
        """損失関数の設定"""
        # Quantile Loss（メイン）
        quantiles = self.config.model.prediction_head.output.quantile_prediction.quantiles
        self.quantile_loss = QuantileLoss(quantiles)

        # 中央分位点のインデックス（Rank損失用のスコア抽出）
        try:
            # 最も0.5に近い分位点を採用
            q_list = list(float(q) for q in quantiles)
            self._median_q_idx = min(range(len(q_list)), key=lambda i: abs(q_list[i] - 0.5))
        except Exception:
            self._median_q_idx = 0

        # 補助損失
        if self.config.train.loss.auxiliary.sharpe_loss.enabled:
            self.sharpe_loss = SharpeLoss(
                weight=self.config.train.loss.auxiliary.sharpe_loss.weight,
                min_periods=self.config.train.loss.auxiliary.sharpe_loss.min_periods
            )

        # ランキング損失（任意）
        try:
            rk_cfg = self.config.train.loss.auxiliary.ranking_loss
            if getattr(rk_cfg, 'enabled', False):
                from ....losses.pairwise_rank_loss import PairwiseRankLoss

                scale = float(getattr(rk_cfg, 'scale', getattr(rk_cfg, 'margin', 5.0)))
                topk = int(getattr(rk_cfg, 'topk', 0))
                self.rank_loss = PairwiseRankLoss(s=scale, topk=topk)
                self.rank_loss_weight = float(getattr(rk_cfg, 'weight', 0.1))
            else:
                self.rank_loss = None
                self.rank_loss_weight = 0.0
        except Exception:
            self.rank_loss = None
            self.rank_loss_weight = 0.0

        # 意思決定層（任意）
        try:
            dl_cfg = self.config.train.loss.auxiliary.decision_layer
            if getattr(dl_cfg, 'enabled', False):
                from ....losses.decision_layer import DecisionLayer, DecisionLossConfig

                self.decision_layer = DecisionLayer(DecisionLossConfig(
                    alpha=float(getattr(dl_cfg, 'alpha', 2.0)),
                    method=str(getattr(dl_cfg, 'method', 'tanh')),
                    sharpe_weight=float(getattr(dl_cfg, 'sharpe_weight', 0.1)),
                    pos_l2=float(getattr(dl_cfg, 'pos_l2', 1e-3)),
                    fee_abs=float(getattr(dl_cfg, 'fee_abs', 0.0)),
                    detach_signal=bool(getattr(dl_cfg, 'detach_signal', True)),
                ))

                # Decision Layer スケジューラ（有効な場合）
                sched_cfg = self.config.train.loss.auxiliary.get('decision_layer_schedule', {})
                if sched_cfg.get('enabled', False):
                    from ....training.decision_scheduler import create_decision_scheduler_from_config
                    self.decision_scheduler = create_decision_scheduler_from_config(
                        self.config, self.decision_layer
                    )
                    logger.info("Decision Layer scheduler enabled")
                else:
                    self.decision_scheduler = None
            else:
                self.decision_layer = None
                self.decision_scheduler = None
        except Exception as e:
            logger.warning(f"Decision Layer initialization failed: {e}")
            self.decision_layer = None
            self.decision_scheduler = None

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Enhanced forward pass with regime features support
        """
        # 入力データの展開
        dynamic_features = batch['dynamic_features']  # [batch, seq_len, n_features]
        static_features = batch.get('static_features', None)  # [batch, n_static]
        edge_index = batch.get('edge_index', None)
        edge_attr = batch.get('edge_attr', None)

        # レジーム特徴量（J-UVX + KAMA/VIDYA + market regimes）
        regime_features = batch.get('regime_features', None)  # [batch, regime_dim]

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

        # 予測（Multi-horizon対応 + レジーム特徴量対応）
        if hasattr(self.prediction_head, 'forward') and 'regime_features' in self.prediction_head.forward.__code__.co_varnames:
            # Enhanced RegimeMoE prediction head
            predictions = self.prediction_head(normalized_features, regime_features)
        else:
            # Standard prediction head (backward compatibility)
            predictions = self.prediction_head(normalized_features)

        # Multi-horizon vs single-horizon の結果統一
        if isinstance(predictions, dict):
            # Multi-horizon: {horizon_1d: tensor, horizon_5d: tensor, ...}
            output_type = 'multi_horizon'
        else:
            # Single-horizon (backward compatibility)
            output_type = 'single_horizon'
            predictions = {'single': predictions}

        # 出力にレジーム特徴量も含める（分析用）
        output = {
            'predictions': predictions,
            'features': normalized_features,
            'output_type': output_type
        }

        if regime_features is not None:
            output['regime_features'] = regime_features

        # MoEゲート分析情報（利用可能な場合）
        if hasattr(self.prediction_head, 'get_gate_analysis'):
            try:
                gate_analysis = self.prediction_head.get_gate_analysis()
                output['gate_analysis'] = gate_analysis
            except:
                pass

        return output

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        """学習ステップ（Multi-horizon対応）"""
        # Decision Layer スケジューラ更新（エポック開始時に一度だけ）
        if (hasattr(self, 'decision_scheduler') and self.decision_scheduler is not None and
            batch_idx == 0):  # エポックの最初のバッチでのみ更新
            current_epoch = self.current_epoch if hasattr(self, 'current_epoch') else 0
            scheduled_params = self.decision_scheduler.step(current_epoch, self.decision_layer)

            # スケジュールされたパラメータをログ
            for param_name, param_value in scheduled_params.items():
                self.log(f'decision_schedule/{param_name}', param_value,
                        on_step=False, on_epoch=True, prog_bar=False)

        outputs = self.forward(batch)
        predictions = outputs['predictions']
        output_type = outputs['output_type']

        total_loss = 0.0
        horizon_losses = {}

        if output_type == 'multi_horizon':
            # Multi-horizon training: 各horizonでの損失計算
            # 新しいconfig構造から重みを取得
            if (hasattr(self.config.training, 'prediction') and
                hasattr(self.config.training.prediction, 'horizon_weights')):
                horizon_weight_list = self.config.training.prediction.horizon_weights
                horizon_weights = {
                    f'horizon_{h}d': w for h, w in zip(self.prediction_horizons, horizon_weight_list)
                }
            else:
                # フォールバック: 従来の設定またはデフォルト
                horizon_weights = getattr(self.config.training, 'horizon_weights', {
                    'horizon_1d': 1.0, 'horizon_5d': 0.8, 'horizon_10d': 0.6, 'horizon_20d': 0.4
                })

            for horizon_key, pred in predictions.items():
                # Extract corresponding target for this horizon
                if horizon_key in batch:
                    target = batch[horizon_key]
                elif 'targets' in batch:
                    # Fallback to single target (assume it matches the prediction format)
                    target = batch['targets']
                else:
                    # Skip if no matching target
                    continue

                # Horizon-specific loss
                horizon_loss = self.quantile_loss(pred, target)

                # Apply horizon weighting (emphasize short-term)
                weight = horizon_weights.get(horizon_key, 0.5)
                weighted_loss = horizon_loss * weight

                total_loss += weighted_loss
                horizon_losses[f'train_loss_{horizon_key}'] = horizon_loss

                # Log individual horizon losses
                self.log(f'train_loss_{horizon_key}', horizon_loss, prog_bar=False)

        else:
            # Single-horizon training (backward compatibility)
            targets = batch['targets']
            main_loss = self.quantile_loss(predictions['single'], targets)
            total_loss = main_loss
            self.log('train_main_loss', main_loss, prog_bar=True)

        # Auxiliary losses (applied to all horizons)
        if hasattr(self, 'sharpe_loss') and output_type == 'multi_horizon':
            # Apply Sharpe loss to primary horizon (usually 1d or 5d)
            primary_horizon = getattr(self.config.training, 'primary_horizon', 'horizon_1d')
            if primary_horizon in predictions and primary_horizon in batch:
                sharpe_loss = self.sharpe_loss(predictions[primary_horizon], batch[primary_horizon])
                total_loss += sharpe_loss
                self.log('train_sharpe_loss', sharpe_loss, prog_bar=False)

        # Rank損失（主ホライズンの中央値スコアでペアワイズ）
        if output_type == 'multi_horizon' and self.rank_loss is not None and self.rank_loss_weight > 0:
            primary_horizon = getattr(self.config.training, 'primary_horizon', 'horizon_1d')
            if primary_horizon in predictions and primary_horizon in batch:
                pred_q = predictions[primary_horizon]  # [B, n_quantiles]
                if pred_q.dim() == 2 and pred_q.size(1) > self._median_q_idx:
                    z = pred_q[:, self._median_q_idx]
                    y = batch[primary_horizon].view(-1)
                    rk = self.rank_loss(z.view(-1), y.view(-1)) * self.rank_loss_weight
                    total_loss = total_loss + rk
                    self.log('train_rank_loss', rk.detach(), prog_bar=False)

        # 意思決定層ロス（主ホライズンの分位点）
        if output_type == 'multi_horizon' and self.decision_layer is not None:
            primary_horizon = getattr(self.config.training, 'primary_horizon', 'horizon_1d')
            if primary_horizon in predictions and primary_horizon in batch:
                q = predictions[primary_horizon]  # [B, n_quantiles]
                y = batch[primary_horizon].view(-1)
                dl_total, comps = self.decision_layer(q, y)
                total_loss = total_loss + dl_total
                # 重要メトリクスをログ
                self.log('train_decision_sharpe', comps['decision_sharpe'], prog_bar=False)
                self.log('train_pos_l2', comps['decision_pos_l2'], prog_bar=False)
                self.log('train_fee', comps['decision_fee'], prog_bar=False)

        # ログ記録
        self.log('train_loss', total_loss, prog_bar=True)

        # MoE load-balance正則化（オプション）
        try:
            moe_cfg = getattr(self.config.model.prediction_head, 'moe', None)
            lb_lambda = float(getattr(moe_cfg, 'balance_lambda', 0.0)) if moe_cfg is not None else 0.0
        except Exception:
            lb_lambda = 0.0

        if lb_lambda > 0 and hasattr(self.prediction_head, 'last_gate_probs'):
            # Lazy import to avoid cost when not needed
            try:
                from .regime_moe import moe_load_balance_penalty
                lb_loss = lb_lambda * moe_load_balance_penalty(self.prediction_head.last_gate_probs)
                total_loss = total_loss + lb_loss
                self.log('train_moe_lb_loss', lb_loss, prog_bar=False)
            except Exception:
                pass

        return total_loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        """検証ステップ（Multi-horizon対応）"""
        outputs = self.forward(batch)
        predictions = outputs['predictions']
        output_type = outputs['output_type']

        total_loss = 0.0

        if output_type == 'multi_horizon':
            # Multi-horizon validation
            for horizon_key, pred in predictions.items():
                if horizon_key in batch:
                    target = batch[horizon_key]
                elif 'targets' in batch:
                    target = batch['targets']
                else:
                    continue

                val_loss = self.quantile_loss(pred, target)
                total_loss += val_loss

                # Log individual horizon validation losses
                self.log(f'val_loss_{horizon_key}', val_loss, prog_bar=False, sync_dist=True)

        else:
            # Single-horizon validation
            targets = batch['targets']
            val_loss = self.quantile_loss(predictions['single'], targets)
            total_loss = val_loss

        self.log('val_loss', total_loss, prog_bar=True, sync_dist=True)
        return total_loss

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

    def forward(self, x: torch.Tensor, static_features: torch.Tensor | None = None):
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

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None):
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
    """改善版予測ヘッド（single-horizon用、backward compatibility）"""
    def __init__(self, hidden_size: int, config: DictConfig, output_std: float = 0.01, layer_scale_gamma: float = 0.1):
        super().__init__()
        self.config = config

        # 隠れ層
        layers = []
        current_size = hidden_size
        for hidden_dim in config.architecture.hidden_layers:
            layers.extend([
                nn.Linear(current_size, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.architecture.dropout)
            ])
            current_size = hidden_dim

        # 出力層（改善版初期化）
        quantiles = config.output.quantile_prediction.quantiles
        self.output_layer = nn.Linear(current_size, len(quantiles))

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


class MultiHorizonPredictionHeads(nn.Module):
    """Multi-horizon prediction heads - 各予測期間専用の出力層"""
    def __init__(self, hidden_size: int, config: DictConfig, output_std: float = 0.01, layer_scale_gamma: float = 0.1):
        super().__init__()
        self.config = config

        # 予測対象期間の設定 (新しいconfig構造をサポート)
        if hasattr(config.training, 'prediction') and hasattr(config.training.prediction, 'horizons'):
            self.prediction_horizons = config.training.prediction.horizons
        else:
            # フォールバック: 従来の設定を試す
            self.prediction_horizons = getattr(config.training, 'prediction_horizons', [1, 5, 10, 20])

        # 共有特徴抽出層（各horizonで共有される中間表現）
        self.shared_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.architecture.dropout),
            nn.LayerNorm(hidden_size // 2)
        )

        # 各horizon専用の出力ヘッド
        self.horizon_heads = nn.ModuleDict()
        quantiles = config.output.quantile_prediction.quantiles

        for horizon in self.prediction_horizons:
            # Horizon-specific architecture
            horizon_layers = []
            current_size = hidden_size // 2

            # Horizon専用の隠れ層（短期と長期で異なるarchitecture）
            if horizon <= 5:  # Short-term horizons (1d, 5d)
                # より細かい特徴抽出 - ノイズが重要
                horizon_layers.extend([
                    nn.Linear(current_size, current_size),
                    nn.ReLU(),
                    nn.Dropout(config.architecture.dropout * 0.5),  # Lower dropout for short-term
                ])
            else:  # Long-term horizons (10d, 20d)
                # よりスムーズな特徴抽出 - トレンドが重要
                horizon_layers.extend([
                    nn.Linear(current_size, current_size // 2),
                    nn.ReLU(),
                    nn.Dropout(config.architecture.dropout * 1.5),  # Higher dropout for long-term
                    nn.Linear(current_size // 2, current_size),
                    nn.ReLU(),
                ])

            # 最終出力層
            horizon_layers.append(nn.Linear(current_size, len(quantiles)))

            # Initialize head
            head = nn.Sequential(*horizon_layers)

            # Horizon-specific initialization
            for layer in head:
                if isinstance(layer, nn.Linear):
                    # Short-term: smaller init (less certainty)
                    # Long-term: larger init (more trend confidence)
                    std = output_std * (0.5 if horizon <= 5 else 1.0)
                    nn.init.trunc_normal_(layer.weight, std=std)
                    nn.init.zeros_(layer.bias)

            self.horizon_heads[f'horizon_{horizon}d'] = head

        # Horizon-specific LayerScale parameters
        self.layer_scales = nn.ParameterDict()
        for horizon in self.prediction_horizons:
            # Short-term: smaller scale (less confident)
            # Long-term: larger scale (more trend confident)
            scale = layer_scale_gamma * (0.8 if horizon <= 5 else 1.2)
            self.layer_scales[f'horizon_{horizon}d'] = nn.Parameter(torch.ones(len(quantiles)) * scale)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: [batch_size, sequence_length, hidden_size]

        Returns:
            Dictionary mapping horizon names to predictions
            {
                'horizon_1d': [batch_size, n_quantiles],
                'horizon_5d': [batch_size, n_quantiles],
                'horizon_10d': [batch_size, n_quantiles],
                'horizon_20d': [batch_size, n_quantiles],
            }
        """
        # 最後のタイムステップを使用 (3D input) or そのまま使用 (2D input)
        if x.dim() == 3:
            x = x[:, -1, :]  # [batch_size, hidden_size]
        elif x.dim() == 2:
            pass  # Already [batch_size, hidden_size]
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")

        # 共有エンコーダで中間表現を獲得
        shared_features = self.shared_encoder(x)  # [batch_size, hidden_size//2]

        # 各horizon専用ヘッドで予測
        predictions = {}
        for horizon in self.prediction_horizons:
            horizon_key = f'horizon_{horizon}d'

            # Horizon専用の処理
            horizon_output = self.horizon_heads[horizon_key](shared_features)

            # LayerScale適用
            scaled_output = horizon_output * self.layer_scales[horizon_key]

            predictions[horizon_key] = scaled_output

        return predictions

    def get_single_horizon_prediction(self, x: torch.Tensor, horizon: int) -> torch.Tensor:
        """特定のhorizonの予測のみを取得（効率的）"""
        x = x[:, -1, :]
        shared_features = self.shared_encoder(x)

        horizon_key = f'horizon_{horizon}d'
        if horizon_key not in self.horizon_heads:
            raise ValueError(f"Horizon {horizon}d not supported. Available: {list(self.horizon_heads.keys())}")

        horizon_output = self.horizon_heads[horizon_key](shared_features)
        return horizon_output * self.layer_scales[horizon_key]


class QuantileLoss(nn.Module):
    """Quantile Loss"""
    def __init__(self, quantiles: list[float]):
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
