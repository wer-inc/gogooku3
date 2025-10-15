"""
ATFT-GAT-FAN アーキテクチャ実装
Adaptive Temporal Fusion Transformer with Graph Attention and Frequency Adaptive Normalization
"""
import logging
import re

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig

from ..components import (
    FrequencyAdaptiveNorm,
    FreqDropout1D,
    MultiLayerGAT,
    SliceAdaptiveNorm,
    TemporalFusionTransformer,
    VariableSelectionNetwork,
)
from src.training.curriculum import (
    CurriculumScheduler,
    create_research_curriculum,
    create_simple_curriculum,
)

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

    _TARGET_KEY_PATTERNS = [
        re.compile(r"^(?:return|returns|ret|target|targets|tgt|y)_(\d+)(?:d)?$", re.IGNORECASE),
        re.compile(r"^label_ret_(\d+)_bps$", re.IGNORECASE),
        re.compile(r"^horizon_(\d+)(?:d)?$", re.IGNORECASE),
        re.compile(r"^point_horizon_(\d+)$", re.IGNORECASE),
        re.compile(r"^h(\d+)$", re.IGNORECASE),
        re.compile(r"^(\d+)(?:d)?$", re.IGNORECASE),
    ]

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

        # Curriculum scheduler (optional)
        self.curriculum_scheduler = self._build_curriculum()
        self.curriculum_horizon_weights: dict[int, float] | None = None
        self.curriculum_active_horizons: set[str] | None = None
        self._target_warning_logged = False

    def _canonicalize_target_key(self, key: object) -> str | None:
        """Map various target key aliases to the canonical 'horizon_{n}d' string."""
        if isinstance(key, int):
            return f"horizon_{int(key)}d"
        if isinstance(key, str):
            stripped = key.strip()
            for pattern in self._TARGET_KEY_PATTERNS:
                match = pattern.match(stripped)
                if match:
                    return f"horizon_{int(match.group(1))}d"
        return None

    def _extract_targets(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Collect target tensors keyed by canonical horizon name from the batch."""

        targets_map: dict[str, torch.Tensor] = {}
        if not isinstance(batch, dict):
            return targets_map

        raw_targets = batch.get("targets")
        if isinstance(raw_targets, dict):
            for key, value in raw_targets.items():
                if not torch.is_tensor(value):
                    continue
                canonical = self._canonicalize_target_key(key)
                if canonical is not None:
                    targets_map[canonical] = value

        # Some collate pipelines may already place horizon keys at the top level
        for key, value in batch.items():
            if key == "targets" or not torch.is_tensor(value):
                continue
            canonical = self._canonicalize_target_key(key)
            if canonical is not None:
                targets_map.setdefault(canonical, value)

        return targets_map

    def _fetch_target_tensor(
        self,
        targets_map: dict[str, torch.Tensor],
        horizon_key: str,
        reference: torch.Tensor,
    ) -> torch.Tensor | None:
        """Get a target tensor aligned with the prediction tensor device/dtype."""

        target = targets_map.get(horizon_key)
        if target is None and horizon_key.endswith("d"):
            # Allow lookup without the trailing 'd' if present in the map
            alt_key = horizon_key[:-1]
            target = targets_map.get(alt_key)
        if target is None:
            return None

        target_tensor = target.to(reference.device, dtype=reference.dtype)
        if target_tensor.dim() > 1:
            target_tensor = target_tensor.squeeze(-1)
        return target_tensor

    def _calculate_feature_dims(self):
        """特徴量次元の計算（シンプル版：カテゴリ分割なし）"""
        # カテゴリ分割を完全に無効化し、total_featuresのみを使用
        manual_dims = getattr(self.config.model, "input_dims", None)

        if manual_dims is not None and hasattr(manual_dims, "total_features"):
            total_features = int(manual_dims.total_features)
            historical_override = int(getattr(manual_dims, "historical_features", 0) or 0)

            self.n_current_features = total_features - historical_override
            if self.n_current_features <= 0:
                self.n_current_features = total_features

            self.n_historical_features = historical_override
            self.n_dynamic_features = total_features

            logger.info(
                "✅ Using simplified feature dimensions (category split DISABLED): "
                f"total={total_features}, current={self.n_current_features}, "
                f"historical={self.n_historical_features}"
            )
        else:
            # Fallback: 設定がない場合はエラー
            raise ValueError(
                "model.input_dims.total_features must be specified in config. "
                "Feature category auto-detection has been disabled."
            )

        # 静的特徴量（market_code_nameのエンコーディング後の次元）
        self.n_static_features = 10  # 仮の値（実際はエンコーディング方法による）

    def _build_model(self):
        """モデルアーキテクチャの構築"""

        self.hidden_size = int(getattr(self.config.model, "hidden_size", 128))
        self.gat_output_dim = 0
        self.gat_edge_weight = 0.0
        self.gat_entropy_weight = 0.0

        # Variable Selection Network (feature-wise gating)
        self.variable_selection = self._build_variable_selection()

        # 入力投影層（VSN出力後の調整）
        self.input_projection = self._build_input_projection()

        # Temporal Fusion Transformer backbone
        self.tft = self._build_tft()

        # Graph Attention Network (銘柄間メッセージパッシング)
        self.gat = self._build_gat() if getattr(self.config.model.gat, "enabled", False) else None

        self.combined_feature_dim = self.hidden_size + (self.gat_output_dim if self.gat is not None else 0)
        self.backbone_projection = nn.Linear(self.combined_feature_dim, self.hidden_size)

        # 適応正規化 (FAN/SAN)
        self.adaptive_norm = self._build_adaptive_normalization()

        # 周波数Dropout（オプション）
        self.freq_dropout = self._build_freq_dropout()

        # 予測ヘッド（RegimeMoE / Multi-horizon 等）
        self.prediction_head = self._build_prediction_head()

        # ランタイムで更新する正則化メトリクスの初期化
        self._vsn_sparsity_loss = torch.tensor(0.0)
        self._last_variable_gates: torch.Tensor | None = None
        self._gat_attention_entropy: torch.Tensor | None = None
        self._gat_edge_reg_value: torch.Tensor | None = None
        self._last_attention_weights = None

    def _get_training_config(self):
        if hasattr(self.config, 'training') and self.config.training is not None:
            return self.config.training
        if hasattr(self.config, 'train') and self.config.train is not None:
            return self.config.train
        from omegaconf import OmegaConf
        return OmegaConf.create({})

    def _build_variable_selection(self) -> VariableSelectionNetwork:
        """Variable Selection Networkの構築"""
        # Safely get tft config
        try:
            tft_cfg = getattr(self.config.model, "tft", None)
        except Exception:
            tft_cfg = None

        # Handle None or non-dict values
        if tft_cfg is None or not hasattr(tft_cfg, '__getitem__'):
            vsn_cfg = {}
        else:
            try:
                vsn_cfg = tft_cfg.get("variable_selection", {}) if hasattr(tft_cfg, "get") else tft_cfg.get("variable_selection") if callable(getattr(tft_cfg, "get", None)) else {}
            except Exception:
                vsn_cfg = {}

        dropout = float(getattr(vsn_cfg, "dropout", 0.1) if hasattr(vsn_cfg, "__getattr__") else vsn_cfg.get("dropout", 0.1) if isinstance(vsn_cfg, dict) else 0.1)
        use_sigmoid = bool(getattr(vsn_cfg, "use_sigmoid", True) if hasattr(vsn_cfg, "__getattr__") else vsn_cfg.get("use_sigmoid", True) if isinstance(vsn_cfg, dict) else True)
        sparsity_coeff = float(getattr(vsn_cfg, "sparsity_coefficient", 0.0) if hasattr(vsn_cfg, "__getattr__") else vsn_cfg.get("sparsity_coefficient", 0.0) if isinstance(vsn_cfg, dict) else 0.0)

        self.vsn_sparsity_weight = sparsity_coeff

        return VariableSelectionNetwork(
            input_size=1,
            num_features=self.n_dynamic_features,
            hidden_size=self.hidden_size,
            dropout=dropout,
            use_sigmoid=use_sigmoid,
            sparsity_coefficient=sparsity_coeff,
        )

    def _reconfigure_dynamic_feature_dim(self, new_dim: int, device: torch.device) -> None:
        """Rebuild feature-dependent modules when runtime feature dimension differs."""
        if new_dim <= 0:
            raise ValueError("new_dim must be positive")
        if new_dim == self.n_dynamic_features:
            return
        logger.warning(
            "Dynamic feature dimension mismatch detected (expected %d, got %d). "
            "Rebuilding variable selection network.",
            self.n_dynamic_features,
            new_dim,
        )
        self.n_dynamic_features = new_dim
        self.variable_selection = self._build_variable_selection().to(device)

    def _ensure_backbone_projection(self, input_dim: int, device: torch.device) -> None:
        """Resize backbone projection when combined features dimensionality shifts."""
        if input_dim == self.combined_feature_dim and hasattr(self, "backbone_projection"):
            return
        logger.warning(
            "Adjusting backbone projection input dim from %d to %d",
            getattr(self, "combined_feature_dim", -1),
            input_dim,
        )
        self.combined_feature_dim = input_dim
        self.backbone_projection = nn.Linear(input_dim, self.hidden_size).to(device)

    def _build_input_projection(self) -> nn.Module:
        """VSN後の正規化・Dropout を含む投影層"""
        proj_cfg = getattr(self.config.model, "input_projection", None)
        if proj_cfg is None:
            use_layer_norm = True
            dropout = 0.1
        else:
            use_layer_norm = bool(getattr(proj_cfg, "use_layer_norm", True))
            dropout = float(getattr(proj_cfg, "dropout", 0.1))

        layers: list[nn.Module] = [nn.Linear(self.hidden_size, self.hidden_size)]
        if use_layer_norm:
            layers.append(nn.LayerNorm(self.hidden_size))
        layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def _build_tft(self) -> TemporalFusionTransformer:
        """Temporal Fusion Transformerの構築"""
        tft_cfg = getattr(self.config.model, "tft", {})
        lstm_cfg = getattr(tft_cfg, "lstm", {})
        att_cfg = getattr(tft_cfg, "attention", {})
        temporal_cfg = getattr(tft_cfg, "temporal", {})

        num_layers = int(getattr(lstm_cfg, "layers", 1))
        dropout = float(getattr(lstm_cfg, "dropout", 0.1))
        num_heads = int(getattr(att_cfg, "heads", 4))
        att_dropout = float(getattr(att_cfg, "dropout", dropout))
        use_positional = bool(getattr(temporal_cfg, "use_positional_encoding", True))
        max_seq_len = int(getattr(temporal_cfg, "max_sequence_length", 64))

        return TemporalFusionTransformer(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=max(dropout, att_dropout),
            use_positional_encoding=use_positional,
            max_sequence_length=max_seq_len,
        )

    def _build_gat(self) -> MultiLayerGAT | None:
        """Graph Attention Networkの構築"""
        gat_cfg = getattr(self.config.model, "gat", None)
        if gat_cfg is None or not getattr(gat_cfg, "enabled", False):
            return None

        arch_cfg = getattr(gat_cfg, "architecture", {})
        hidden_channels = list(getattr(arch_cfg, "hidden_channels", [self.hidden_size]))
        heads = list(getattr(arch_cfg, "heads", [4] * len(hidden_channels)))
        concat = list(getattr(arch_cfg, "concat", [True] * len(hidden_channels)))
        num_layers = int(getattr(arch_cfg, "num_layers", len(hidden_channels)))

        layer_cfg = getattr(gat_cfg, "layer_config", {})
        dropout = float(getattr(layer_cfg, "dropout", 0.2))
        edge_dropout = float(getattr(layer_cfg, "edge_dropout", 0.1))
        edge_cfg = getattr(gat_cfg, "edge_features", {})
        edge_dim = int(getattr(edge_cfg, "edge_dim", 0)) or None

        reg_cfg = getattr(gat_cfg, "regularization", {})
        edge_penalty = float(getattr(reg_cfg, "edge_weight_penalty", 0.0))
        entropy_penalty = float(getattr(reg_cfg, "attention_entropy_penalty", 0.0))

        self.gat_output_dim = hidden_channels[-1] * (heads[-1] if concat[-1] else 1)
        self.gat_entropy_weight = entropy_penalty
        self.gat_edge_weight = edge_penalty

        # 🔧 DEBUG (2025-10-06): Log initialized GAT weights
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[GAT-INIT] gat_entropy_weight={self.gat_entropy_weight}, gat_edge_weight={self.gat_edge_weight}, gat_output_dim={self.gat_output_dim}")

        return MultiLayerGAT(
            num_layers=num_layers,
            in_channels=self.hidden_size,
            hidden_channels=hidden_channels,
            heads=heads,
            concat_list=concat,
            dropout=dropout,
            edge_dropout=edge_dropout,
            edge_dim=edge_dim,
            edge_weight_penalty=edge_penalty,
            attention_entropy_penalty=entropy_penalty,
            use_graph_norm=True,
        )

    def _build_adaptive_normalization(self) -> nn.Module:
        """適応正規化層の構築 (FAN -> SAN)"""
        norm_cfg = getattr(self.config.model, "adaptive_normalization", {})
        fan_cfg = getattr(norm_cfg, "fan", {})
        san_cfg = getattr(norm_cfg, "san", {})

        fan_enabled = bool(getattr(fan_cfg, "enabled", True))
        san_enabled = bool(getattr(san_cfg, "enabled", True))

        fan = FrequencyAdaptiveNorm(
            num_features=self.hidden_size,
            window_sizes=list(getattr(fan_cfg, "window_sizes", [5, 10, 20])),
            aggregation=str(getattr(fan_cfg, "aggregation", "weighted_mean")),
            learn_weights=bool(getattr(fan_cfg, "learn_weights", True)),
        ) if fan_enabled else nn.Identity()

        san = SliceAdaptiveNorm(
            num_features=self.hidden_size,
            num_slices=int(getattr(san_cfg, "num_slices", 3)),
            overlap=float(getattr(san_cfg, "overlap", 0.5)),
            slice_aggregation=str(getattr(san_cfg, "slice_aggregation", "learned")),
        ) if san_enabled else nn.Identity()

        return nn.Sequential(fan, san)

    def _build_freq_dropout(self) -> nn.Module | None:
        """周波数領域のDropout設定"""
        # Try to get freq_dropout_p from improvements section first, then from root
        if hasattr(self.config, "improvements") and hasattr(self.config.improvements, "freq_dropout_p"):
            freq_dropout_p = self.config.improvements.freq_dropout_p
        else:
            freq_dropout_p = getattr(self.config, "freq_dropout_p", 0.0)

        # Handle None values
        if freq_dropout_p is None:
            freq_dropout_p = 0.0

        freq_dropout_p = float(freq_dropout_p)
        if freq_dropout_p <= 0:
            return None

        # Try to get min/max width from improvements section first
        if hasattr(self.config, "improvements"):
            min_width = getattr(self.config.improvements, "freq_dropout_min_width", 0.05)
            max_width = getattr(self.config.improvements, "freq_dropout_max_width", 0.2)
        else:
            min_width = getattr(self.config, "freq_dropout_min_width", 0.05)
            max_width = getattr(self.config, "freq_dropout_max_width", 0.2)

        return FreqDropout1D(
            p=freq_dropout_p,
            min_width=float(min_width) if min_width is not None else 0.05,
            max_width=float(max_width) if max_width is not None else 0.2,
        )

    def _build_curriculum(self) -> CurriculumScheduler | None:
        """Build curriculum scheduler if enabled in config."""
        training_cfg = self._get_training_config()
        curriculum_cfg = getattr(training_cfg, "curriculum", None)
        if curriculum_cfg is None or not getattr(curriculum_cfg, "enabled", False):
            return None

        trainer_cfg = getattr(training_cfg, "trainer", None)
        max_epochs = int(getattr(trainer_cfg, "max_epochs", 100))
        profile = str(getattr(curriculum_cfg, "profile", "research")).lower()

        if profile == "simple":
            scheduler = create_simple_curriculum(max_epochs=max_epochs)
        else:
            scheduler = create_research_curriculum(max_epochs=max_epochs)

        logger.info("Curriculum scheduler enabled (%s)", profile)
        return scheduler

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
        training_cfg = self._get_training_config()
        use_multi_horizon = getattr(training_cfg, 'use_multi_horizon_heads', True)
        if use_multi_horizon:
            return MultiHorizonPredictionHeads(
                hidden_size=self.config.model.hidden_size,
                config=self.config.model.prediction_head,
                training_cfg=training_cfg
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

        training_cfg = self._get_training_config()
        loss_cfg = getattr(training_cfg, "loss", None)
        aux_cfg = getattr(loss_cfg, "auxiliary", None) if loss_cfg is not None else None

        # 補助損失
        sharpe_cfg = getattr(aux_cfg, "sharpe_loss", None) if aux_cfg is not None else None
        if getattr(sharpe_cfg, "enabled", False):
            self.sharpe_loss = SharpeLoss(
                weight=float(getattr(sharpe_cfg, "weight", 0.1)),
                min_periods=int(getattr(sharpe_cfg, "min_periods", 20))
            )
        else:
            self.sharpe_loss = None

        # ランキング損失（任意）
        try:
            rk_cfg = getattr(aux_cfg, "ranking_loss", None)
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
            dl_cfg = getattr(aux_cfg, "decision_layer", None)
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
                sched_cfg = {}
                if aux_cfg is not None:
                    sched_cfg = getattr(aux_cfg, 'decision_layer_schedule', {})
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
        if torch.is_tensor(batch):
            batch = {"dynamic_features": batch}
        elif not isinstance(batch, dict):
            raise TypeError(
                "ATFT_GAT_FAN.forward expects a dict batch or feature tensor"
            )

        # 入力データの展開
        dynamic_features = batch.get('dynamic_features')
        if dynamic_features is None and 'features' in batch:
            dynamic_features = batch['features']
        if dynamic_features is None:
            raise KeyError("Batch must contain 'dynamic_features' or 'features'")
        if dynamic_features.dim() == 2:
            dynamic_features = dynamic_features.unsqueeze(0)
        dynamic_features = dynamic_features.to(torch.float32)
        current_feature_dim = dynamic_features.size(-1)
        if current_feature_dim != self.n_dynamic_features:
            self._reconfigure_dynamic_feature_dim(current_feature_dim, dynamic_features.device)
        static_features = batch.get('static_features', None)  # [batch, n_static]
        edge_index = batch.get('edge_index', None)
        edge_attr = batch.get('edge_attr', None)

        # レジーム特徴量（J-UVX + KAMA/VIDYA + market regimes）
        regime_features = batch.get('regime_features', None)  # [batch, regime_dim]

        # Variable Selection Network (feature gating)
        vsn_input = dynamic_features.unsqueeze(-1)
        selected_features, feature_gates, sparsity_loss = self.variable_selection(vsn_input)
        self._vsn_sparsity_loss = sparsity_loss
        self._last_variable_gates = feature_gates.detach()

        # 入力投影
        projected = self.input_projection(selected_features)

        # Temporal Fusion Transformer
        # FIX: Always compute attention (train/eval mode consistency)
        # Loss is only added during training, but forward behavior should be identical
        return_attention = self.gat is not None and self.gat_entropy_weight > 0

        # 🔧 DEBUG (2025-10-07): Log return_attention decision
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"[RETURN-ATT] self.training={self.training}, gat_is_not_none={self.gat is not None}, entropy_weight={self.gat_entropy_weight} → return_attention={return_attention}")

        tft_output, attn_weights = self.tft(
            projected,
            return_attention_weights=return_attention,
        )
        self._last_tft_attention = attn_weights

        # Graph Attention Network（有効な場合）
        gat_features = None
        self._gat_attention_entropy = None
        self._gat_edge_reg_value = None
        if self.gat is not None and edge_index is not None:
            # 🔧 DEBUG (2025-10-06): Log GAT execution
            logger.debug(f"[GAT-EXEC] GAT layer executing with edge_index.shape={edge_index.shape}")

            last_step = tft_output[:, -1, :]  # [batch, hidden]
            if return_attention:
                logger.debug(f"[RETURN-ATT] Taking return_attention=True branch - computing GAT loss metrics")
                gat_emb, attention_weights = self.gat(
                    last_step, edge_index, edge_attr, return_attention_weights=True
                )
                self._gat_attention_entropy = self.gat.get_attention_entropy(attention_weights)
                # Use final layer attention weights for edge regularization proxy
                _, att_tensor = attention_weights[-1]
                self._gat_edge_reg_value = att_tensor.pow(2).mean()
                self._last_attention_weights = attention_weights
                logger.debug(f"[RETURN-ATT] Set _gat_attention_entropy={self._gat_attention_entropy.item():.6f}, _gat_edge_reg_value={self._gat_edge_reg_value.item():.6f}")
            else:
                logger.debug(f"[RETURN-ATT] Taking return_attention=False branch - GAT loss metrics will be None")
                gat_emb = self.gat(last_step, edge_index, edge_attr)
                self._last_attention_weights = None

            gat_features = gat_emb.unsqueeze(1).expand(-1, tft_output.size(1), -1)
            logger.debug(f"[GAT-EXEC] GAT output shape={gat_emb.shape}, expanded={gat_features.shape}")
            # 🔧 DEBUG (2025-10-06): Verify gat_features status
            logger.debug(f"[GAT-DEBUG] gat_features is None: {gat_features is None}")
            if gat_features is not None:
                logger.debug(f"[GAT-DEBUG] gat_features.shape={gat_features.shape}, requires_grad={gat_features.requires_grad}")

        # Combine temporal and graph context
        # 🔧 FIX (2025-10-06): Always use consistent dimensions to prevent backbone_projection recreation
        # When GAT is skipped, pad with zeros to maintain dimension=hidden_size+gat_output_dim
        # 🔧 DEBUG (2025-10-06): Log which branch is taken
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"[GAT-DEBUG] Checking concatenation: gat_features is not None = {gat_features is not None}")
        if gat_features is not None:
            logger.debug(f"[GAT-DEBUG] Using GAT features branch, combined shape will be {tft_output.size()[-1] + gat_features.size()[-1]}")
            combined_features = torch.cat([tft_output, gat_features], dim=-1)
        else:
            logger.debug(f"[GAT-DEBUG] Using zero-padding branch")
            # GAT disabled or no edges: pad with zeros to match expected dimension
            if self.gat is not None:
                # GAT exists but not executed: use gat_output_dim for padding
                zero_pad = torch.zeros(
                    tft_output.size(0), tft_output.size(1), self.gat_output_dim,
                    device=tft_output.device, dtype=tft_output.dtype
                )
                combined_features = torch.cat([tft_output, zero_pad], dim=-1)
            else:
                # GAT completely disabled: no padding needed
                combined_features = tft_output

        # FreqDropout適用（オプション）
        if self.freq_dropout is not None and self.training:
            combined_features = self.freq_dropout(combined_features)

        # Project back to hidden size and apply adaptive normalization
        # 🔧 FIX (2025-10-06): No longer need dynamic dimension check - dimensions are now fixed
        combined_features = self.backbone_projection(combined_features)
        normalized_features = self.adaptive_norm(combined_features)
        # 🔧 DEBUG (2025-10-06): Check gradient flow
        logger.debug(f"[GAT-DEBUG] combined_features.requires_grad={combined_features.requires_grad}, normalized.requires_grad={normalized_features.requires_grad}")

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

    def on_train_epoch_start(self) -> None:
        if self.curriculum_scheduler is None:
            return

        phase = self.curriculum_scheduler.get_phase_config(self.current_epoch)
        self.curriculum_horizon_weights = phase.horizon_weights
        self.curriculum_active_horizons = {f'horizon_{h}d' for h in phase.prediction_horizons}
        self.log('curriculum/phase', phase.name, prog_bar=False, on_step=False, on_epoch=True)

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
        targets_map = self._extract_targets(batch)
        if not targets_map and not self._target_warning_logged:
            logger.warning("No target tensors were found in batch; training loss will be zero.")
            self._target_warning_logged = True

        total_loss = 0.0
        horizon_losses = {}

        if output_type == 'multi_horizon':
            # Multi-horizon training: 各horizonでの損失計算
            # Horizon weights (curriculum優先)
            if self.curriculum_horizon_weights:
                horizon_weights = {f'horizon_{h}d': w for h, w in self.curriculum_horizon_weights.items()}
            elif hasattr(self.config.training, 'prediction') and hasattr(self.config.training.prediction, 'horizons'):
                horizons_cfg = list(self.config.training.prediction.horizons)
                weight_list = list(getattr(self.config.training.prediction, 'horizon_weights', [1.0] * len(horizons_cfg)))
                horizon_weights = {f'horizon_{h}d': w for h, w in zip(horizons_cfg, weight_list)}
            else:
                # フォールバック: 従来の設定またはデフォルト
                horizon_weights = getattr(self.config.training, 'horizon_weights', {
                    'horizon_1d': 1.0, 'horizon_5d': 0.8, 'horizon_10d': 0.6, 'horizon_20d': 0.4
                })

            for horizon_key, pred in predictions.items():
                if self.curriculum_active_horizons and horizon_key not in self.curriculum_active_horizons:
                    continue

                target_tensor = self._fetch_target_tensor(targets_map, horizon_key, pred)
                if target_tensor is None:
                    continue

                # Horizon-specific loss
                horizon_loss = self.quantile_loss(pred, target_tensor)

                # Apply horizon weighting (emphasize short-term)
                weight = horizon_weights.get(horizon_key, 0.5)
                weighted_loss = horizon_loss * weight

                total_loss += weighted_loss
                horizon_losses[f'train_loss_{horizon_key}'] = horizon_loss

                # Log individual horizon losses
                self.log(f'train_loss_{horizon_key}', horizon_loss, prog_bar=False)

        else:
            # Single-horizon training (backward compatibility)
            target_tensor = next(iter(targets_map.values()), None)
            if target_tensor is None:
                raise RuntimeError("No target tensor available for single-horizon training")
            target_tensor = target_tensor.to(predictions['single'].device, dtype=predictions['single'].dtype)
            if target_tensor.dim() > 1:
                target_tensor = target_tensor.squeeze(-1)
            main_loss = self.quantile_loss(predictions['single'], target_tensor)
            total_loss = main_loss
            self.log('train_main_loss', main_loss, prog_bar=True)

        # Auxiliary losses (applied to all horizons)
        if hasattr(self, 'sharpe_loss') and output_type == 'multi_horizon':
            # Apply Sharpe loss to primary horizon (usually 1d or 5d)
            primary_horizon = getattr(self.config.training, 'primary_horizon', 'horizon_1d')
            if primary_horizon in predictions:
                target_tensor = self._fetch_target_tensor(targets_map, primary_horizon, predictions[primary_horizon])
                if target_tensor is not None:
                    sharpe_loss = self.sharpe_loss(predictions[primary_horizon], target_tensor)
                    total_loss = total_loss + sharpe_loss
                    self.log('train_sharpe_loss', sharpe_loss, prog_bar=False)

        # Rank損失（主ホライズンの中央値スコアでペアワイズ）
        if output_type == 'multi_horizon' and self.rank_loss is not None and self.rank_loss_weight > 0:
            primary_horizon = getattr(self.config.training, 'primary_horizon', 'horizon_1d')
            if primary_horizon in predictions:
                pred_q = predictions[primary_horizon]
                target_tensor = self._fetch_target_tensor(targets_map, primary_horizon, pred_q)
                if target_tensor is not None and pred_q.dim() == 2 and pred_q.size(1) > self._median_q_idx:
                    z = pred_q[:, self._median_q_idx]
                    y = target_tensor.view(-1)
                    rk = self.rank_loss(z.view(-1), y) * self.rank_loss_weight
                    total_loss = total_loss + rk
                    self.log('train_rank_loss', rk.detach(), prog_bar=False)

        # 意思決定層ロス（主ホライズンの分位点）
        if output_type == 'multi_horizon' and self.decision_layer is not None:
            primary_horizon = getattr(self.config.training, 'primary_horizon', 'horizon_1d')
            if primary_horizon in predictions:
                q = predictions[primary_horizon]
                target_tensor = self._fetch_target_tensor(targets_map, primary_horizon, q)
                if target_tensor is not None:
                    dl_total, comps = self.decision_layer(q, target_tensor.view(-1))
                    total_loss = total_loss + dl_total
                    # 重要メトリクスをログ
                    self.log('train_decision_sharpe', comps['decision_sharpe'], prog_bar=False)
                    self.log('train_pos_l2', comps['decision_pos_l2'], prog_bar=False)
                    self.log('train_fee', comps['decision_fee'], prog_bar=False)

        # Variable selection sparsity正則化
        if isinstance(self._vsn_sparsity_loss, torch.Tensor) and self._vsn_sparsity_loss.numel() > 0:
            total_loss = total_loss + self._vsn_sparsity_loss
            self.log('train_vsn_sparsity', self._vsn_sparsity_loss.detach(), prog_bar=False)

        # GAT正則化 (edge weight / attention entropy)
        if self.gat is not None:
            # 🔧 DEBUG (2025-10-07): Log condition checks
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"[GAT-LOSS-CHECK] edge_reg_value is None: {self._gat_edge_reg_value is None}, is Tensor: {isinstance(self._gat_edge_reg_value, torch.Tensor) if self._gat_edge_reg_value is not None else 'N/A'}, weight > 0: {self.gat_edge_weight > 0}")
            logger.debug(f"[GAT-LOSS-CHECK] attention_entropy is None: {self._gat_attention_entropy is None}, is Tensor: {isinstance(self._gat_attention_entropy, torch.Tensor) if self._gat_attention_entropy is not None else 'N/A'}, weight > 0: {self.gat_entropy_weight > 0}")

            if (self._gat_edge_reg_value is not None and isinstance(self._gat_edge_reg_value, torch.Tensor)
                    and self.gat_edge_weight > 0):
                edge_reg = self.gat_edge_weight * self._gat_edge_reg_value
                total_loss = total_loss + edge_reg
                self.log('train_gat_edge_penalty', self._gat_edge_reg_value.detach(), prog_bar=False)
                # 🔧 DEBUG (2025-10-06): Log GAT edge regularization
                logger.debug(f"[GAT-LOSS] Adding edge_reg={edge_reg.item():.6f} (weight={self.gat_edge_weight})")
            else:
                logger.debug(f"[GAT-LOSS] Skipping edge_reg (condition not met)")

            if (self._gat_attention_entropy is not None and isinstance(self._gat_attention_entropy, torch.Tensor)
                    and self.gat_entropy_weight > 0):
                entropy_reg = -self.gat_entropy_weight * self._gat_attention_entropy
                total_loss = total_loss + entropy_reg
                self.log('train_gat_entropy', self._gat_attention_entropy.detach(), prog_bar=False)
                # 🔧 DEBUG (2025-10-06): Log GAT entropy regularization
                logger.debug(f"[GAT-LOSS] Adding entropy_reg={entropy_reg.item():.6f} (weight={self.gat_entropy_weight})")
            else:
                logger.debug(f"[GAT-LOSS] Skipping entropy_reg (condition not met)")

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
        targets_map = self._extract_targets(batch)

        total_loss = 0.0

        if output_type == 'multi_horizon':
            # Multi-horizon validation
            for horizon_key, pred in predictions.items():
                if self.curriculum_active_horizons and horizon_key not in self.curriculum_active_horizons:
                    continue
                target_tensor = self._fetch_target_tensor(targets_map, horizon_key, pred)
                if target_tensor is None:
                    continue

                val_loss = self.quantile_loss(pred, target_tensor)
                total_loss += val_loss

                # Log individual horizon validation losses
                self.log(f'val_loss_{horizon_key}', val_loss, prog_bar=False, sync_dist=True)

        else:
            # Single-horizon validation
            target_tensor = next(iter(targets_map.values()), None)
            if target_tensor is None:
                raise RuntimeError("No target tensor available for single-horizon validation")
            target_tensor = target_tensor.to(predictions['single'].device, dtype=predictions['single'].dtype)
            if target_tensor.dim() > 1:
                target_tensor = target_tensor.squeeze(-1)
            val_loss = self.quantile_loss(predictions['single'], target_tensor)
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
        training_cfg = self._get_training_config()
        optimizer_config = getattr(training_cfg, "optimizer", None)

        opt_type = str(getattr(optimizer_config, "type", "AdamW")) if optimizer_config is not None else "AdamW"
        lr = float(getattr(optimizer_config, "lr", 1e-3)) if optimizer_config is not None else 1e-3
        weight_decay = float(getattr(optimizer_config, "weight_decay", 0.0)) if optimizer_config is not None else 0.0
        betas = getattr(optimizer_config, "betas", (0.9, 0.999)) if optimizer_config is not None else (0.9, 0.999)
        eps = float(getattr(optimizer_config, "eps", 1e-8)) if optimizer_config is not None else 1e-8

        if opt_type.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=tuple(betas),
                eps=eps
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_type}")

        # スケジューラー設定
        scheduler_config = getattr(training_cfg, "scheduler", None)

        if scheduler_config is not None and getattr(scheduler_config, 'type', '') == 'CosineAnnealingWarmRestarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=int(getattr(scheduler_config, 'T_0', 10)),
                T_mult=int(getattr(scheduler_config, 'T_mult', 1)),
                eta_min=float(getattr(scheduler_config, 'eta_min', 1e-6))
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
    def __init__(
        self,
        hidden_size: int,
        config: DictConfig,
        training_cfg: DictConfig | None = None,
        output_std: float = 0.01,
        layer_scale_gamma: float = 0.1,
    ):
        super().__init__()
        self.config = config

        # 予測対象期間の設定 (新しいconfig構造をサポート)
        self._training_cfg = training_cfg or getattr(config, 'training', None)

        def _extract_horizons(cfg: DictConfig | None) -> list[int] | None:
            if cfg is None:
                return None
            try:
                if hasattr(cfg, 'prediction') and hasattr(cfg.prediction, 'horizons'):
                    return list(int(h) for h in cfg.prediction.horizons)
                if hasattr(cfg, 'prediction_horizons'):
                    return list(int(h) for h in cfg.prediction_horizons)
            except Exception:
                return None
            return None

        horizons = _extract_horizons(self._training_cfg)
        if horizons is None:
            horizons = _extract_horizons(getattr(config, 'training', None))
        if horizons is None and hasattr(config, 'prediction_horizons'):
            try:
                horizons = list(int(h) for h in config.prediction_horizons)
            except Exception:
                horizons = None
        if horizons is None:
            horizons = [1, 5, 10, 20]

        self.prediction_horizons = horizons

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
