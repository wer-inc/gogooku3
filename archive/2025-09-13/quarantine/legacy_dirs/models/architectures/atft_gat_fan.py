"""ATFT-GAT-FAN: Main model architecture."""

import logging
from typing import Dict, Optional
from types import SimpleNamespace

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch_geometric.utils import dropout_edge as _pyg_dropout_edge
import os as _os

from ..components import (
    FrequencyAdaptiveNorm,
    SliceAdaptiveNorm,
    TemporalFusionTransformer,
    VariableSelectionNetwork,
)
from ..components.gat_layer import MultiLayerGAT
from ...graph.dynamic_knn import build_knn_from_embeddings

logger = logging.getLogger(__name__)


class ATFT_GAT_FAN(nn.Module):
    """Adaptive Temporal Fusion Transformer with Graph Attention and Frequency Adaptive Normalization."""

    def __init__(self, config: DictConfig):
        """Initialize ATFT-GAT-FAN model.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        # Optional overrides controlled by trainer
        self._knn_k_override: int | None = None
        try:
            self._edge_input_dropout_p: float = float(
                _os.getenv("EDGE_DROPOUT_INPUT_P", "0.0")
            )
        except Exception:
            self._edge_input_dropout_p = 0.0

        # Fusion Alpha: controls GAT branch contribution (0=TFT only, 1=GAT only)
        # Load from config or use defaults, with environment variable override
        gat_config = getattr(self.config.model, "gat", {})

        # Environment variable override (highest priority)
        alpha_init = float(_os.getenv("GAT_ALPHA_INIT", "0.2"))
        alpha_min = float(_os.getenv("GAT_ALPHA_MIN", "0.30"))
        alpha_penalty = float(_os.getenv("GAT_ALPHA_PENALTY", "1e-4"))

        # Config override (if not set via env vars)
        if alpha_init == 0.2 and hasattr(gat_config, "alpha_init"):
            alpha_init = float(gat_config.alpha_init)

        if alpha_min == 0.30 and hasattr(gat_config, "alpha_min"):
            alpha_min = float(gat_config.alpha_min)

        if alpha_penalty == 1e-4 and hasattr(gat_config, "alpha_penalty"):
            alpha_penalty = float(gat_config.alpha_penalty)

        self.alpha_graph_min = alpha_min
        self.alpha_penalty = alpha_penalty

        # Log the actual values being used
        logger.info(
            f"GAT Alpha settings: alpha_init={alpha_init:.3f}, alpha_min={self.alpha_graph_min:.3f}, alpha_penalty={self.alpha_penalty}"
        )

        # Initialize alpha_logit from target alpha using inverse sigmoid
        import numpy as np

        alpha_prime = max(
            1e-3,
            min(
                0.999,
                (alpha_init - self.alpha_graph_min) / (1.0 - self.alpha_graph_min),
            ),
        )
        logit_init = np.log(alpha_prime) - np.log(1.0 - alpha_prime)
        self.alpha_logit = nn.Parameter(
            torch.tensor(logit_init, dtype=torch.float32)
        )  # Learnable fusion weight

        # Calculate feature dimensions
        self._calculate_feature_dims()

        # Build model components
        self._build_input_projection()
        self._build_adaptive_normalization()
        self._build_tft()
        self._build_gat()
        self._build_prediction_head()

        # Initialize weights
        self._init_weights()

    def _calculate_feature_dims(self) -> None:
        """Calculate feature dimensions from config."""
        data_config = self.config.data.features

        # Parquetデータの場合、固定次元を使用
        if hasattr(data_config, "input_dim"):
            self.n_dynamic_features = data_config.input_dim
            self.n_current_features = data_config.input_dim
            self.n_historical_features = 0
            self.n_static_features = 10
            logger.info(f"Using fixed input dimension: {self.n_dynamic_features}")
            return

        # 従来の詳細な特徴量計算
        # 基本特徴量
        n_basic = len(data_config.basic.price_volume) + len(data_config.basic.flags)

        # テクニカル指標
        n_technical = (
            len(data_config.technical.momentum)
            + len(data_config.technical.volatility)
            + len(data_config.technical.trend)
            + len(data_config.technical.moving_averages)
            + len(data_config.technical.macd)
        )

        # リターン特徴量
        n_returns = len(data_config.returns.columns)

        # 現在値特徴量の合計
        self.n_current_features = n_basic + n_technical + n_returns

        # 履歴特徴量の計算
        n_historical = 0
        for hist_name, hist_config in data_config.historical.items():
            n_historical += hist_config.range[1] - hist_config.range[0] + 1

        self.n_historical_features = n_historical

        # 合計特徴量数
        self.n_dynamic_features = self.n_current_features + self.n_historical_features

        # 静的特徴量（market_code_nameのエンコーディング後の次元）
        self.n_static_features = 10  # 仮の値（実際はエンコーディング方法による）

        logger.info(
            f"Feature dimensions - Current: {self.n_current_features}, "
            f"Historical: {self.n_historical_features}, "
            f"Total: {self.n_dynamic_features}"
        )

    def _build_input_projection(self) -> None:
        """Build input projection layers."""
        model_config = self.config.model
        # Handle nested model config structure
        # Prefer flat `model.hidden_size`; fallback to nested or `hidden_dim`; default to 128
        hidden_size = None
        for path in (
            ("hidden_size",),
            ("model", "hidden_size"),
            ("hidden_dim",),
            ("model", "hidden_dim"),
        ):
            try:
                cur = model_config
                for k in path:
                    cur = getattr(cur, k)
                if cur is not None:
                    hidden_size = int(cur)
                    break
            except Exception:
                continue
        if hidden_size is None:
            hidden_size = 128
            logger.warning("hidden_size missing; falling back to 128")
        self.hidden_size = hidden_size

        # Dynamic features projection
        self.dynamic_projection = nn.Linear(self.n_dynamic_features, hidden_size)

        # Layer norm and dropout
        # Support both nested (model.input_projection) and flat (input_projection) configurations
        if hasattr(model_config, "model") and hasattr(
            model_config.model, "input_projection"
        ):
            input_proj_config = model_config.model.input_projection
        elif hasattr(model_config, "input_projection"):
            input_proj_config = model_config.input_projection
        else:
            # Safe defaults if not provided in config
            class _InputProjDefaults:
                use_layer_norm = True
                dropout = 0.1

            input_proj_config = _InputProjDefaults()
            logger.warning(
                "input_projection config missing; using defaults {use_layer_norm=True, dropout=0.1}"
            )

        if input_proj_config.use_layer_norm:
            self.input_layer_norm = nn.LayerNorm(hidden_size)
        else:
            self.input_layer_norm = nn.Identity()

        self.input_dropout = nn.Dropout(input_proj_config.dropout)

    def _build_adaptive_normalization(self) -> None:
        """Build adaptive normalization layers."""
        # Fallback defaults when config is missing - FAN/SAN enabled by default
        default_norm = SimpleNamespace(
            fan=SimpleNamespace(
                enabled=True,  # Changed to True for A+ approach
                window_sizes=[5, 10, 20],
                aggregation="weighted_mean",
                learn_weights=True,
            ),
            san=SimpleNamespace(
                enabled=True,  # Changed to True for A+ approach
                num_slices=3,  # Increased from 1 to 3
                overlap=0.5,  # Increased from 0.0 to 0.5
                slice_aggregation="mean",
            ),
        )
        norm_config = getattr(self.config.model, "adaptive_normalization", None)
        used_defaults = False
        if norm_config is None:
            norm_config = default_norm
            used_defaults = True
        if used_defaults:
            logger.warning(
                "adaptive_normalization config missing; enabling FAN and SAN by default"
            )
        hidden_size = self.hidden_size  # Use stored hidden_size

        # Frequency Adaptive Normalization
        if norm_config.fan.enabled:
            self.fan = FrequencyAdaptiveNorm(
                num_features=hidden_size,
                window_sizes=norm_config.fan.window_sizes,
                aggregation=norm_config.fan.aggregation,
                learn_weights=norm_config.fan.learn_weights,
            )
        else:
            self.fan = nn.Identity()

        # Slice Adaptive Normalization
        if norm_config.san.enabled:
            self.san = SliceAdaptiveNorm(
                num_features=hidden_size,
                num_slices=norm_config.san.num_slices,
                overlap=norm_config.san.overlap,
                slice_aggregation=norm_config.san.slice_aggregation,
            )
        else:
            self.san = nn.Identity()

    def _build_tft(self) -> None:
        """Build Temporal Fusion Transformer."""
        tft_defaults = SimpleNamespace(
            variable_selection=SimpleNamespace(
                dropout=0.1, use_sigmoid=True, sparsity_coefficient=0.0
            ),
            attention=SimpleNamespace(heads=4),
            lstm=SimpleNamespace(layers=1, dropout=0.1),
            temporal=SimpleNamespace(
                use_positional_encoding=True, max_sequence_length=20
            ),
        )
        tft_config = getattr(self.config.model, "tft", None)
        used_defaults = False
        if tft_config is None:
            tft_config = tft_defaults
            used_defaults = True
        if used_defaults:
            logger.warning("tft config missing; using lightweight default TFT settings")
        hidden_size = self.hidden_size  # Use stored hidden_size

        # Variable Selection Network
        self.vsn = VariableSelectionNetwork(
            input_size=1,  # Each feature is univariate
            num_features=self.n_dynamic_features,
            hidden_size=hidden_size,
            dropout=tft_config.variable_selection.dropout,
            use_sigmoid=tft_config.variable_selection.use_sigmoid,
            sparsity_coefficient=tft_config.variable_selection.sparsity_coefficient,
        )

        # Temporal Fusion Transformer
        self.tft = TemporalFusionTransformer(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_heads=tft_config.attention.heads,
            num_layers=tft_config.lstm.layers,
            dropout=tft_config.lstm.dropout,
            use_positional_encoding=tft_config.temporal.use_positional_encoding,
            max_sequence_length=tft_config.temporal.max_sequence_length,
        )

    def _build_gat(self) -> None:
        """Build Graph Attention Network."""
        gat_defaults = SimpleNamespace(
            enabled=False,
            architecture=SimpleNamespace(
                num_layers=2,
                hidden_channels=[self.hidden_size, self.hidden_size],
                heads=[2, 2],
                concat=[True, False],
            ),
            layer_config=SimpleNamespace(dropout=0.0, edge_dropout=0.0),
            edge_features=SimpleNamespace(use_edge_attr=False, edge_dim=None),
            regularization=SimpleNamespace(
                edge_weight_penalty=0.0, attention_entropy_penalty=0.0
            ),
        )
        gat_config = getattr(self.config.model, "gat", None)
        used_defaults = False
        if gat_config is None:
            gat_config = gat_defaults
            used_defaults = True
        if used_defaults:
            logger.warning("gat config missing; GAT disabled by default")

        if gat_config.enabled:
            # Check for GraphNorm configuration
            use_graph_norm = getattr(gat_config.layer_config, "use_graph_norm", True)
            graph_norm_type = getattr(
                gat_config.layer_config, "graph_norm_type", "graph"
            )

            self.gat = MultiLayerGAT(
                num_layers=gat_config.architecture.num_layers,
                in_channels=self.hidden_size,
                hidden_channels=gat_config.architecture.hidden_channels,
                heads=gat_config.architecture.heads,
                concat_list=gat_config.architecture.concat,
                dropout=gat_config.layer_config.dropout,
                edge_dropout=gat_config.layer_config.edge_dropout,
                edge_dim=gat_config.edge_features.edge_dim
                if gat_config.edge_features.use_edge_attr
                else None,
                edge_weight_penalty=gat_config.regularization.edge_weight_penalty,
                attention_entropy_penalty=gat_config.regularization.attention_entropy_penalty,
                use_graph_norm=use_graph_norm,
                graph_norm_type=graph_norm_type,
            )
        else:
            self.gat = None

    # ===== Public controls for trainer =====
    def set_knn_k(self, k: int | None) -> None:
        """Override K for dynamic KNN graph construction. Set None to disable override."""
        if k is None:
            self._knn_k_override = None
            return
        try:
            k_int = int(k)
            self._knn_k_override = max(1, k_int)
        except Exception:
            self._knn_k_override = None

    def _build_prediction_head(self) -> None:
        """Build prediction head."""
        pred_defaults = SimpleNamespace(
            architecture=SimpleNamespace(hidden_layers=[], dropout=0.0),
            output=SimpleNamespace(
                point_prediction=True,
                quantile_prediction=SimpleNamespace(
                    enabled=False, quantiles=[0.1, 0.5, 0.9]
                ),
            ),
        )
        pred_config = getattr(self.config.model, "prediction_head", None)
        used_defaults = False
        if pred_config is None:
            pred_config = pred_defaults
            used_defaults = True
        if used_defaults:
            logger.warning(
                "prediction_head config missing; using default point-prediction head"
            )
        hidden_size = self.hidden_size  # Use stored hidden_size

        # Build MLP layers
        layers = []
        in_features = hidden_size

        # Always add at least one hidden layer to prevent identity mapping
        hidden_layers = pred_config.architecture.hidden_layers
        if not hidden_layers:
            hidden_layers = [hidden_size // 2]  # Default to half the hidden size

        for out_features in hidden_layers:
            layers.append(nn.Linear(in_features, out_features))
            # Use GELU instead of ReLU to avoid dying ReLU problem
            layers.append(nn.GELU())
            layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.Dropout(pred_config.architecture.dropout))
            in_features = out_features

        self.prediction_mlp = nn.Sequential(*layers)

        # Output heads
        # Dropout before output heads to maintain variance
        self.output_dropout = nn.Dropout(0.1)

        self.output_heads = nn.ModuleDict()
        # Optional probabilistic heads
        self.t_heads = nn.ModuleDict()
        # Optional direction classification heads (per horizon)
        self.dir_heads = nn.ModuleDict()

        # Point prediction head
        if pred_config.output.point_prediction:
            for horizon in self.config.data.time_series.prediction_horizons:
                self.output_heads[f"point_horizon_{horizon}"] = nn.Linear(
                    in_features, 1
                )

        # Student-t parameter heads (mu, sigma_raw, nu_raw) — enabled via config if present, otherwise off by default
        try:
            use_t_head = getattr(pred_config.output, "student_t", False)
        except Exception:
            use_t_head = False
        if use_t_head:
            for horizon in self.config.data.time_series.prediction_horizons:
                self.t_heads[f"t_params_horizon_{horizon}"] = nn.Linear(in_features, 3)

        # Quantile heads
        try:
            q_cfg = getattr(pred_config.output, "quantile_prediction", None)
            use_q = bool(q_cfg.enabled) if q_cfg is not None else False
            quantiles = (
                list(q_cfg.quantiles)
                if (q_cfg is not None and hasattr(q_cfg, "quantiles"))
                else []
            )
        except Exception:
            use_q = False
            quantiles = []
        if use_q and len(quantiles) > 0:
            self.quantiles = quantiles
            for horizon in self.config.data.time_series.prediction_horizons:
                self.output_heads[f"quantile_horizon_{horizon}"] = nn.Linear(
                    in_features, len(quantiles)
                )

        # Quantile prediction head
        if pred_config.output.quantile_prediction.enabled:
            n_quantiles = len(pred_config.output.quantile_prediction.quantiles)
            for horizon in self.config.data.time_series.prediction_horizons:
                self.output_heads[f"quantile_horizon_{horizon}"] = nn.Linear(
                    in_features, n_quantiles
                )

        # Direction classification heads (BCE with logits)
        try:
            use_dir = bool(getattr(pred_config.output, "direction", False))
        except Exception:
            use_dir = False
        # 既定: 環境変数 ENABLE_DIRECTION=1 ならON
        import os as _os

        if _os.getenv("ENABLE_DIRECTION", "1") == "1":
            use_dir = True or use_dir
        if use_dir:
            for horizon in self.config.data.time_series.prediction_horizons:
                self.dir_heads[f"direction_horizon_{horizon}"] = nn.Linear(
                    in_features, 1
                )

    def _init_weights(self) -> None:
        """Initialize model weights."""
        init_defaults = SimpleNamespace(method="xavier_uniform", gain=1.0)
        init_config = getattr(self.config.model, "initialization", None)
        if init_config is None:
            init_config = init_defaults
            logger.warning(
                "initialization config missing; using xavier_uniform by default"
            )

        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                # Special initialization for output heads to prevent collapse
                if "output_heads" in name or "prediction_mlp" in name:
                    # Use LARGER initialization for output heads to escape constant prediction
                    if "output_heads" in name and "point_horizon" in name:
                        # Much larger initialization for point predictions
                        nn.init.normal_(param, mean=0.0, std=0.1)
                        logger.info(f"Initialized {name} with larger std=0.1")
                    elif init_config.method == "xavier_uniform":
                        nn.init.xavier_uniform_(param, gain=init_config.gain * 2.0)
                    elif init_config.method == "xavier_normal":
                        nn.init.xavier_normal_(param, gain=init_config.gain * 2.0)
                    else:
                        nn.init.normal_(param, mean=0.0, std=0.05)
                else:
                    # Standard initialization for other layers
                    if init_config.method == "xavier_uniform":
                        nn.init.xavier_uniform_(param, gain=init_config.gain)
                    elif init_config.method == "xavier_normal":
                        nn.init.xavier_normal_(param, gain=init_config.gain)
                    elif init_config.method == "kaiming_uniform":
                        nn.init.kaiming_uniform_(param, nonlinearity="relu")
                    elif init_config.method == "kaiming_normal":
                        nn.init.kaiming_normal_(param, nonlinearity="relu")
            elif "bias" in name:
                # Random bias for output heads to prevent collapse
                if "output_heads" in name or "t_heads" in name or "dir_heads" in name:
                    # Larger random bias to break symmetry and prevent collapse
                    nn.init.uniform_(param, -0.1, 0.1)
                else:
                    nn.init.constant_(param, 0.0)

        # LSTM specific initialization
        for module in self.modules():
            if isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "weight_ih" in name:
                        # Input-hidden weights
                        nn.init.xavier_uniform_(param)
                    elif "weight_hh" in name:
                        # Hidden-hidden weights
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        # Biases: set forget gate bias to 1
                        n = param.size(0)
                        param.data.fill_(0.0)
                        param.data[n // 4 : n // 2].fill_(1.0)  # Forget gate bias

    def forward(
        self,
        dynamic_features: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            dynamic_features: Dynamic features of shape (batch, seq_len, n_features)
            edge_index: Edge indices for GAT (optional)
            edge_attr: Edge attributes for GAT (optional)
            return_attention_weights: Whether to return attention weights

        Returns:
            Dictionary containing predictions and auxiliary outputs
        """
        batch_size, seq_len, n_features = dynamic_features.shape

        # Input projection
        x = self.dynamic_projection(dynamic_features)
        x = self.input_layer_norm(x)
        x = self.input_dropout(x)

        # Adaptive normalization (FAN) on projected features
        x = self.fan(x)

        # Variable selection (prepare for VSN)
        # VSN expects (batch, seq_len, num_features, feature_size)
        # We treat each feature as univariate
        # Keep default contiguous (3D -> 4D) without channels_last to avoid rank mismatch
        features_expanded = dynamic_features.unsqueeze(-1).contiguous()
        selected_features, feature_weights, sparsity_loss = self.vsn(features_expanded)

        # Combine projected features with selected features
        # This maintains the input projection information while incorporating VSN selection
        x = x + selected_features

        # TFT processing
        # Reduce activation memory via checkpointing if enabled in config
        use_ckpt = False
        try:
            use_ckpt = bool(
                getattr(self.config.model, "optimization").get(
                    "gradient_checkpointing", False
                )
            )
        except Exception:
            use_ckpt = False
        if use_ckpt and self.training:
            from torch.utils.checkpoint import checkpoint

            tft_out, attention_weights = checkpoint(
                lambda y: self.tft(y, return_attention_weights), x, use_reentrant=False
            )
        else:
            tft_out, attention_weights = self.tft(x, return_attention_weights)

        # Adaptive normalization (SAN)
        tft_out = self.san(tft_out)

        gat_attention_weights = None
        # Take the last valid time step representation
        # SAN may zero out the last few timesteps, so we check for a valid one
        last_repr = None
        for t in range(tft_out.size(1) - 1, -1, -1):
            timestep = tft_out[:, t, :]
            if not (timestep == 0).all():
                last_repr = timestep
                break

        # Fallback to last timestep if all are zero (shouldn't happen)
        if last_repr is None:
            last_repr = tft_out[:, -1, :]
        final_repr = last_repr

        # GAT processing (if enabled). If no edges were provided, optionally build dynamic KNN.
        if self.gat is not None:
            use_dyn = True
            knn_k = 15
            use_edge_attr = False
            try:
                use_dyn = bool(getattr(self.config.model.gat, "dynamic_knn", True))
            except Exception:
                use_dyn = True
            # Determine knn_k with optional trainer override
            try:
                cfg_k = int(getattr(self.config.model.gat, "knn_k", 15))
            except Exception:
                cfg_k = 15
            knn_k = (
                int(self._knn_k_override)
                if (self._knn_k_override is not None)
                else cfg_k
            )
            try:
                use_edge_attr = bool(
                    getattr(self.config.model.gat, "edge_features").use_edge_attr
                )
            except Exception:
                use_edge_attr = False

            eidx, eattr = edge_index, edge_attr
            if eidx is None and use_dyn:
                try:
                    # Build KNN graph in FP32 to avoid bf16/fp16 overflow under AMP
                    device_type = (
                        last_repr.device.type
                        if hasattr(last_repr, "device")
                        else "cuda"
                    )
                    emb_fp32 = last_repr.float()
                    # Disable autocast only for KNN construction
                    autocast_ctx = (
                        torch.amp.autocast
                        if hasattr(torch, "amp")
                        else torch.cuda.amp.autocast
                    )
                    with autocast_ctx(device_type=device_type, enabled=False):
                        eidx, eattr = build_knn_from_embeddings(
                            emb_fp32,
                            k=knn_k,
                            exclude_self=True,
                            symmetric=True,
                        )
                except Exception as e:
                    logger.warning(f"KNN graph construction failed: {e}")
                    eidx, eattr = None, None

            # Hard fallback (self-loops) if no edges
            if eidx is None:
                device = last_repr.device
                num_nodes = last_repr.size(0)
                # Create self-loops as fallback
                eidx = torch.arange(num_nodes, device=device)
                eidx = torch.stack([eidx, eidx], dim=0)  # (2, N) self-loops
                eattr = None
                logger.debug("Using self-loops as fallback graph structure")

            # Ensure correct dtype and device
            eidx = eidx.to(last_repr.device, non_blocking=True).long().contiguous()
            if eattr is not None:
                eattr = eattr.to(last_repr.device, non_blocking=True).contiguous()

            if not use_edge_attr:
                eattr = None

            # Optional input-level edge dropout (in addition to per-layer dropout)
            try:
                p = float(self._edge_input_dropout_p)
            except Exception:
                p = 0.0
            if self.training and p > 0.0:
                eidx, mask = _pyg_dropout_edge(eidx, p=p, training=True)
                if eattr is not None and mask is not None:
                    eattr = eattr[mask]

            # Create batch indices if needed for GraphNorm
            # For now, assume single graph (batch_idx=None) unless GraphNorm is explicitly needed
            batch_for_gat = None
            if hasattr(self.gat, "norms") and len(self.gat.norms) > 0:
                # Check if any norm layer is GraphNorm
                from ..components.graph_norm import GraphNorm

                if any(isinstance(norm, GraphNorm) for norm in self.gat.norms):
                    # Create batch indices assuming all nodes belong to single graph
                    batch_for_gat = torch.zeros(
                        last_repr.size(0), dtype=torch.long, device=last_repr.device
                    )

            # Always run GAT forward (edge dropout already applied if enabled)
            if return_attention_weights:
                gat_out, gat_attention_weights = self.gat(
                    last_repr,
                    eidx,
                    eattr,
                    batch=batch_for_gat,
                    return_attention_weights=True,
                )
            else:
                gat_out = self.gat(last_repr, eidx, eattr, batch=batch_for_gat)

            # Fusion with learnable alpha parameter
            # alpha = alpha_min + (1 - alpha_min) * sigmoid(alpha_logit)
            alpha = self.alpha_graph_min + (1 - self.alpha_graph_min) * torch.sigmoid(
                self.alpha_logit
            )

            # Check for force mode (for debugging)
            force_mode = _os.getenv("FUSE_FORCE_MODE", "").strip().lower()
            if force_mode == "graph_only":
                # Force GAT only for gradient verification
                final_repr = gat_out  # Keep gradient flow
                # Override alpha to 1.0 while keeping gradient flow
                alpha = alpha * 0.0 + 1.0  # This keeps the gradient connection
            elif force_mode == "tft_only":
                # Force TFT only
                final_repr = last_repr
                # Override alpha to 0.0 while keeping gradient flow
                alpha = alpha * 0.0  # This keeps the gradient connection
            else:
                # Normal weighted combination: (1-alpha)*TFT + alpha*GAT
                # IMPORTANT: Never detach here - both branches need gradients
                final_repr = (1 - alpha) * last_repr + alpha * gat_out

        # Prediction head
        features = self.prediction_mlp(final_repr)

        # Generate outputs - ALL heads in FP32 to prevent collapse
        outputs = {}

        # Disable autocast for ALL prediction heads - support both cuda and cpu
        device_type = (
            features.device.type if hasattr(features.device, "type") else "cuda"
        )
        autocast_ctx = (
            torch.amp.autocast if hasattr(torch, "amp") else torch.cuda.amp.autocast
        )
        with autocast_ctx(device_type=device_type, enabled=False):
            # Convert features to FP32 once
            features_fp32 = features.float()
            # Apply dropout to maintain variance
            if self.training:
                features_fp32 = self.output_dropout(features_fp32)

            # Point predictions — rely on existence of heads instead of config
            for horizon in self.config.data.time_series.prediction_horizons:
                key = f"point_horizon_{horizon}"
                if key in self.output_heads:
                    out = self.output_heads[key](features_fp32)

                    # Optional: head noise (env-controlled, off by default)
                    try:
                        _noise_std = float(
                            _os.getenv("HEAD_NOISE_STD", "0.0")
                        )  # Default: disabled
                        _noise_warmup_epochs = int(
                            _os.getenv("HEAD_NOISE_WARMUP_EPOCHS", "2")
                        )
                        _epoch = int(getattr(self, "_epoch", 0))
                        # Also add OUTPUT noise during warmup
                        _output_noise_std = float(_os.getenv("OUTPUT_NOISE_STD", "0.0"))
                    except Exception:
                        _noise_std, _noise_warmup_epochs, _epoch = 0.0, 2, 0
                        _output_noise_std = 0.0

                    if (
                        self.training
                        and _noise_std > 0.0
                        and _epoch <= _noise_warmup_epochs
                    ):
                        out = out + torch.randn_like(out) * _noise_std

                    # Additional output noise to force variance
                    if self.training and _output_noise_std > 0.0:
                        out = out + torch.randn_like(out) * _output_noise_std

                    outputs[key] = out

            # Quantile predictions — rely on existence of heads instead of config
            for horizon in self.config.data.time_series.prediction_horizons:
                key = f"quantile_horizon_{horizon}"
                if key in self.output_heads:
                    q_raw = self.output_heads[key](features_fp32)
                    # Enforce non-crossing via cumulative softplus deltas if configured
                    enforce = False
                    try:
                        enforce = bool(
                            self.config.model.prediction_head.output.quantile_prediction.get(
                                "enforce_monotonic", True
                            )
                        )
                    except Exception:
                        enforce = True
                    if enforce and q_raw.shape[-1] >= 2:
                        deltas = torch.nn.functional.softplus(
                            q_raw[:, 1:] - q_raw[:, :-1]
                        )
                        q0 = q_raw[:, :1]
                        q_rest = torch.cumsum(deltas, dim=1)
                        q_mono = torch.cat([q0, q0 + q_rest], dim=1)
                        outputs[key] = q_mono
                    else:
                        outputs[key] = q_raw

            # Student-t parameters
            for horizon in self.config.data.time_series.prediction_horizons:
                key = f"t_params_horizon_{horizon}"
                if key in self.t_heads:
                    outputs[key] = self.t_heads[key](features_fp32)

            # Direction logits
            for horizon in self.config.data.time_series.prediction_horizons:
                key = f"direction_horizon_{horizon}"
                if key in self.dir_heads:
                    outputs[key] = self.dir_heads[key](features_fp32)

        # Add auxiliary outputs
        outputs["feature_weights"] = feature_weights

        # Calculate total sparsity loss including alpha penalty
        total_sparsity_loss = sparsity_loss

        # Add fusion alpha value for monitoring and penalty
        if self.gat is not None:
            alpha_value = self.alpha_graph_min + (
                1 - self.alpha_graph_min
            ) * torch.sigmoid(self.alpha_logit)
            outputs["fusion_alpha"] = alpha_value

            # Add penalty to encourage alpha not to collapse too low (small hinge loss)
            if self.alpha_penalty > 0:
                alpha_penalty_loss = (
                    self.alpha_penalty * torch.relu(0.20 - alpha_value).mean()
                )
                total_sparsity_loss = total_sparsity_loss + alpha_penalty_loss
                outputs["alpha_penalty_loss"] = alpha_penalty_loss

        outputs["sparsity_loss"] = total_sparsity_loss

        if return_attention_weights:
            outputs["tft_attention_weights"] = attention_weights
            outputs["gat_attention_weights"] = gat_attention_weights

        return outputs
