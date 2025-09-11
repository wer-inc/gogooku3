"""
ATFT-GAT-FAN Configuration Management System
Pydanticベースの統一設定管理
"""

import os
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from functools import lru_cache
import logging

# Pydantic設定
try:
    from pydantic import Field
    from pydantic_settings import BaseSettings
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

logger = logging.getLogger(__name__)

if PYDANTIC_AVAILABLE:
    class ModelSettings(BaseSettings):
        """Pydantic-based configuration management"""
        model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "case_sensitive": False, "extra": "allow"}

        # GPU/メモリ設定
        mixed_precision: bool = Field(default=True)
        gradient_checkpointing: bool = Field(default=True)
        compile_model: bool = Field(default=True)

        # GAT設定
        gat_alpha_init: float = Field(default=0.2, ge=0.0, le=1.0)
        gat_alpha_min: float = Field(default=0.05, ge=0.0, le=1.0)  # 緩和
        gat_alpha_penalty: float = Field(default=2e-3, ge=0.0)
        edge_dropout_p: float = Field(default=0.15, ge=0.0, le=0.5)
        gat_temperature: float = Field(default=1.0, ge=0.7, le=1.3)

        # FAN設定
        freq_dropout_p: float = Field(default=0.1, ge=0.0, le=0.3)
        freq_dropout_min_width: float = Field(default=0.05, ge=0.01, le=0.5)
        freq_dropout_max_width: float = Field(default=0.2, ge=0.01, le=0.5)

        # データローダー設定
        num_workers: int = Field(default=4, ge=0, le=32)
        prefetch_factor: int = Field(default=2, ge=1)
        pin_memory: bool = Field(default=True)

        # 学習安定化設定
        huber_delta: float = Field(default=0.01, ge=0.001, le=0.1)
        warmup_steps: int = Field(default=1500, ge=0)
        ema_decay: float = Field(default=0.999, ge=0.9, le=0.9999)
        grad_clip_val: float = Field(default=1.0, ge=0.1, le=10.0)

        # 出力ヘッド設定
        output_head_std: float = Field(default=0.01, ge=0.001, le=0.1)
        layer_scale_gamma: float = Field(default=0.1, ge=0.01, le=1.0)

        # 監視設定
        enable_wandb: bool = Field(default=False)
        enable_tensorboard: bool = Field(default=True)
        log_grad_norm: bool = Field(default=True)
        wandb_project: str = Field(default="atft-gat-fan")

        # 再現性設定
        seed: int = Field(default=42, ge=0)
        deterministic: bool = Field(default=True)

        # 堅牢性設定
        auto_recover_oom: bool = Field(default=True)
        emergency_checkpoint: bool = Field(default=True)
        max_retries: int = Field(default=3, ge=1, le=10)

        # 最適化設定
        base_lr: float = Field(default=5e-4, ge=1e-6, le=1e-2)
        base_weight_decay: float = Field(default=1e-4, ge=0.0, le=1e-2)

        # ParamGroup設定
        fan_lr_multiplier: float = Field(default=0.6, ge=0.1, le=2.0)
        fan_wd_multiplier: float = Field(default=5.0, ge=0.1, le=10.0)
        gat_lr_multiplier: float = Field(default=0.8, ge=0.1, le=2.0)
        gat_wd_multiplier: float = Field(default=2.0, ge=0.1, le=10.0)

        def get_param_groups(self, model):
            """ParamGroup設定を取得"""
            return {
                'base_lr': self.base_lr,
                'base_weight_decay': self.base_weight_decay,
                'fan_lr': self.base_lr * self.fan_lr_multiplier,
                'fan_wd': self.base_weight_decay * self.fan_wd_multiplier,
                'gat_lr': self.base_lr * self.gat_lr_multiplier,
                'gat_wd': self.base_weight_decay * self.gat_wd_multiplier,
            }
else:
    # Fallback to dataclass if pydantic not available
    from dataclasses import dataclass, field as dataclass_field
    from typing import get_type_hints

    def Field(default=None, **kwargs):
        return dataclass_field(default=default)

    @dataclass
    class ModelSettings:
        """Fallback dataclass version"""
        # GPU/メモリ設定
        mixed_precision: bool = Field(default=True)
        gradient_checkpointing: bool = Field(default=True)
        compile_model: bool = Field(default=True)

        # GAT設定
        gat_alpha_init: float = Field(default=0.2)
        gat_alpha_min: float = Field(default=0.3)
        gat_alpha_penalty: float = Field(default=1e-4)
        edge_dropout_p: float = Field(default=0.0)

        # データローダー設定
        num_workers: int = Field(default=4)
        prefetch_factor: int = Field(default=2)
        pin_memory: bool = Field(default=True)

        # 学習安定化設定
        huber_delta: float = Field(default=0.01)
        warmup_steps: int = Field(default=1500)
        ema_decay: float = Field(default=0.999)
        grad_clip_val: float = Field(default=1.0)

        # 監視設定
        enable_wandb: bool = Field(default=False)
        enable_tensorboard: bool = Field(default=True)
        log_grad_norm: bool = Field(default=True)

        # 再現性設定
        seed: int = Field(default=42)
        deterministic: bool = Field(default=True)

        # 堅牢性設定
        auto_recover_oom: bool = Field(default=True)
        emergency_checkpoint: bool = Field(default=True)


@lru_cache()
def get_settings() -> ModelSettings:
    """シングルトン設定インスタンス"""
    return ModelSettings()

def save_environment_snapshot(output_dir: str = "."):
    """環境情報のスナップショットを保存"""
    import sys
    import torch
    import numpy as np
    import json
    import random
    from datetime import datetime
    
    env_info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": sys.platform,
        "torch_version": torch.__version__ if 'torch' in globals() else "not installed",
        "cuda_version": torch.version.cuda if 'torch' in globals() and torch.cuda.is_available() else "not available",
        "cudnn_version": torch.backends.cudnn.version() if 'torch' in globals() else "not available",
        "numpy_version": np.__version__,
        "random_state": random.getstate(),
        "numpy_random_state": np.random.get_state().tolist(),
        "torch_random_state": torch.get_rng_state().tolist() if 'torch' in globals() else None,
        "cuda_random_state": torch.cuda.get_rng_state_all() if 'torch' in globals() and torch.cuda.is_available() else None,
    }
    
    output_path = Path(output_dir) / "environment_snapshot.json"
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(env_info, f, indent=2, default=str, ensure_ascii=False)
    
    logger.info(f"Environment snapshot saved to {output_path}")
    return output_path

def set_reproducibility(seed: int = 42, deterministic: bool = True):
    """再現性のための完全なシード設定"""
    import random
    import numpy as np
    import torch
    
    # Python random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # cuBLAS workspace config for reproducibility
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    logger.info(f"Reproducibility set with seed={seed}, deterministic={deterministic}")

# 後方互換性のためのエイリアス
Settings = ModelSettings
# config = get_settings()  # インポート時の初期化を避ける
