"""
ATFT Training Environment Setup Module
環境設定とシード固定を管理するモジュール
"""

import logging
import os
import random
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class ATFTEnvironment:
    """ATFT training environment configuration manager."""

    # Default environment variables for stability
    DEFAULT_ENV_VARS = {
        "USE_T_NLL": "1",
        "OUTPUT_NOISE_STD": "0.02",
        "HEAD_NOISE_STD": "0.05",
        "HEAD_NOISE_WARMUP_EPOCHS": "5",
        "GAT_ALPHA_INIT": "0.3",
        "GAT_ALPHA_MIN": "0.1",
        "GAT_ALPHA_PENALTY": "1e-3",
        "EDGE_DROPOUT_INPUT_P": "0.1",
        "DEGENERACY_GUARD": "1",
        "DEGENERACY_WARMUP_STEPS": "1000",
        "DEGENERACY_CHECK_EVERY": "200",
        "DEGENERACY_MIN_RATIO": "0.05",
        "USE_AMP": "1",
        "AMP_DTYPE": "bf16",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }

    # Training profile settings
    TRAINING_PROFILES = {
        "production": {
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_NO_ADVISORY_WARNINGS": "true",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TF_CPP_MIN_LOG_LEVEL": "2",
            "PYTHONWARNINGS": "ignore",
            "HYDRA_FULL_ERROR": "1",
        },
        "debug": {
            "TOKENIZERS_PARALLELISM": "false",
            "HYDRA_FULL_ERROR": "1",
            "PYTHONWARNINGS": "default",
        },
        "benchmark": {
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "CUDA_LAUNCH_BLOCKING": "0",
        },
    }

    def __init__(self, profile: str = "production"):
        """
        Initialize the environment manager.

        Args:
            profile: Training profile to use ('production', 'debug', 'benchmark')
        """
        self.profile = profile
        self.original_env = {}

    def setup(self, override_vars: dict[str, str] | None = None) -> None:
        """
        Set up the training environment.

        Args:
            override_vars: Additional environment variables to set
        """
        # Store original environment
        self.original_env = os.environ.copy()

        # Apply profile settings
        self._apply_profile()

        # Apply default stability settings
        for key, value in self.DEFAULT_ENV_VARS.items():
            self._setenv_if_unset(key, value)

        # Apply user overrides
        if override_vars:
            for key, value in override_vars.items():
                os.environ[key] = str(value)

        # Setup CUDA environment
        self._setup_cuda()

        logger.info(f"✅ Environment configured with profile: {self.profile}")

    def _apply_profile(self) -> None:
        """Apply training profile environment settings."""
        if self.profile not in self.TRAINING_PROFILES:
            logger.warning(f"Unknown profile: {self.profile}, using 'production'")
            self.profile = "production"

        profile_vars = self.TRAINING_PROFILES[self.profile]
        for key, value in profile_vars.items():
            self._setenv_if_unset(key, value)

    def _setenv_if_unset(self, key: str, value: str) -> None:
        """Set environment variable if not already set."""
        if key not in os.environ:
            os.environ[key] = value

    def _setup_cuda(self) -> None:
        """Configure CUDA settings for optimal performance."""
        force_gpu = os.getenv("FORCE_GPU", "0") == "1"
        if torch.cuda.is_available() or force_gpu:
            # Enable TF32 for A100 GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Set cudnn benchmarking for performance
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

            # Check for expandable segments support
            if hasattr(torch.cuda, "memory"):
                try:
                    # PyTorch 2.0+ memory management
                    torch.cuda.empty_cache()
                    logger.info("✅ CUDA memory management configured")
                except Exception as e:
                    logger.warning(f"Could not configure CUDA memory: {e}")

    def fix_seed(self, seed: int = 42, deterministic: bool = False) -> None:
        """
        Fix random seeds for reproducibility.

        Args:
            seed: Random seed value
            deterministic: Whether to enable fully deterministic mode (slower)
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            logger.info(f"✅ Deterministic mode enabled with seed: {seed}")
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            logger.info(f"✅ Random seed fixed: {seed} (non-deterministic mode)")

    def restore(self) -> None:
        """Restore original environment variables."""
        os.environ.clear()
        os.environ.update(self.original_env)
        logger.info("✅ Environment restored to original state")

    @staticmethod
    def get_device(gpu_id: int | None = None) -> torch.device:
        """
        Get the appropriate torch device.

        Args:
            gpu_id: Specific GPU ID to use (None for auto-detect)

        Returns:
            torch.device object
        """
        force_gpu = os.getenv("FORCE_GPU", "0") == "1"
        if torch.cuda.is_available() or force_gpu:
            if gpu_id is not None:
                device = torch.device(f"cuda:{gpu_id}")
            else:
                device = torch.device("cuda")
            try:
                gpu_name = torch.cuda.get_device_name(device)
                memory_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
                logger.info(f"🚀 Using GPU: {gpu_name} ({memory_gb:.1f}GB)")
            except Exception:
                logger.info("🚀 Using GPU device (properties unavailable during init)")
        else:
            device = torch.device("cpu")
            logger.warning("⚠️ GPU not available, using CPU")

        return device

    @staticmethod
    def apply_env_overrides(config: DictConfig) -> DictConfig:
        """
        Apply environment variable overrides to config.

        Args:
            config: Hydra configuration object

        Returns:
            Updated configuration
        """
        # Map of environment variables to config paths
        env_mappings = {
            "BATCH_SIZE": "train.batch.train_batch_size",
            "LR": "train.optimizer.lr",
            "MAX_EPOCHS": "train.trainer.max_epochs",
            "USE_AMP": "train.trainer.use_amp",
            "NUM_WORKERS": "data.num_workers",
            "SEQUENCE_LENGTH": "data.time_series.sequence_length",
            "USE_DAY_BATCH": "data.use_day_batch_sampler",
            "CV_FOLDS": "train.cross_validation.n_folds",
            "EMBARGO_DAYS": "train.cross_validation.embargo_days",
        }

        for env_key, config_path in env_mappings.items():
            if env_key in os.environ:
                value = os.environ[env_key]

                # Parse value type
                try:
                    # Try to parse as number
                    if "." in value:
                        parsed_value = float(value)
                    else:
                        parsed_value = int(value)
                except ValueError:
                    # Parse as boolean or string
                    if value.lower() in ["true", "1", "yes"]:
                        parsed_value = True
                    elif value.lower() in ["false", "0", "no"]:
                        parsed_value = False
                    else:
                        parsed_value = value

                # Apply to config
                keys = config_path.split(".")
                current = config
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[keys[-1]] = parsed_value

                logger.info(f"✅ Override: {config_path} = {parsed_value} (from ${env_key})")

        return config

    def get_hardware_info(self) -> dict[str, Any]:
        """Get hardware information for logging."""
        info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "pytorch_version": torch.__version__,
            "cpu_count": os.cpu_count(),
        }

        if torch.cuda.is_available():
            info.update({
                "gpu_count": torch.cuda.device_count(),
                "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                "gpu_memory_gb": [
                    torch.cuda.get_device_properties(i).total_memory / 1e9
                    for i in range(torch.cuda.device_count())
                ],
            })

        return info
