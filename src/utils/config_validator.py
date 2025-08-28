"""
Hydra configuration validator with strict checks
"""

import logging
from typing import List
from omegaconf import DictConfig, OmegaConf
import yaml

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Hydra設定の厳格な検証"""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self, config: DictConfig) -> bool:
        """
        設定全体を検証

        Args:
            config: Hydra設定

        Returns:
            有効な場合True
        """
        self.errors = []
        self.warnings = []

        # Enable strict mode to forbid unknown keys
        try:
            OmegaConf.set_struct(config, True)
        except Exception as e:
            logger.warning(f"Could not set struct mode: {e}")

        # Check for common misconfigurations first
        self._check_common_errors(config)

        # 必須セクションの確認
        self._check_required_sections(config)

        # データ設定の検証
        self._validate_data_config(config)

        # モデル設定の検証
        self._validate_model_config(config)

        # 訓練設定の検証
        self._validate_train_config(config)

        # 相互依存性のチェック
        self._validate_dependencies(config)

        # エラーレポート
        if self.errors:
            logger.error("Configuration validation failed:")
            for error in self.errors:
                logger.error(f"  - {error}")
            return False

        if self.warnings:
            logger.warning("Configuration warnings:")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")

        logger.info("Configuration validation passed")
        return True

    def _check_common_errors(self, config: DictConfig):
        """Check for common configuration errors"""
        # Check for invalid key path 'model.model.hidden_size'
        if OmegaConf.select(config, "model.model.hidden_size") is not None:
            self.errors.append(
                "Invalid key path 'model.model.hidden_size'. Use 'model.hidden_size' instead."
            )
            logger.error("Found hidden_size at incorrect path: model.model.hidden_size")

        # Check for duplicate hidden_size definitions
        if "model" in config:
            if "model" in config.model and "hidden_size" in config.model:
                self.errors.append(
                    "Ambiguous configuration: both 'model.hidden_size' and 'model.model.*' exist"
                )

        # Check for other common typos
        if OmegaConf.select(config, "train.train") is not None:
            self.errors.append(
                "Invalid key path 'train.train.*'. Check for duplicate nesting."
            )

        if OmegaConf.select(config, "data.data") is not None:
            self.errors.append(
                "Invalid key path 'data.data.*'. Check for duplicate nesting."
            )

    def _check_required_sections(self, config: DictConfig):
        """必須セクションの存在確認"""
        required_sections = ["data", "model", "train"]

        for section in required_sections:
            if section not in config:
                self.errors.append(f"Missing required section: {section}")

    def _validate_data_config(self, config: DictConfig):
        """データ設定の検証"""
        if "data" not in config:
            return

        data = config.data

        # パス設定
        if "paths" in data:
            if "raw_data_dir" not in data.paths:
                self.errors.append("data.paths.raw_data_dir is required")

        # 時系列設定
        if "time_series" in data:
            ts = data.time_series

            # シーケンス長
            if "sequence_length" in ts:
                seq_len = ts.sequence_length
                if seq_len < 5:
                    self.errors.append(
                        f"sequence_length={seq_len} is too short (min: 5)"
                    )
                elif seq_len > 100:
                    self.warnings.append(f"sequence_length={seq_len} may be too long")

            # 予測ホライズン
            if "prediction_horizons" in ts:
                horizons = ts.prediction_horizons
                if not horizons:
                    self.errors.append("prediction_horizons cannot be empty")
                elif max(horizons) > 30:
                    self.warnings.append(f"Max horizon {max(horizons)} may be too far")

        # バッチサイズ
        if "batch" in data:
            batch = data.batch
            if "train_batch_size" in batch:
                bs = batch.train_batch_size
                if bs < 1:
                    self.errors.append(f"train_batch_size={bs} must be positive")
                elif bs > 10000:
                    self.warnings.append(f"train_batch_size={bs} may be too large")

    def _validate_model_config(self, config: DictConfig):
        """モデル設定の検証"""
        if "model" not in config:
            return

        model = config.model

        # 隠れ層サイズ
        if "hidden_size" in model:
            hs = model.hidden_size
            if hs < 32:
                self.errors.append(f"hidden_size={hs} is too small (min: 32)")
            elif hs > 2048:
                self.warnings.append(f"hidden_size={hs} may be too large")

        # GAT設定
        if "gat" in model and model.gat.get("enabled", False):
            gat = model.gat

            # アーキテクチャ
            if "architecture" in gat:
                arch = gat.architecture
                if "num_layers" in arch:
                    nl = arch.num_layers
                    if nl < 1:
                        self.errors.append(f"GAT num_layers={nl} must be >= 1")
                    elif nl > 10:
                        self.warnings.append(f"GAT num_layers={nl} may be too deep")

            # グラフ構築
            if "graph" in gat:
                graph = gat.graph
                if "k_neighbors" in graph:
                    k = graph.k_neighbors
                    if k < 2:
                        self.errors.append(f"k_neighbors={k} must be >= 2")
                    elif k > 100:
                        self.warnings.append(f"k_neighbors={k} may be too large")

    def _validate_train_config(self, config: DictConfig):
        """訓練設定の検証"""
        if "train" not in config:
            return

        train = config.train

        # オプティマイザ
        if "optimizer" in train:
            opt = train.optimizer

            # 学習率
            if "lr" in opt:
                lr = opt.lr
                if lr <= 0:
                    self.errors.append(f"learning rate={lr} must be positive")
                elif lr > 1.0:
                    self.warnings.append(f"learning rate={lr} may be too large")

        # トレーナー
        if "trainer" in train:
            trainer = train.trainer

            # エポック数
            if "max_epochs" in trainer:
                epochs = trainer.max_epochs
                if epochs < 1:
                    self.errors.append(f"max_epochs={epochs} must be >= 1")

            # グラディエント累積
            if "accumulate_grad_batches" in trainer:
                acc = trainer.accumulate_grad_batches
                if acc < 1:
                    self.errors.append(f"accumulate_grad_batches={acc} must be >= 1")
                elif acc > 100:
                    self.warnings.append(
                        f"accumulate_grad_batches={acc} may be too large"
                    )

    def _validate_dependencies(self, config: DictConfig):
        """設定間の依存関係を検証"""

        # バッチサイズとメモリの関係
        if "data" in config and "batch" in config.data:
            batch_size = config.data.batch.get("train_batch_size", 32)
            seq_len = config.data.time_series.get("sequence_length", 20)

            # 概算メモリ使用量（MB）
            estimated_memory = (batch_size * seq_len * 8 * 4) / (1024 * 1024)
            if estimated_memory > 8000:  # 8GB以上
                self.warnings.append(
                    f"Estimated memory usage ~{estimated_memory:.0f}MB may be too high"
                )

        # Mixed Precisionとモデルサイズ
        if "train" in config and "mixed_precision" in config.train:
            if config.train.mixed_precision.get("enabled", False):
                if "model" in config and "hidden_size" in config.model:
                    hs = config.model.hidden_size
                    if hs < 128:
                        self.warnings.append(
                            f"Mixed precision with hidden_size={hs} may cause instability"
                        )

    def save_validated_config(self, config: DictConfig, path: str):
        """検証済み設定を保存"""
        if not self.validate(config):
            raise ValueError("Configuration validation failed")

        # OmegaConfをYAMLに変換
        config_dict = OmegaConf.to_container(config, resolve=True)

        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        logger.info(f"Validated configuration saved to {path}")


def validate_config_file(config_path: str) -> bool:
    """
    設定ファイルを検証

    Args:
        config_path: 設定ファイルパス

    Returns:
        有効な場合True
    """
    try:
        # YAMLファイルを読み込み
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # DictConfigに変換
        config = OmegaConf.create(config_dict)

        # 検証
        validator = ConfigValidator()
        return validator.validate(config)

    except Exception as e:
        logger.error(f"Failed to validate config file {config_path}: {e}")
        return False
