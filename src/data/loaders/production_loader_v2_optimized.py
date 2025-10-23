"""Optimized production data loader V2 for backward compatibility."""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, Union

import polars as pl
import torch
from torch.utils.data import Dataset

from src.gogooku3.training.atft.data_module import StreamingParquetDataset

logger = logging.getLogger(__name__)


class ProductionDatasetOptimized(Dataset):
    """
    Optimized Production Dataset for backward compatibility with train_atft.py.

    The legacy pipeline expects a map-style Dataset, but we now delegate actual
    window construction to the streaming dataset used in the modern training
    stack. This keeps memory usage low and preserves the expected interface.
    """

    _DEFAULT_NUMERICS: tuple = (
        pl.Float32,
        pl.Float64,
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    )

    def __init__(
        self,
        files: Union[list[Path], list[str], Path, str],
        config: Any,
        mode: str = "train",
        target_scalers: Optional[dict] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        required_feature_cols: Optional[list[str]] = None,
    ):
        self.config = config
        self.mode = mode
        self.target_scalers = target_scalers or {}
        self.start_date = start_date
        self.end_date = end_date

        if isinstance(files, (str, Path)):
            files = [files]
        self.files: list[Path] = [Path(f) for f in files]

        if not self.files:
            raise FileNotFoundError(
                "ProductionDatasetOptimized requires at least one parquet file"
            )

        if self.start_date or self.end_date:
            logger.warning(
                "ProductionDatasetOptimized streaming backend ignores start/end date filters; "
                "please pre-filter source parquet files if required."
            )

        schema = self._read_schema(self.files)
        feature_cols = self._resolve_feature_columns(schema, required_feature_cols)
        target_cols = self._resolve_target_columns(schema)

        seq_len = 60
        try:
            seq_len = int(self.config.data.time_series.sequence_length)
        except Exception:
            logger.debug(
                "sequence_length not found in config; using default %d", seq_len
            )

        normalization_cfg = getattr(self.config, "normalization", None)
        online_cfg = getattr(normalization_cfg, "online_normalization", None)
        normalize_online = bool(getattr(online_cfg, "enabled", False))

        self._dataset = StreamingParquetDataset(
            file_paths=self.files,
            feature_columns=feature_cols,
            target_columns=target_cols,
            code_column=self.config.data.schema.code_column,
            date_column=self.config.data.schema.date_column,
            sequence_length=seq_len,
            normalize_online=normalize_online,
        )

        self.feature_cols = list(self._dataset.feature_columns)
        self.target_columns = list(self._dataset.target_columns)
        self.prediction_horizons = self._resolve_prediction_horizons(
            self.target_columns
        )

        # Legacy attribute expected by some callers
        self.targets: dict[str, torch.Tensor] = {}

        logger.info(
            "ProductionDatasetOptimized initialized: samples=%d, features=%d, targets=%d, mode=%s",
            len(self),
            len(self.feature_cols),
            len(self.target_columns),
            mode,
        )

    @staticmethod
    def _read_schema(files: Sequence[Path]) -> pl.Schema:
        for path in files:
            if path.exists():
                try:
                    return pl.scan_parquet(path).collect_schema()
                except Exception as exc:
                    logger.warning("Failed to read schema from %s: %s", path, exc)
        raise FileNotFoundError("Unable to read schema from provided parquet files")

    def _resolve_feature_columns(
        self,
        schema: pl.Schema,
        required_feature_cols: Optional[list[str]],
    ) -> list[str]:
        if required_feature_cols:
            available = set(schema.names())
            selected = [col for col in required_feature_cols if col in available]
            missing = set(required_feature_cols) - available
            if missing:
                logger.warning("Required features missing from schema: %s", missing)
            if selected:
                return selected

        exclude_cols = {
            self.config.data.schema.date_column,
            self.config.data.schema.code_column,
            getattr(self.config.data.schema, "target_column", "target"),
        }

        feature_cols: list[str] = []
        for col, dtype in schema.items():
            lower = col.lower()
            if col in exclude_cols or lower.startswith("target_"):
                continue
            if dtype in self._DEFAULT_NUMERICS:
                feature_cols.append(col)

        if not feature_cols:
            raise ValueError(
                "No numeric feature columns detected for ProductionDatasetOptimized"
            )
        return feature_cols

    @staticmethod
    def _resolve_target_columns(schema: pl.Schema) -> list[str]:
        targets: list[str] = []
        for col in schema.names():
            lower = col.lower()
            if lower.startswith("target_") or lower.startswith("feat_ret_"):
                targets.append(col)
        if not targets:
            logger.warning(
                "No target columns detected; defaulting to empty target list"
            )
        return targets

    def _resolve_prediction_horizons(self, target_cols: Sequence[str]) -> list[int]:
        horizons = set()
        try:
            cfg_horizons = self.config.data.time_series.prediction_horizons
            if cfg_horizons:
                return list(cfg_horizons)
        except Exception:
            pass

        import re

        for col in target_cols:
            match = re.search(r"_(\d+)(?:d)?$", col)
            if match:
                horizons.add(int(match.group(1)))

        if not horizons:
            logger.warning("Falling back to default prediction horizons [1, 5, 10, 20]")
            return [1, 5, 10, 20]
        return sorted(horizons)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._dataset[idx]
