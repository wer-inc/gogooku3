from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from gogooku3.features_ext.cs_standardize import CSStats, fit_cs_stats, transform_cs
from gogooku3.features_ext.outliers import fit_winsor_stats, transform_winsor
from gogooku3.training.cv_purged import Fold


@dataclass(frozen=True)
class PanelDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]):
    X: torch.Tensor
    y: torch.Tensor
    vol20: torch.Tensor | None

    def __len__(self) -> int:  # type: ignore[override]
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:  # type: ignore[override]
        if self.vol20 is None:
            return self.X[idx], self.y[idx], None
        return self.X[idx], self.y[idx], self.vol20[idx]


class PanelDataModule:
    """Minimal datamodule for cross-sectional time-series panels.

    - Fits CS statistics on the training fold only.
    - Applies transform to both train/val without leakage.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        *,
        feature_cols: Sequence[str],
        target_col: str,
        date_col: str = "Date",
        by_cols: Iterable[str] | None = None,
        outlier_cols: Iterable[str] | None = None,
        vol_col: str | None = "volatility_20d",
    ) -> None:
        self.df = df
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.date_col = date_col
        self.by_cols = list(by_cols) if by_cols else None
        self.outlier_cols = list(outlier_cols) if outlier_cols else []
        self.vol_col = vol_col if (vol_col and vol_col in df.columns) else None
        self._cs_stats: dict[str, CSStats] | None = None
        self._winsor_stats: dict[str, tuple[float, float]] | None = None

    def setup_fold(self, fold: Fold) -> tuple[PanelDataset, PanelDataset, pl.DataFrame, pl.DataFrame]:
        # Split
        train_df = self.df.take(fold.train_idx)
        val_df = self.df.take(fold.val_idx)

        # 1) Fit winsor stats (train only) and apply to both train/val (leak-safe)
        if self.outlier_cols:
            self._winsor_stats = fit_winsor_stats(train_df, self.outlier_cols)
            train_df = transform_winsor(train_df, self._winsor_stats)
            val_df = transform_winsor(val_df, self._winsor_stats)

        # 2) Fit CS-Z on train and transform both
        self._cs_stats = fit_cs_stats(train_df, self.feature_cols, date_col=self.date_col, by_cols=self.by_cols)
        tcols = self.feature_cols
        train_df_t = transform_cs(train_df, self._cs_stats, tcols)
        val_df_t = transform_cs(val_df, self._cs_stats, tcols)

        # Prepare tensors (use standardized columns when available)
        feat_cols_final = [f"{c}_cs_z" for c in tcols]
        X_train = torch.tensor(np.asarray(train_df_t.select(feat_cols_final).to_numpy()), dtype=torch.float32)
        y_train = torch.tensor(np.asarray(train_df_t.select(self.target_col).to_numpy()).squeeze(), dtype=torch.float32)
        X_val = torch.tensor(np.asarray(val_df_t.select(feat_cols_final).to_numpy()), dtype=torch.float32)
        y_val = torch.tensor(np.asarray(val_df_t.select(self.target_col).to_numpy()).squeeze(), dtype=torch.float32)

        # Optional volatility weight column
        vtr: torch.Tensor | None = None
        vva: torch.Tensor | None = None
        if self.vol_col is not None and self.vol_col in train_df_t.columns:
            vtr = torch.tensor(np.asarray(train_df_t.select(self.vol_col).to_numpy()).squeeze(), dtype=torch.float32)
            vva = torch.tensor(np.asarray(val_df_t.select(self.vol_col).to_numpy()).squeeze(), dtype=torch.float32)

        return PanelDataset(X_train, y_train, vtr), PanelDataset(X_val, y_val, vva), train_df_t, val_df_t


__all__ = ["PanelDataModule", "PanelDataset"]
