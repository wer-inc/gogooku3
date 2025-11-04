from __future__ import annotations

"""
Feature selection utilities for financial ML datasets.

Implements simple Mutual Information, RandomForest impurity, and L1-based
selection. Designed to run offline against a Parquet/Polars DataFrame and
emit a JSON list of selected column names for the training DataModule to use.
"""

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV

logger = logging.getLogger(__name__)


SelectorMethod = Literal["mutual_info", "lasso", "random_forest"]


@dataclass
class SelectionConfig:
    method: SelectorMethod = "mutual_info"
    top_k: int = 100
    min_importance: float = 0.0
    target_column: str = "target_1d"


def _ensure_numeric(df: pl.DataFrame, cols: Iterable[str]) -> list[str]:
    out: list[str] = []
    for c in cols:
        try:
            if df.schema.get(c) in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8):
                out.append(c)
        except Exception:
            continue
    return out


def select_features(df: pl.DataFrame, cfg: SelectionConfig) -> list[str]:
    """Return a ranked list of selected feature names according to cfg."""
    if cfg.target_column not in df.columns:
        raise ValueError(f"target column '{cfg.target_column}' not found")

    exclude = {cfg.target_column, "Date", "date", "Code", "code"}
    candidate_cols = [c for c in df.columns if c not in exclude]
    candidate_cols = _ensure_numeric(df, candidate_cols)
    if not candidate_cols:
        return []

    X = df.select(candidate_cols).to_numpy()
    y = df.select(cfg.target_column).to_numpy().ravel()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    importances: np.ndarray
    if cfg.method == "mutual_info":
        importances = mutual_info_regression(X, y, discrete_features=False, random_state=42)
    elif cfg.method == "lasso":
        # L1 shrinkage; standardize implicitly via normalization in solver
        model = LassoCV(cv=5, random_state=42, n_jobs=None).fit(X, y)
        importances = np.abs(model.coef_)
    else:  # random_forest
        rf = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        importances = rf.feature_importances_

    pairs = list(zip(candidate_cols, importances))
    # Filter by threshold and take top_k
    pairs = [p for p in pairs if float(p[1]) >= float(cfg.min_importance)]
    pairs.sort(key=lambda x: float(x[1]), reverse=True)

    return [name for name, _ in pairs[: max(1, int(cfg.top_k))]]


def save_selected(features: list[str], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"selected_features": features}, indent=2), encoding="utf-8")
    logger.info("Saved selected features: %s (k=%d)", path, len(features))
    return path

