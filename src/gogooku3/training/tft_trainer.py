from __future__ import annotations

"""Minimal production-style TFT trainer (quantile regression surrogate).

Features:
- Joins known-future and static features if provided (flat I/O).
- Builds forward targets per horizon (shift -h by id) with leakage prevention via
  WalkForwardSplitterV2 (embargo) for Purged CV.
- Trains QuantileRegressor models per horizon+quantile; evaluates WQL on test.

Inputs expect columns:
  obs:   id, ts, y, <feature...>
  known: id|'*', ts, <future-feature...>
  static:id, <static-feature...>
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

from gogooku3.training.split import WalkForwardSplitterV2
from gogooku3.metrics.forecast_metrics import weighted_quantile_loss
from gogooku3.features.feature_builder import add_default_features
from gogooku3.features.feature_params import FeatureParams
from gogooku3.features.known_future import add_jp_holiday_features, normalize_static


def _left_join_known_future(df: pd.DataFrame, df_known: pd.DataFrame | None) -> pd.DataFrame:
    if df_known is None or df_known.empty:
        return df
    k = df_known.copy()
    k["ts"] = pd.to_datetime(k["ts"])  # type: ignore[assignment]
    # Support wildcard id='*' to broadcast
    broadcast = k[k.get("id", "").astype(str) == "*"] if "id" in k.columns else k.iloc[0:0]
    specific = k if broadcast.empty else k[k.get("id", "").astype(str) != "*"]
    d = df.copy()
    d["ts"] = pd.to_datetime(d["ts"])  # type: ignore[assignment]
    if not specific.empty:
        d = d.merge(specific, on=["id", "ts"], how="left", suffixes=(None, None))
    if not broadcast.empty:
        d = d.merge(broadcast.drop(columns=["id"]), on=["ts"], how="left", suffixes=(None, None))
    return d


def _left_join_static(df: pd.DataFrame, df_static: pd.DataFrame | None) -> pd.DataFrame:
    if df_static is None or df_static.empty:
        return df
    return df.merge(df_static, on=["id"], how="left", suffixes=(None, None))


@dataclass
class TFTTrainerConfig:
    horizons: Sequence[int] = (1, 5, 10, 20, 30)
    quantiles: Sequence[float] = (0.1, 0.5, 0.9)
    embargo_days: int = 20
    n_splits: int = 3
    features: FeatureParams = FeatureParams()
    # Categorical encoding for non-numeric columns: 'codes' | 'hash' | 'target'
    cat_encode: str = "codes"
    cat_cols: list[str] | None = None
    hash_buckets: int = 64


def train_tft_quantile(
    df_obs: pd.DataFrame,
    df_known_future: pd.DataFrame | None = None,
    df_static: pd.DataFrame | None = None,
    config: TFTTrainerConfig | None = None,
) -> dict:
    cfg = config or TFTTrainerConfig()
    d = df_obs.copy()
    if not {"id", "ts", "y"}.issubset(d.columns):
        raise ValueError("df_obs must contain id, ts, y")
    d["ts"] = pd.to_datetime(d["ts"])  # type: ignore[assignment]
    d.sort_values(["id", "ts"], inplace=True)

    # Join and build features
    d = add_jp_holiday_features(d)
    d = _left_join_known_future(d, df_known_future)
    if df_static is not None:
        df_static = normalize_static(df_static)
    d = _left_join_static(d, df_static)
    d = add_default_features(d, id_col="id", ts_col="ts", y_col="y", params=cfg.features)

    # Build forward targets per horizon
    for h in cfg.horizons:
        d[f"target_{h}"] = d.groupby("id", sort=False)["y"].shift(-int(h))

    # Feature columns (exclude id/ts/y/target_*)
    exclude = {"id", "ts", "y"} | {f"target_{h}" for h in cfg.horizons}
    feat_cols = [c for c in d.columns if c not in exclude]
    # Ensure at least 'y' if no features beyond y exist
    if not feat_cols:
        feat_cols = ["y"]

    # Prepare splitter
    splitter = WalkForwardSplitterV2(date_col="ts", embargo_days=cfg.embargo_days)
    splits = list(splitter.split(d, n_splits=cfg.n_splits))
    results: Dict[str, float] = {}

    for h in cfg.horizons:
        # Collect fold losses
        fold_losses: list[float] = []
        for tr_idx, te_idx in splits:
            tr = d.loc[tr_idx]
            te = d.loc[te_idx]
            tr_h = tr.dropna(subset=[f"target_{h}"])
            te_h = te.dropna(subset=[f"target_{h}"])
            if tr_h.empty or te_h.empty:
                continue
            # Categorical encoding per fold (leakage-safe)
            cat_cols = cfg.cat_cols or [c for c in feat_cols if (tr_h[c].dtype == object or str(tr_h[c].dtype).startswith("category"))]
            num_cols = [c for c in feat_cols if c not in cat_cols]
            Xtr_df = tr_h[num_cols].copy()
            Xte_df = te_h[num_cols].copy()
            if cfg.cat_encode == "codes":
                for c in cat_cols:
                    codes = tr_h[c].astype("category").cat.codes.astype(float)
                    Xtr_df[c] = codes
                    # map categories from train to test, unseen -> -1
                    cats = tr_h[c].astype("category").cat.categories
                    Xte_df[c] = te_h[c].astype(pd.CategoricalDtype(categories=cats)).cat.codes.astype(float)
            elif cfg.cat_encode == "hash":
                import hashlib
                def _hash_to_bucket(val: str) -> float:
                    h = hashlib.md5(str(val).encode()).hexdigest()
                    bucket = int(h[:8], 16) % max(cfg.hash_buckets, 2)
                    return bucket / float(max(cfg.hash_buckets - 1, 1))
                for c in cat_cols:
                    Xtr_df[c] = tr_h[c].astype(str).map(_hash_to_bucket).astype(float)
                    Xte_df[c] = te_h[c].astype(str).map(_hash_to_bucket).astype(float)
            elif cfg.cat_encode == "target":
                tgt = f"target_{h}"
                for c in cat_cols:
                    means = tr_h.groupby(c)[tgt].mean()
                    Xtr_df[c] = tr_h[c].map(means).astype(float).fillna(means.mean())
                    Xte_df[c] = te_h[c].map(means).astype(float).fillna(means.mean())
            else:
                raise ValueError(f"Unknown cat_encode: {cfg.cat_encode}")
            Xtr = Xtr_df.astype(float).to_numpy()
            ytr = tr_h[f"target_{h}"].astype(float).to_numpy()
            # Fit quantile models
            models: Dict[float, QuantileRegressor] = {}
            for q in cfg.quantiles:
                m = QuantileRegressor(quantile=float(q), alpha=1e-4, solver="highs")
                try:
                    m.fit(Xtr, ytr)
                    models[q] = m
                except Exception:
                    continue
            if not models:
                continue
            # Predict on test
            Xte = Xte_df.astype(float).to_numpy()
            preds = {}
            for q, m in models.items():
                try:
                    preds[f"p{int(q*100):02d}"] = m.predict(Xte)
                except Exception:
                    continue
            if not preds:
                continue
            df_pred = te_h[["id", "ts"]].copy()
            for k, v in preds.items():
                df_pred[k] = v
            # Evaluate WQL on this fold
            fold_losses.append(weighted_quantile_loss(te_h[f"target_{h}"].to_numpy(), df_pred))
        if fold_losses:
            results[f"WQL_h{h}"] = float(np.mean(fold_losses))
    return {"metrics": results, "splits": len(splits), "horizons": list(cfg.horizons)}
