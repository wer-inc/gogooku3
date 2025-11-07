"""Reusable inference helpers for APEX-Ranker."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date as Date
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch

from ..data import (
    add_cross_sectional_zscores,
    build_panel_cache,
    load_panel_cache,
    panel_cache_key,
    save_panel_cache,
)
from ..data.loader import load_backtest_frame
from ..models import APEXRankerV0

DATE_EPOCH = Date(1970, 1, 1)


def ensure_date(value: Date | datetime | np.datetime64 | str) -> Date:
    """Convert various date-like objects to ``datetime.date``."""
    if isinstance(value, Date):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, np.datetime64):
        days = int(value.astype("datetime64[D]").astype("int64"))
        return DATE_EPOCH + timedelta(days=days)
    if isinstance(value, str):
        return datetime.strptime(value, "%Y-%m-%d").date()
    raise TypeError(f"Unsupported date type: {type(value)}")


def date_to_int(value: Date | datetime | np.datetime64 | str) -> int:
    """Convert a date-like value to integer days since epoch."""
    normalized = ensure_date(value)
    return (normalized - DATE_EPOCH).days


def int_to_date(value: int) -> Date:
    """Convert integer days since epoch back to ``datetime.date``."""
    return DATE_EPOCH + timedelta(days=int(value))


def resolve_device(device_str: str) -> torch.device:
    """Resolve device string to ``torch.device``."""
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_str)


def load_model_checkpoint(
    model_path: Path,
    config: Mapping[str, object],
    device: torch.device,
    n_features: int,
    feature_names: list[str] | None = None,
    validate_features: bool = True,
    add_csz: bool = False,
) -> APEXRankerV0:
    """Instantiate ``APEXRankerV0`` and load weights onto ``device``.

    Args:
        model_path: Path to model checkpoint
        config: Model configuration
        device: Target device (CPU/CUDA)
        n_features: Number of raw input features
        feature_names: Optional feature names for ABI validation
        validate_features: If True, validates feature compatibility (default: True)
        add_csz: If True, uses CS-Z normalized features (CS-Z features replace raw features in cache)
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model_cfg = config["model"]  # type: ignore[index]
    horizons = config["train"]["horizons"]  # type: ignore[index]

    # Determine in_features based on add_csz mode
    # Training uses CS-Z features only (z_features) when add_csz=True
    # - Training: add_cross_sectional_zscores() adds *_cs_z columns, then z_features only are cached
    # - Training: in_features=len(z_features) = 89 (CS-Z features count)
    # - Inference: Same pattern - CS-Z features only are cached and used
    # Therefore: in_features always equals n_features (raw feature count)
    # because CS-Z features replace raw features in the cache, not append to them
    in_features = n_features  # Always raw feature count (e.g., 89)

    print(f"[Model Init] in_features={in_features}, n_features={n_features}, add_csz={add_csz}")

    model = APEXRankerV0(
        in_features=in_features,
        horizons=horizons,
        d_model=model_cfg["d_model"],  # type: ignore[index]
        depth=model_cfg["depth"],  # type: ignore[index]
        patch_len=model_cfg["patch_len"],  # type: ignore[index]
        stride=model_cfg["stride"],  # type: ignore[index]
        n_heads=model_cfg["n_heads"],  # type: ignore[index]
        dropout=model_cfg.get("dropout", 0.1),  # type: ignore[call-arg]
        patch_multiplier=model_cfg.get("patch_multiplier", None),  # type: ignore[call-arg]
    ).to(device)

    # Feature-ABI validation (Phase 1.1)
    if validate_features and feature_names is not None:
        from apex_ranker.models.ranker import load_with_validation

        ckpt = load_with_validation(str(model_path), expected_features=feature_names, strict=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
    else:
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        # Handle both old format (raw state_dict) and new format (dict with metadata)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_weight_turnover(
    current_weights: Mapping[str, float],
    target_weights: Mapping[str, float],
) -> float:
    """Calculate turnover implied by weight change."""
    codes = set(current_weights.keys()) | set(target_weights.keys())
    total_change = sum(abs(target_weights.get(code, 0.0) - current_weights.get(code, 0.0)) for code in codes)
    return total_change / 2.0


class BacktestInferenceEngine:
    """Wrap Phase 2 inference utilities for backtest and API usage."""

    def __init__(
        self,
        model_path: Path,
        config: Mapping[str, object],
        frame: pl.DataFrame,
        feature_cols: Sequence[str],
        *,
        device: str = "auto",
        dataset_path: Path | None = None,
        panel_cache_dir: Path | None = None,
        cache_salt: str | None = None,
        aliases_yaml: str | None = None,
        canonical_features_json: str | None = None,
        add_csz: bool = False,
        csz_eps: float = 1e-6,
        csz_clip: float = 5.0,
    ) -> None:
        self.config = config
        self.device = resolve_device(device)
        self.aliases_yaml = aliases_yaml
        self.canonical_features_json = canonical_features_json
        self.add_csz = add_csz
        self.csz_eps = csz_eps
        self.csz_clip = csz_clip
        data_cfg = config["data"]  # type: ignore[index]
        train_cfg = config["train"]  # type: ignore[index]
        self.date_col = data_cfg["date_column"]  # type: ignore[index]
        self.code_col = data_cfg["code_column"]  # type: ignore[index]
        self.lookback = data_cfg["lookback"]  # type: ignore[index]
        self.feature_cols = list(feature_cols)
        self.z_features = [f"{col}_cs_z" for col in self.feature_cols]
        self.horizons = {int(h) for h in train_cfg["horizons"]}  # type: ignore[index]
        horizon_salt = ",".join(str(h) for h in sorted(self.horizons))
        # FIX: Include CS-Z flag in cache key to prevent collision (raw vs csz normalization)
        csz_flag = "csz" if self.add_csz else "raw"
        combined_salt = f"{horizon_salt}|{csz_flag}"
        if cache_salt:
            combined_salt = f"{combined_salt}|{cache_salt}"
        self.cache_salt = cache_salt

        self.model = load_model_checkpoint(
            model_path=model_path,
            config=config,
            device=self.device,
            n_features=len(self.feature_cols),
            feature_names=self.feature_cols,  # Feature-ABI validation (Phase 1.1)
            validate_features=True,
            add_csz=self.add_csz,  # FIX: Pass CS-Z flag to ensure correct in_features (89 vs 178)
        )

        cache_loaded = False
        cache_path: Path | None = None
        if panel_cache_dir is not None and dataset_path is not None:
            cache_dir = Path(panel_cache_dir).expanduser()
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_key = panel_cache_key(
                dataset_path,
                lookback=self.lookback,
                feature_cols=self.feature_cols,
                extra_salt=combined_salt,
            )
            cache_path = cache_dir / f"{cache_key}.pkl"
            if cache_path.exists():
                try:
                    self.cache = load_panel_cache(cache_path)
                    cache_loaded = True
                    print(f"[Inference] Loaded panel cache from {cache_path}")
                except Exception as exc:
                    print(f"[Inference] Failed to load panel cache ({exc}); rebuilding.")
                    self.cache = None

        if not cache_loaded:
            source_frame = frame
            if dataset_path is not None and Path(dataset_path).exists():
                source_frame = load_backtest_frame(
                    Path(dataset_path),
                    start_date=None,
                    end_date=None,
                    feature_cols=list(self.feature_cols),
                    lookback=self.lookback,
                    aliases_yaml=self.aliases_yaml,
                )

            feature_frame = source_frame.select([self.date_col, self.code_col] + list(self.feature_cols))

            # Apply CS-Z normalization if enabled
            if self.add_csz:
                normalization_cfg = config.get("normalization", {})  # type: ignore[call-arg]
                clip_sigma = normalization_cfg.get("clip_sigma", 5.0)
                feature_frame = add_cross_sectional_zscores(
                    feature_frame,
                    columns=self.feature_cols,
                    date_col=self.date_col,
                    clip_sigma=clip_sigma,
                )
                # Use CS-Z features only (matching training behavior)
                cache_feature_cols = self.z_features
            else:
                # Use raw features only
                cache_feature_cols = self.feature_cols

            unique_days = feature_frame.select(self.date_col).unique().height
            if unique_days < self.lookback:
                raise ValueError(
                    "Insufficient trading history to build panel cache: "
                    f"found {unique_days} days, require >= {self.lookback}."
                )

            self.cache = build_panel_cache(
                feature_frame,
                feature_cols=cache_feature_cols,
                target_cols=[],
                mask_cols=[],
                date_col=self.date_col,
                code_col=self.code_col,
                lookback=self.lookback,
                min_stocks_per_day=0,
            )
            if cache_path is not None:
                try:
                    save_panel_cache(self.cache, cache_path)
                    print(f"[Inference] Saved panel cache to {cache_path}")
                except Exception as exc:
                    print(f"[Inference] Failed to save panel cache: {exc}")

        # cache constructed; horizons already initialised

    def available_dates(self) -> set[Date]:
        """Return set of dates for which inference can be generated."""
        return {int_to_date(date_int) for date_int in self.cache.date_to_codes.keys()}

    def _tensor_for_date(
        self,
        target_date: Date,
    ) -> tuple[torch.Tensor | None, list[str]]:
        date_int = date_to_int(target_date)
        codes = self.cache.date_to_codes.get(date_int)
        if not codes:
            return None, []

        feature_windows: list[np.ndarray] = []
        valid_codes: list[str] = []

        for code in codes:
            payload = self.cache.codes.get(code)
            if payload is None:
                continue
            dates = payload["dates"]
            idx = np.searchsorted(dates, date_int)
            if idx == len(dates) or dates[idx] != date_int:
                continue
            start = idx - self.lookback + 1
            if start < 0:
                continue
            window = payload["features"][start : idx + 1]
            if window.shape[0] != self.lookback:
                continue
            feature_windows.append(window)
            valid_codes.append(code)

        if not feature_windows:
            return None, []

        features = np.stack(feature_windows, axis=0).astype(np.float32, copy=False)

        # Note: When add_csz=True, cache contains CS-Z features only (z_features)
        # These are already normalized and ready to use - no additional processing needed
        # The cache was built with feature_cols=self.z_features, so features from cache
        # are already CS-Z normalized values, matching training behavior

        # Fail-fast check (Phase 1.2): Use model's expected dimension as single source of truth
        # When add_csz=True: cache contains CS-Z features (same count as raw features)
        # When add_csz=False: cache contains raw features
        # In both cases, feature count equals n_features (raw feature count)
        expected_dim = self.model.in_features
        if features.shape[-1] != expected_dim:
            raise ValueError(
                f"❌ Dimension mismatch at {target_date}!\n"
                f"   Model expects: {expected_dim} features (in_features)\n"
                f"   Data provides: {features.shape[-1]} features\n"
                f"   Raw features: {len(self.feature_cols)}\n"
                f"   CS-Z enabled: {self.add_csz}\n"
                f"   Cache contains: {'CS-Z features only' if self.add_csz else 'raw features'}\n"
                f"   First 3 features: {self.feature_cols[:3]}\n"
                f"   This indicates model/data configuration mismatch!"
            )

        tensor = torch.from_numpy(features)
        return tensor, valid_codes

    def predict(
        self,
        target_date: Date,
        horizon: int,
        *,
        top_k: int | None = None,
    ) -> pl.DataFrame:
        if horizon not in self.horizons:
            raise ValueError(f"Horizon {horizon} not available. Options: {sorted(self.horizons)}")

        tensor, codes = self._tensor_for_date(target_date)
        if tensor is None or not codes:
            return pl.DataFrame({"Date": [], "Rank": [], "Code": [], "Score": [], "Horizon": []})

        tensor = tensor.to(self.device)
        with torch.no_grad():
            output = self.model(tensor)

        scores = output[horizon].detach().cpu().numpy()
        ranked_idx = np.argsort(scores)[::-1]
        if top_k is not None:
            ranked_idx = ranked_idx[:top_k]

        ranked_codes = [codes[i] for i in ranked_idx]
        ranked_scores = [float(scores[i]) for i in ranked_idx]
        ranks = list(range(1, len(ranked_codes) + 1))

        return pl.DataFrame(
            {
                "Date": [str(target_date)] * len(ranks),
                "Rank": ranks,
                "Code": ranked_codes,
                "Score": ranked_scores,
                "Horizon": [f"{horizon}d"] * len(ranks),
            }
        )


__all__ = [
    "BacktestInferenceEngine",
    "compute_weight_turnover",
    "date_to_int",
    "ensure_date",
    "int_to_date",
    "load_model_checkpoint",
    "resolve_device",
]
