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
) -> APEXRankerV0:
    """Instantiate ``APEXRankerV0`` and load weights onto ``device``."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model_cfg = config["model"]  # type: ignore[index]
    horizons = config["train"]["horizons"]  # type: ignore[index]

    model = APEXRankerV0(
        in_features=n_features,
        horizons=horizons,
        d_model=model_cfg["d_model"],  # type: ignore[index]
        depth=model_cfg["depth"],  # type: ignore[index]
        patch_len=model_cfg["patch_len"],  # type: ignore[index]
        stride=model_cfg["stride"],  # type: ignore[index]
        n_heads=model_cfg["n_heads"],  # type: ignore[index]
        dropout=model_cfg.get("dropout", 0.1),  # type: ignore[call-arg]
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_weight_turnover(
    current_weights: Mapping[str, float],
    target_weights: Mapping[str, float],
) -> float:
    """Calculate turnover implied by weight change."""
    codes = set(current_weights.keys()) | set(target_weights.keys())
    total_change = sum(
        abs(target_weights.get(code, 0.0) - current_weights.get(code, 0.0))
        for code in codes
    )
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
    ) -> None:
        self.config = config
        self.device = resolve_device(device)
        data_cfg = config["data"]  # type: ignore[index]
        self.date_col = data_cfg["date_column"]  # type: ignore[index]
        self.code_col = data_cfg["code_column"]  # type: ignore[index]
        self.lookback = data_cfg["lookback"]  # type: ignore[index]
        self.feature_cols = list(feature_cols)
        self.z_features = [f"{col}_cs_z" for col in self.feature_cols]

        self.model = load_model_checkpoint(
            model_path=model_path,
            config=config,
            device=self.device,
            n_features=len(self.feature_cols),
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
            )
            cache_path = cache_dir / f"{cache_key}.pkl"
            if cache_path.exists():
                try:
                    self.cache = load_panel_cache(cache_path)
                    cache_loaded = True
                    print(f"[Inference] Loaded panel cache from {cache_path}")
                except Exception as exc:
                    print(
                        f"[Inference] Failed to load panel cache ({exc}); rebuilding."
                    )
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
                )

            normalization_cfg = config.get("normalization", {})  # type: ignore[call-arg]
            clip_sigma = normalization_cfg.get("clip_sigma", 5.0)
            feature_frame = source_frame.select(
                [self.date_col, self.code_col] + list(self.feature_cols)
            )
            feature_frame = add_cross_sectional_zscores(
                feature_frame,
                columns=self.feature_cols,
                date_col=self.date_col,
                clip_sigma=clip_sigma,
            )

            unique_days = feature_frame.select(self.date_col).unique().height
            if unique_days < self.lookback:
                raise ValueError(
                    "Insufficient trading history to build panel cache: "
                    f"found {unique_days} days, require >= {self.lookback}."
                )

            self.cache = build_panel_cache(
                feature_frame,
                feature_cols=self.z_features,
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

        self.horizons = set(config["train"]["horizons"])  # type: ignore[index]

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
            raise ValueError(
                f"Horizon {horizon} not available. Options: {sorted(self.horizons)}"
            )

        tensor, codes = self._tensor_for_date(target_date)
        if tensor is None or not codes:
            return pl.DataFrame(
                {"Date": [], "Rank": [], "Code": [], "Score": [], "Horizon": []}
            )

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
