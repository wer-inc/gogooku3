#!/usr/bin/env python3
"""
Permutation-based feature importance for APEX-Ranker v0.

The script loads a trained checkpoint, generates validation predictions,
then measures how much each feature influences the chosen evaluation metric
by shuffling its values across the cross-section.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
import yaml
from _bootstrap import ensure_import_paths
from gogooku5.data.src.builder.utils.lazy_io import lazy_load
from tqdm import tqdm

ensure_import_paths()

from apex_ranker.data import (  # noqa: E402
    DayPanelDataset,
    FeatureSelector,
    add_cross_sectional_zscores,
    build_panel_cache,
)
from apex_ranker.models import APEXRankerV0  # noqa: E402
from apex_ranker.utils import precision_at_k, spearman_ic  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Permutation feature importance for APEX-Ranker v0"
    )
    parser.add_argument("--model", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument("--output", required=True, help="Path to JSON report")
    parser.add_argument(
        "--horizon", type=int, default=20, help="Prediction horizon to evaluate"
    )
    parser.add_argument(
        "--metric", choices=["pak", "rankic"], default="pak", help="Evaluation metric"
    )
    parser.add_argument(
        "--top-k", type=int, default=50, help="Top-K for P@K (when metric=pak)"
    )
    parser.add_argument(
        "--val-days", type=int, default=None, help="Override validation days"
    )
    parser.add_argument(
        "--num-permutations",
        type=int,
        default=3,
        help="Permutation repeats per feature",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Limit number of features (for debug)",
    )
    parser.add_argument(
        "--max-days", type=int, default=None, help="Limit number of validation days"
    )
    parser.add_argument("--device", default="auto", help="Device cuda/cpu/auto")
    return parser.parse_args()


def load_model_and_config(
    model_path: str, config_path: str
) -> tuple[APEXRankerV0, dict]:
    with open(config_path, encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    checkpoint = torch.load(model_path, map_location="cpu")
    model_state = checkpoint.get("model_state_dict", checkpoint)

    attn_key = "encoder.blocks.0.attn.in_proj_weight"
    if attn_key in model_state:
        d_model = model_state[attn_key].shape[1]
    else:
        d_model = config["model"].get("d_model", 256)

    patch_key = "encoder.patch_embed.conv.weight"
    patch_len = (
        model_state.get(patch_key).shape[2]
        if patch_key in model_state
        else config["model"].get("patch_len", 16)
    )

    depth = None
    prefix = "encoder.blocks."
    for key in model_state.keys():
        if key.startswith(prefix):
            try:
                idx = int(key.split(".")[2])
                depth = max(depth or 0, idx + 1)
            except (IndexError, ValueError):
                continue
    depth = depth or config["model"].get("depth", 4)

    data_cfg = config["data"]
    feature_selector = FeatureSelector(data_cfg["feature_groups_config"])
    groups = list(data_cfg.get("feature_groups", []))
    if data_cfg.get("use_plus30", True):
        groups = groups + ["plus30"]
    selection = feature_selector.select(
        groups=groups,
        optional_groups=data_cfg.get("optional_groups", []),
        metadata_path=data_cfg.get("metadata_path"),
    )

    horizons = [int(h) for h in config["train"]["horizons"]]
    model = APEXRankerV0(
        in_features=len(selection.features),
        horizons=horizons,
        d_model=d_model,
        depth=depth,
        patch_len=patch_len,
        stride=config["model"].get("stride", 8),
        n_heads=config["model"].get("n_heads", 8),
        dropout=config["model"].get("dropout", 0.2),
    )
    model.load_state_dict(model_state)
    model.eval()
    return model, config


def prepare_dataset(config: dict, val_days: int | None, max_days: int | None):
    data_cfg = config["data"]
    # Use lazy_load for IPC cache support (3-5x faster reads)
    df = lazy_load(data_cfg["parquet_path"], prefer_ipc=True)

    feature_selector = FeatureSelector(data_cfg["feature_groups_config"])
    groups = list(data_cfg.get("feature_groups", []))
    if data_cfg.get("use_plus30", True):
        groups = groups + ["plus30"]

    selection = feature_selector.select(
        groups=groups,
        optional_groups=data_cfg.get("optional_groups", []),
        metadata_path=data_cfg.get("metadata_path"),
    )

    df = add_cross_sectional_zscores(
        df,
        columns=selection.features,
        date_col=data_cfg["date_column"],
        clip_sigma=config.get("normalization", {}).get("clip_sigma", 5.0),
    )
    feature_cols = [f"{col}_cs_z" for col in selection.features]
    mask_cols = selection.masks

    target_map = data_cfg["target_columns"]
    horizons = [int(h) for h in config["train"]["horizons"]]

    def resolve_target(h: int) -> str:
        if isinstance(target_map, dict):
            if str(h) in target_map:
                return target_map[str(h)]
            if h in target_map:
                return target_map[h]
        raise KeyError(f"target column for horizon {h} is not defined")

    target_cols = [resolve_target(h) for h in horizons]

    val_days = val_days or config["train"].get("val_days", 120)
    date_col = data_cfg["date_column"]
    dates_series = df.select(pl.col(date_col).unique().sort()).to_series()
    dates = dates_series.to_list()
    date_ints = np.asarray(dates_series.to_numpy(), dtype="datetime64[D]").astype(
        "int64"
    )

    if val_days > len(dates):
        val_days = len(dates)

    lookback = data_cfg["lookback"]
    val_start_idx = max(0, len(dates) - val_days)
    data_start_idx = max(0, val_start_idx - lookback)

    val_start_date = dates[val_start_idx]
    data_start_date = dates[data_start_idx]

    df = df.filter(pl.col(date_col) >= data_start_date)
    print(
        f"[INFO] Validation period starts {val_start_date} (days={val_days}, lookback={lookback})"
    )

    cache = build_panel_cache(
        df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        mask_cols=mask_cols,
        date_col=date_col,
        code_col=data_cfg["code_column"],
        lookback=lookback,
        min_stocks_per_day=data_cfg["min_stocks_per_day"],
    )

    val_start_int = int(date_ints[val_start_idx])
    val_dates = [d for d in cache.date_ints if d >= val_start_int]
    if max_days is not None:
        val_dates = val_dates[-max_days:]

    dataset = DayPanelDataset(
        cache,
        feature_cols=feature_cols,
        mask_cols=mask_cols,
        target_cols=target_cols,
        dates_subset=val_dates,
    )
    print(f"[INFO] Panel days available: {len(dataset)} (used: {len(val_dates)})")
    return dataset, selection.features


def collect_day_batches(dataset: DayPanelDataset) -> list[dict[str, object]]:
    day_batches: list[dict[str, object]] = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        if item is None:
            continue
        day_batches.append(
            {
                "date": int(item["date_int"]),
                "X": item["X"].clone(),
                "y": item["y"].clone(),
                "codes": list(item["codes"]),
            }
        )
    return day_batches


def run_model_on_batches(
    model: APEXRankerV0,
    day_batches: list[dict[str, object]],
    horizon_value: int,
    horizon_idx: int,
    device: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    predictions: list[np.ndarray] = []
    actuals: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(day_batches, desc="Baseline predictions"):
            X = batch["X"].to(device)
            y = batch["y"]
            outputs = model(X)
            preds = outputs[horizon_value].detach().cpu().numpy()
            targets = y[:, horizon_idx].detach().cpu().numpy()
            predictions.append(preds)
            actuals.append(targets)
    return predictions, actuals


def evaluate_metric(
    predictions: list[np.ndarray],
    actuals: list[np.ndarray],
    metric: str,
    top_k: int,
) -> float:
    values: list[float] = []
    for preds, acts in zip(predictions, actuals, strict=False):
        scores = torch.from_numpy(preds)
        targets = torch.from_numpy(acts)
        if metric == "pak":
            k = max(1, min(top_k, scores.numel()))
            values.append(precision_at_k(scores, targets, k))
        else:
            values.append(spearman_ic(scores, targets))
    return float(np.mean(values)) if values else float("nan")


def permute_feature(
    day_batches: list[dict[str, object]], feature_idx: int
) -> list[torch.Tensor]:
    shuffled_inputs: list[torch.Tensor] = []
    for batch in day_batches:
        X = batch["X"].clone()
        stocks = X.shape[0]
        perm = torch.randperm(stocks)
        feature_slice = X[:, :, feature_idx]
        X[:, :, feature_idx] = feature_slice[perm, :]
        shuffled_inputs.append(X)
    return shuffled_inputs


def evaluate_permuted(
    model: APEXRankerV0,
    day_batches: list[dict[str, object]],
    shuffled_inputs: list[torch.Tensor],
    horizon_value: int,
    device: str,
) -> list[np.ndarray]:
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for _batch, X_perm in zip(day_batches, shuffled_inputs, strict=False):
            X = X_perm.to(device)
            outputs = model(X)
            preds.append(outputs[horizon_value].detach().cpu().numpy())
    return preds


def main() -> None:
    args = parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, config = load_model_and_config(args.model, args.config)
    dataset, feature_names = prepare_dataset(config, args.val_days, args.max_days)

    horizon_list = [int(h) for h in config["train"]["horizons"]]
    if args.horizon not in horizon_list:
        raise ValueError(
            f"Horizon {args.horizon} not in model horizons: {horizon_list}"
        )
    horizon_idx = horizon_list.index(args.horizon)

    try:
        model = model.to(device)
    except RuntimeError as exc:
        if device != "cpu":
            print(f"[WARN] Falling back to CPU due to device error: {exc}")
            device = "cpu"
            model = model.to(device)
        else:
            raise

    day_batches = collect_day_batches(dataset)
    if not day_batches:
        raise RuntimeError(
            "No valid day panels available for feature importance analysis"
        )

    baseline_preds, actuals = run_model_on_batches(
        model, day_batches, args.horizon, horizon_idx, device
    )
    baseline_metric = evaluate_metric(baseline_preds, actuals, args.metric, args.top_k)
    print(f"[INFO] Baseline {args.metric}: {baseline_metric:.6f}")

    max_features = args.max_features or len(feature_names)
    results: list[dict[str, object]] = []

    for feat_idx, feat_name in enumerate(feature_names[:max_features]):
        metric_values: list[float] = []
        print(f"[INFO] Permuting feature {feat_idx+1}/{max_features}: {feat_name}")
        for _repeat in range(args.num_permutations):
            shuffled_inputs = permute_feature(day_batches, feat_idx)
            permuted_preds = evaluate_permuted(
                model, day_batches, shuffled_inputs, args.horizon, device
            )
            metric_val = evaluate_metric(
                permuted_preds, actuals, args.metric, args.top_k
            )
            metric_values.append(metric_val)
        metric_values_np = np.array(metric_values, dtype=np.float64)
        importance = baseline_metric - metric_values_np.mean()
        results.append(
            {
                "feature": feat_name,
                "baseline_metric": baseline_metric,
                "permuted_metric_mean": float(metric_values_np.mean()),
                "permuted_metric_std": float(metric_values_np.std()),
                "importance": float(importance),
                "num_permutations": args.num_permutations,
            }
        )

    results_sorted = sorted(results, key=lambda x: x["importance"], reverse=True)
    report = {
        "model_path": args.model,
        "config_path": args.config,
        "metric": args.metric,
        "baseline_metric": baseline_metric,
        "horizon": args.horizon,
        "top_k": args.top_k,
        "num_permutations": args.num_permutations,
        "device": device,
        "results": results_sorted,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    print(f"[INFO] Feature importance saved to {output_path}")


if __name__ == "__main__":
    main()
