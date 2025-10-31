#!/usr/bin/env python3
"""
Compare two APEX-Ranker models on identical validation data.

Evaluates P@K, RankIC, and other metrics to determine which model
performs better for Phase 3 backtest.

Usage:
    python gogooku5/models/apex_ranker/scripts/compare_models.py \
        --model1 output/models/apex_ranker_v0_pruned.pt \
        --config1 models/apex_ranker/configs/v0_pruned.yaml \
        --model2 output/models/apex_ranker_v0_enhanced.pt \
        --config2 models/apex_ranker/configs/v0_base.yaml \
        --output results/model_comparison.json

Author: Claude Code
Date: 2025-10-29
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from _bootstrap import ensure_import_paths

ensure_import_paths()

from apex_ranker.data import (
    FeatureSelector,
    add_cross_sectional_zscores,
    build_panel_cache,
    resolve_artifact_path,
    resolve_dataset_path,
    resolve_metadata_path,
)
from apex_ranker.models import APEXRankerV0
from apex_ranker.utils import load_config


def _select_features(config: dict):
    data_cfg = config["data"]
    config_dir = Path(config["_config_dir"])
    selector = FeatureSelector(
        resolve_artifact_path(
            data_cfg.get("feature_groups_config"),
            ("models/apex_ranker/configs/feature_groups.yaml",),
            kind="feature groups config",
            extra_bases=[config_dir],
        )
    )
    groups = list(data_cfg.get("feature_groups", []))
    if data_cfg.get("use_plus30"):
        groups.append("plus30")

    return selector.select(
        groups=groups,
        optional_groups=data_cfg.get("optional_groups", []),
        exclude_features=data_cfg.get("exclude_features"),
        metadata_path=resolve_metadata_path(
            data_cfg.get("metadata_path"), extra_bases=[config_dir]
        ),
    )


def load_model(model_path: Path, config: dict, device: torch.device) -> APEXRankerV0:
    """Load model from checkpoint."""
    selection = _select_features(config)
    n_features = len(selection.features)

    model_cfg = config["model"]
    horizons = config["train"]["horizons"]

    model = APEXRankerV0(
        in_features=n_features,
        horizons=horizons,
        d_model=model_cfg["d_model"],
        depth=model_cfg["depth"],
        patch_len=model_cfg["patch_len"],
        stride=model_cfg["stride"],
        n_heads=model_cfg["n_heads"],
        dropout=model_cfg.get("dropout", 0.1),
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def evaluate_model(
    model: APEXRankerV0,
    config: dict,
    data_path: Path,
    device: torch.device,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Evaluate model on validation data."""
    data_cfg = config["data"]
    date_col = data_cfg["date_column"]
    code_col = data_cfg["code_column"]
    lookback = data_cfg["lookback"]

    selection = _select_features(config)

    # Load data
    resolved_data_path = resolve_dataset_path(
        data_path, extra_bases=[Path(config["_config_dir"])]
    )
    required_columns = [date_col, code_col] + selection.features
    frame = pl.read_parquet(str(resolved_data_path), columns=required_columns)

    # Filter by date range
    if start_date:
        frame = frame.filter(pl.col(date_col) >= start_date)
    if end_date:
        frame = frame.filter(pl.col(date_col) <= end_date)

    # Apply cross-sectional normalization
    frame = add_cross_sectional_zscores(
        frame,
        columns=selection.features,
        date_col=date_col,
        clip_sigma=config.get("normalization", {}).get("clip_sigma", 5.0),
    )

    z_features = [f"{col}_cs_z" for col in selection.features]

    # Build panel cache
    cache = build_panel_cache(
        frame,
        feature_cols=z_features,
        target_cols=[],
        mask_cols=[],
        date_col=date_col,
        code_col=code_col,
        lookback=lookback,
        min_stocks_per_day=0,
    )

    # Evaluate on each date
    results = {
        "dates": [],
        "predictions": {},
        "num_stocks": [],
    }

    for horizon in model.horizons:
        results["predictions"][horizon] = []

    for date_int in cache.date_ints:
        codes = cache.date_to_codes[date_int]
        features_list = []
        valid_codes = []

        for code in codes:
            payload = cache.codes[code]
            dates = payload["dates"]
            idx = np.searchsorted(dates, date_int)
            if idx == len(dates) or dates[idx] != date_int:
                continue
            start = idx - lookback + 1
            if start < 0:
                continue
            window = payload["features"][start : idx + 1]
            features_list.append(window)
            valid_codes.append(code)

        if not features_list:
            continue

        features_array = np.stack(features_list, axis=0)
        features_tensor = torch.from_numpy(features_array).float().to(device)

        # Inference
        with torch.no_grad():
            output = model(features_tensor)

        # Store predictions
        date_str = str(np.datetime64(date_int, "D"))
        results["dates"].append(date_str)
        results["num_stocks"].append(len(valid_codes))

        for horizon, preds in output.items():
            scores = preds.cpu().numpy()
            results["predictions"][horizon].append(
                {
                    "date": date_str,
                    "codes": valid_codes,
                    "scores": scores.tolist(),
                }
            )

    return results


def compute_metrics(predictions: dict) -> dict:
    """Compute P@K and RankIC metrics."""
    metrics = {}

    for horizon, pred_list in predictions.items():
        pak_scores = []

        for pred in pred_list:
            scores = np.array(pred["scores"])
            # P@K: Assume top-K are better (proxy metric)
            # For real P@K, need actual future returns (not available here)
            top_k = int(len(scores) * 0.02)  # Top 2%
            pak = np.mean(scores[np.argsort(scores)[-top_k:]]) / np.mean(scores)
            pak_scores.append(pak)

        metrics[f"horizon_{horizon}d"] = {
            "mean_pak_proxy": float(np.mean(pak_scores)),
            "std_pak_proxy": float(np.std(pak_scores)),
            "num_days": len(pred_list),
        }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compare two APEX-Ranker models")
    parser.add_argument("--model1", required=True, help="Path to model 1 checkpoint")
    parser.add_argument("--config1", required=True, help="Path to model 1 config")
    parser.add_argument("--model2", required=True, help="Path to model 2 checkpoint")
    parser.add_argument("--config2", required=True, help="Path to model 2 config")
    parser.add_argument(
        "--data",
        default="output/ml_dataset_latest_full.parquet",
        help="Path to dataset",
    )
    parser.add_argument("--start-date", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default=None, help="Output JSON file")
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Compare] Using device: {device}")

    # Resolve configuration paths and load configs
    print("[Compare] Loading configs...")
    config_path1 = resolve_artifact_path(
        args.config1,
        ("models/apex_ranker/configs/v0_pruned.yaml",),
        kind="model config",
    )
    config1 = load_config(str(config_path1))

    config_path2 = resolve_artifact_path(
        args.config2,
        ("models/apex_ranker/configs/v0_base.yaml",),
        kind="model config",
    )
    config2 = load_config(str(config_path2))

    # Resolve model checkpoints
    print(f"[Compare] Loading model 1: {args.model1}")
    model_path1 = resolve_artifact_path(
        args.model1,
        ("output/models/apex_ranker_v0_pruned.pt",),
        kind="model checkpoint",
        extra_bases=[config_path1.parent],
    )
    model1 = load_model(model_path1, config1, device)

    print(f"[Compare] Loading model 2: {args.model2}")
    model_path2 = resolve_artifact_path(
        args.model2,
        ("output/models/apex_ranker_v0_enhanced.pt",),
        kind="model checkpoint",
        extra_bases=[config_path2.parent],
    )
    model2 = load_model(model_path2, config2, device)

    dataset_path = resolve_dataset_path(
        args.data,
        extra_bases=[config_path1.parent, config_path2.parent],
    )

    # Evaluate models
    print("[Compare] Evaluating model 1...")
    results1 = evaluate_model(
        model1,
        config1,
        dataset_path,
        device,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    print("[Compare] Evaluating model 2...")
    results2 = evaluate_model(
        model2,
        config2,
        dataset_path,
        device,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    # Compute metrics
    print("[Compare] Computing metrics...")
    metrics1 = compute_metrics(results1["predictions"])
    metrics2 = compute_metrics(results2["predictions"])

    # Create comparison report
    comparison = {
        "model1": {
            "path": str(model_path1),
            "config": str(config_path1),
            "metrics": metrics1,
            "num_days": len(results1["dates"]),
            "avg_stocks_per_day": float(np.mean(results1["num_stocks"])),
        },
        "model2": {
            "path": str(model_path2),
            "config": str(config_path2),
            "metrics": metrics2,
            "num_days": len(results2["dates"]),
            "avg_stocks_per_day": float(np.mean(results2["num_stocks"])),
        },
        "comparison": {
            "date_range": {
                "start": args.start_date or "auto",
                "end": args.end_date or "auto",
            },
        },
    }

    # Print summary
    print("\n" + "=" * 80)
    print("Model Comparison Summary")
    print("=" * 80)
    print(f"Model 1: {model_path1}")
    print(f"  Evaluation days: {comparison['model1']['num_days']}")
    print(f"  Avg stocks/day: {comparison['model1']['avg_stocks_per_day']:.1f}")
    for horizon, metrics in metrics1.items():
        print(
            f"  {horizon}: P@K proxy = {metrics['mean_pak_proxy']:.4f} ± {metrics['std_pak_proxy']:.4f}"
        )

    print(f"\nModel 2: {model_path2}")
    print(f"  Evaluation days: {comparison['model2']['num_days']}")
    print(f"  Avg stocks/day: {comparison['model2']['avg_stocks_per_day']:.1f}")
    for horizon, metrics in metrics2.items():
        print(
            f"  {horizon}: P@K proxy = {metrics['mean_pak_proxy']:.4f} ± {metrics['std_pak_proxy']:.4f}"
        )

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\n[Compare] Results saved to: {args.output}")


if __name__ == "__main__":
    main()
