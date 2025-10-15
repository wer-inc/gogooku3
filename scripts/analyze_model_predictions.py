#!/usr/bin/env python3
"""
Model Prediction Analysis Script

Loads trained ATFT-GAT-FAN model and analyzes predictions to identify
why Sharpe ratio is low despite good IC/RankIC performance.

Usage:
    python scripts/analyze_model_predictions.py \
        --model-path models/checkpoints/atft_gat_fan_final.pt \
        --data-path output/ml_dataset_latest_full.parquet \
        --config configs/atft/config_production_optimized.yaml \
        --output output/analysis/prediction_analysis.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from omegaconf import OmegaConf
from scipy import stats as scipy_stats
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.atft_gat_fan.models.architectures.atft_gat_fan import ATFT_GAT_FAN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_model(
    model_path: Path, config_path: Path, device: str = "cuda"
) -> ATFT_GAT_FAN:
    """Load trained ATFT-GAT-FAN model"""
    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Use config from checkpoint if available, otherwise load from file
    if "config" in checkpoint:
        logger.info("Using config from checkpoint")
        config = checkpoint["config"]
        # Convert to OmegaConf if it's a dict
        if isinstance(config, dict):
            config = OmegaConf.create(config)
    else:
        logger.info(f"Loading config from {config_path}")
        config = OmegaConf.load(config_path)

    # Initialize model
    model = ATFT_GAT_FAN(config)

    # Load state dict (handle different key names)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    # Use strict=False to handle potential architecture mismatches
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        logger.warning(f"Missing keys in checkpoint: {len(missing_keys)} keys")
    if unexpected_keys:
        logger.warning(f"Unexpected keys in checkpoint: {len(unexpected_keys)} keys (ignoring)")

    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded successfully ({sum(p.numel() for p in model.parameters()):,} parameters)")
    return model


def load_dataset(data_path: Path, test_ratio: float = 0.2) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load dataset and split into train/test"""
    logger.info(f"Loading dataset from {data_path}")
    df = pl.read_parquet(data_path)
    logger.info(f"Dataset loaded: {len(df):,} rows, {len(df.columns)} columns")

    # Sort by date
    df = df.sort("Date")

    # Split by date (last 20% as test)
    unique_dates = df.select("Date").unique().sort("Date")
    n_dates = len(unique_dates)
    split_idx = int(n_dates * (1 - test_ratio))

    split_date = unique_dates[split_idx, "Date"]
    logger.info(f"Test split date: {split_date}")

    train_df = df.filter(pl.col("Date") < split_date)
    test_df = df.filter(pl.col("Date") >= split_date)

    logger.info(f"Train set: {len(train_df):,} rows")
    logger.info(f"Test set: {len(test_df):,} rows")

    return train_df, test_df


def prepare_features(df: pl.DataFrame, feature_cols: list[str]) -> torch.Tensor:
    """Prepare feature tensor from dataframe"""
    # Extract features
    features = df.select(feature_cols).to_numpy().astype(np.float32)

    # Handle NaN/Inf
    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

    return torch.from_numpy(features)


def generate_predictions(
    model: ATFT_GAT_FAN,
    test_df: pl.DataFrame,
    batch_size: int = 1024,
    device: str = "cuda",
) -> dict[str, np.ndarray]:
    """Generate predictions for test set"""
    logger.info("Generating predictions on test set...")

    # Identify feature columns (exclude metadata and targets)
    exclude_cols = {
        "Code",
        "Date",
        "Section",
        "MarketCode",
        "section_norm",
        "row_idx",
        "sector17_code",
        "sector17_name",
    }
    # Exclude target columns
    target_patterns = ["returns_", "feat_ret_", "horizon_"]
    all_cols = set(test_df.columns)
    feature_cols = sorted(
        [
            c
            for c in all_cols
            if c not in exclude_cols
            and not any(c.startswith(p) for p in target_patterns)
        ]
    )

    logger.info(f"Using {len(feature_cols)} features for prediction")

    # Prepare batches
    n_samples = len(test_df)
    predictions_by_horizon = {}

    with torch.no_grad():
        for start_idx in tqdm(range(0, n_samples, batch_size), desc="Predicting"):
            end_idx = min(start_idx + batch_size, n_samples)

            # Prepare batch
            batch_df = test_df[start_idx:end_idx]
            features = prepare_features(batch_df, feature_cols).to(device)

            # Forward pass
            batch_dict = {"dynamic_features": features}
            outputs = model.forward(batch_dict)

            predictions = outputs["predictions"]

            # Collect predictions by horizon
            for horizon_key, pred_tensor in predictions.items():
                # Take median quantile (index 2 for [0.1, 0.3, 0.5, 0.7, 0.9])
                median_pred = pred_tensor[:, 2].cpu().numpy()

                if horizon_key not in predictions_by_horizon:
                    predictions_by_horizon[horizon_key] = []

                predictions_by_horizon[horizon_key].append(median_pred)

    # Concatenate batches
    for horizon_key in predictions_by_horizon:
        predictions_by_horizon[horizon_key] = np.concatenate(
            predictions_by_horizon[horizon_key]
        )

    logger.info(f"Generated predictions for {len(predictions_by_horizon)} horizons")
    return predictions_by_horizon


def analyze_predictions(
    test_df: pl.DataFrame, predictions: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Analyze prediction quality and identify Sharpe issues"""
    logger.info("Analyzing predictions...")

    results = {}

    for horizon_key, pred_values in predictions.items():
        # Extract corresponding actual returns
        # Try multiple target column patterns
        target_col = None
        horizon_num = int(horizon_key.split("_")[1].replace("d", ""))

        for pattern in [f"returns_{horizon_num}d", f"feat_ret_{horizon_num}d"]:
            if pattern in test_df.columns:
                target_col = pattern
                break

        if target_col is None:
            logger.warning(f"No target column found for {horizon_key}")
            continue

        actual_values = test_df.select(target_col).to_numpy().flatten()

        # Remove NaN pairs
        valid_mask = ~(np.isnan(pred_values) | np.isnan(actual_values))
        pred_clean = pred_values[valid_mask]
        actual_clean = actual_values[valid_mask]

        if len(pred_clean) < 100:
            logger.warning(f"Insufficient valid samples for {horizon_key}: {len(pred_clean)}")
            continue

        # ===== Core Metrics =====
        # IC (Pearson correlation)
        ic, ic_pvalue = scipy_stats.pearsonr(pred_clean, actual_clean)

        # RankIC (Spearman correlation)
        rankic, rankic_pvalue = scipy_stats.spearmanr(pred_clean, actual_clean)

        # Sharpe Ratio (assuming equal-weighted positions based on predictions)
        # Simple strategy: long if pred > 0, short if pred < 0
        positions = np.sign(pred_clean)
        strategy_returns = positions * actual_clean

        mean_return = strategy_returns.mean()
        std_return = strategy_returns.std()
        sharpe = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0.0

        # ===== Turnover Analysis =====
        # Daily position changes (approximate)
        turnover = np.abs(np.diff(positions)).mean()

        # ===== Prediction Distribution =====
        pred_stats = {
            "mean": float(pred_clean.mean()),
            "std": float(pred_clean.std()),
            "min": float(pred_clean.min()),
            "max": float(pred_clean.max()),
            "q25": float(np.percentile(pred_clean, 25)),
            "q50": float(np.percentile(pred_clean, 50)),
            "q75": float(np.percentile(pred_clean, 75)),
        }

        # ===== Prediction Strength Analysis =====
        pred_abs = np.abs(pred_clean)
        strength_stats = {
            "mean": float(pred_abs.mean()),
            "std": float(pred_abs.std()),
            "q90": float(np.percentile(pred_abs, 90)),
        }

        # ===== Direction Accuracy =====
        pred_direction = pred_clean > 0
        actual_direction = actual_clean > 0
        hit_rate = (pred_direction == actual_direction).mean()

        # ===== Store Results =====
        results[horizon_key] = {
            "n_samples": int(len(pred_clean)),
            "ic": float(ic),
            "ic_pvalue": float(ic_pvalue),
            "rankic": float(rankic),
            "rankic_pvalue": float(rankic_pvalue),
            "sharpe": float(sharpe),
            "mean_return": float(mean_return),
            "std_return": float(std_return),
            "turnover": float(turnover),
            "hit_rate": float(hit_rate),
            "prediction_stats": pred_stats,
            "strength_stats": strength_stats,
        }

        logger.info(
            f"{horizon_key}: IC={ic:.4f}, RankIC={rankic:.4f}, Sharpe={sharpe:.4f}, "
            f"HitRate={hit_rate:.4f}, Turnover={turnover:.4f}"
        )

    # ===== Sector-wise Analysis =====
    if "sector17_name" in test_df.columns:
        logger.info("Performing sector-wise analysis...")
        sector_results = analyze_by_sector(test_df, predictions)
        results["sector_analysis"] = sector_results

    return results


def analyze_by_sector(
    test_df: pl.DataFrame, predictions: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Analyze predictions by sector"""
    sector_results = {}

    # Use horizon_1d for sector analysis (most samples)
    if "horizon_1d" not in predictions:
        return {}

    pred_1d = predictions["horizon_1d"]

    # Find target column
    target_col = None
    for col in ["returns_1d", "feat_ret_1d"]:
        if col in test_df.columns:
            target_col = col
            break

    if target_col is None:
        return {}

    # Add predictions to dataframe
    test_with_pred = test_df.with_columns(
        pl.Series("pred_1d", pred_1d)
    )

    # Group by sector
    sectors = test_with_pred.select("sector17_name").unique().to_series().to_list()

    for sector in sectors:
        if sector is None:
            continue

        sector_df = test_with_pred.filter(pl.col("sector17_name") == sector)

        pred_sector = sector_df.select("pred_1d").to_numpy().flatten()
        actual_sector = sector_df.select(target_col).to_numpy().flatten()

        # Remove NaN
        valid_mask = ~(np.isnan(pred_sector) | np.isnan(actual_sector))
        pred_clean = pred_sector[valid_mask]
        actual_clean = actual_sector[valid_mask]

        if len(pred_clean) < 30:
            continue

        # Calculate metrics
        ic, _ = scipy_stats.pearsonr(pred_clean, actual_clean)

        positions = np.sign(pred_clean)
        strategy_returns = positions * actual_clean
        mean_return = strategy_returns.mean()
        std_return = strategy_returns.std()
        sharpe = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0.0

        sector_results[sector] = {
            "n_samples": int(len(pred_clean)),
            "ic": float(ic),
            "sharpe": float(sharpe),
            "mean_return": float(mean_return),
        }

    return sector_results


def main():
    parser = argparse.ArgumentParser(description="Analyze model predictions")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/checkpoints/atft_gat_fan_final.pt"),
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("output/ml_dataset_latest_full.parquet"),
        help="Path to ML dataset",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/atft/config_production_optimized.yaml"),
        help="Path to model config",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/analysis/prediction_analysis.json"),
        help="Path to save analysis results",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Ratio of dataset to use as test set",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2048, help="Batch size for prediction"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference",
    )

    args = parser.parse_args()

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.model_path, args.config, args.device)

    # Load dataset
    train_df, test_df = load_dataset(args.data_path, args.test_ratio)

    # Generate predictions
    predictions = generate_predictions(model, test_df, args.batch_size, args.device)

    # Analyze predictions
    analysis_results = analyze_predictions(test_df, predictions)

    # Save results
    logger.info(f"Saving analysis results to {args.output}")
    with args.output.open("w") as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("PREDICTION ANALYSIS SUMMARY")
    logger.info("=" * 80)

    for horizon_key, metrics in analysis_results.items():
        if horizon_key == "sector_analysis":
            continue

        logger.info(f"\n{horizon_key}:")
        logger.info(f"  IC:       {metrics['ic']:>8.4f} (p={metrics['ic_pvalue']:.4f})")
        logger.info(f"  RankIC:   {metrics['rankic']:>8.4f} (p={metrics['rankic_pvalue']:.4f})")
        logger.info(f"  Sharpe:   {metrics['sharpe']:>8.4f}")
        logger.info(f"  HitRate:  {metrics['hit_rate']:>8.4f}")
        logger.info(f"  Turnover: {metrics['turnover']:>8.4f}")

    logger.info("\n" + "=" * 80)
    logger.info("Analysis complete!")
    logger.info(f"Results saved to: {args.output}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
