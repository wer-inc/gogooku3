#!/usr/bin/env python3
"""
LightGBM Baseline for Stock Prediction

Purpose: Establish baseline performance before deep learning
Expected: Sharpe 0.10-0.15 (if higher, deep learning may be unnecessary)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import mean_squared_error

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# No imports needed for manual splitting

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def calculate_sharpe_ratio(
    returns: np.ndarray, annualization_factor: float = 252
) -> float:
    """Calculate Sharpe Ratio"""
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(annualization_factor))


def calculate_ic(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate Information Coefficient (Pearson correlation)"""
    if len(predictions) != len(targets) or len(predictions) == 0:
        return 0.0

    # Remove NaN
    mask = ~(np.isnan(predictions) | np.isnan(targets))
    pred_clean = predictions[mask]
    tgt_clean = targets[mask]

    if len(pred_clean) < 10:
        return 0.0

    return float(np.corrcoef(pred_clean, tgt_clean)[0, 1])


def calculate_rank_ic(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate Rank IC (Spearman correlation)"""
    from scipy.stats import spearmanr

    if len(predictions) != len(targets) or len(predictions) == 0:
        return 0.0

    # Remove NaN
    mask = ~(np.isnan(predictions) | np.isnan(targets))
    pred_clean = predictions[mask]
    tgt_clean = targets[mask]

    if len(pred_clean) < 10:
        return 0.0

    corr, _ = spearmanr(pred_clean, tgt_clean)
    return float(corr) if not np.isnan(corr) else 0.0


def create_walk_forward_splits(
    dates: np.ndarray,
    n_splits: int = 5,
    embargo_days: int = 20,
    min_train_days: int = 252,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Create walk-forward splits manually

    Returns:
        List of (train_indices, val_indices) tuples
    """
    # Get unique dates and sort
    unique_dates = np.unique(dates)
    unique_dates = np.sort(unique_dates)

    total_days = len(unique_dates)
    test_days = (total_days - min_train_days) // n_splits

    splits = []

    for i in range(n_splits):
        # Train period
        train_end_idx = min_train_days + i * test_days
        train_end_date = unique_dates[train_end_idx]

        # Embargo
        embargo_date = train_end_date + np.timedelta64(embargo_days, "D")

        # Test period
        test_end_idx = min(train_end_idx + test_days + embargo_days, total_days - 1)
        test_end_date = unique_dates[test_end_idx]

        # Create masks
        train_mask = dates <= train_end_date
        test_mask = (dates >= embargo_date) & (dates <= test_end_date)

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM Baseline")
    parser.add_argument(
        "--data-path",
        type=str,
        default="output/ml_dataset_latest_full.parquet",
        help="Path to dataset",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="target_5d",
        help="Target column (target_1d, target_5d, target_10d, target_20d)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=1000,
        help="Number of boosting iterations",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=8,
        help="Maximum tree depth",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Learning rate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/baselines",
        help="Output directory",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("LightGBM Baseline Training")
    logger.info("=" * 80)
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Target: {args.target}")
    logger.info(f"N estimators: {args.n_estimators}")
    logger.info(f"Max depth: {args.max_depth}")
    logger.info(f"Learning rate: {args.learning_rate}")

    # Load dataset
    logger.info("Loading dataset...")
    df = pl.read_parquet(args.data_path)
    logger.info(f"Dataset shape: {df.shape}")

    # Define feature columns (exclude metadata and targets)
    exclude_cols = [
        "Code",
        "Date",
        "Section",
        "MarketCode",
        "LocalCode",
        "CompanyName",
        "row_idx",
        "sector17_code",
        "sector17_name",
        "sector17_id",
        "sector33_code",
        "sector33_name",
        "sector33_id",
        "target_1d",
        "target_5d",
        "target_10d",
        "target_20d",
        "target_1d_binary",
        "target_5d_binary",
        "target_10d_binary",
        "target_20d_binary",
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]
    logger.info(f"Feature columns: {len(feature_cols)}")

    # Check target exists
    if args.target not in df.columns:
        logger.error(f"Target column '{args.target}' not found in dataset!")
        logger.info(
            f"Available target columns: {[c for c in df.columns if 'target' in c]}"
        )
        sys.exit(1)

    # Prepare data
    logger.info("Preparing data...")

    # Convert to numeric only (exclude string columns that might have snuck in)
    numeric_cols = []
    for col in feature_cols:
        dtype = df[col].dtype
        # Only include numeric types
        if dtype in [
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
        ]:
            numeric_cols.append(col)
        else:
            logger.warning(f"Skipping non-numeric column: {col} (dtype: {dtype})")

    logger.info(f"Selected {len(numeric_cols)}/{len(feature_cols)} numeric features")
    feature_cols = numeric_cols

    # Fast conversion via Pandas (10-20x faster than Polars direct to_numpy)
    logger.info("Converting to NumPy arrays (via Pandas for speed)...")
    X = df.select(feature_cols).to_pandas().values
    y = df.select(args.target).to_pandas().values.flatten()
    dates = df.select("Date").to_pandas().values.flatten()
    codes = df.select("Code").to_pandas().values.flatten()

    # Remove NaN in target
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    dates = dates[mask]
    codes = codes[mask]

    logger.info(f"After NaN removal: {X.shape[0]:,} samples")

    # Walk-forward split
    logger.info("Creating walk-forward splits...")

    # Convert dates to datetime if needed
    if dates.dtype != np.dtype("datetime64[ns]"):
        dates = dates.astype("datetime64[ns]")

    splits = create_walk_forward_splits(
        dates=dates, n_splits=5, embargo_days=20, min_train_days=252
    )
    logger.info(f"Created {len(splits)} walk-forward splits")

    # Train on each split
    results = []
    all_val_preds = []
    all_val_targets = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Fold {fold_idx + 1}/{len(splits)}")
        logger.info(f"{'=' * 60}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        logger.info(f"Train: {len(X_train):,} samples")
        logger.info(f"Val: {len(X_val):,} samples")

        # LightGBM can handle NaN natively, no preprocessing needed

        # Train LightGBM
        logger.info("Training LightGBM...")
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
        val_data = lgb.Dataset(
            X_val, label=y_val, reference=train_data, feature_name=feature_cols
        )

        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 2**args.max_depth,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "seed": 42,
            # GPU acceleration
            "device": "gpu",
            "gpu_platform_id": 0,
            "gpu_device_id": 0,
        }

        model = lgb.train(
            params,
            train_data,
            num_boost_round=args.n_estimators,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )

        # Predict
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        ic = calculate_ic(val_pred, y_val)
        rank_ic = calculate_rank_ic(val_pred, y_val)

        # Sharpe (using predicted returns as portfolio returns)
        sharpe = calculate_sharpe_ratio(val_pred)

        logger.info(f"\nFold {fold_idx + 1} Results:")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  IC: {ic:.6f}")
        logger.info(f"  Rank IC: {rank_ic:.6f}")
        logger.info(f"  Sharpe (pred): {sharpe:.6f}")

        results.append(
            {
                "fold": fold_idx + 1,
                "rmse": float(rmse),
                "ic": float(ic),
                "rank_ic": float(rank_ic),
                "sharpe": float(sharpe),
                "n_train": len(X_train),
                "n_val": len(X_val),
                "best_iteration": model.best_iteration,
            }
        )

        all_val_preds.extend(val_pred.tolist())
        all_val_targets.extend(y_val.tolist())

        # Save model for first fold
        if fold_idx == 0:
            model_path = output_dir / "lgbm_baseline.txt"
            model.save_model(str(model_path))
            logger.info(f"Model saved: {model_path}")

    # Calculate overall metrics
    logger.info(f"\n{'=' * 80}")
    logger.info("Overall Results (All Folds)")
    logger.info(f"{'=' * 80}")

    avg_ic = np.mean([r["ic"] for r in results])
    avg_rank_ic = np.mean([r["rank_ic"] for r in results])
    avg_sharpe = np.mean([r["sharpe"] for r in results])

    # Calculate IC/RankIC on all validation predictions
    overall_ic = calculate_ic(np.array(all_val_preds), np.array(all_val_targets))
    overall_rank_ic = calculate_rank_ic(
        np.array(all_val_preds), np.array(all_val_targets)
    )
    overall_sharpe = calculate_sharpe_ratio(np.array(all_val_preds))

    logger.info(f"Average IC: {avg_ic:.6f}")
    logger.info(f"Average Rank IC: {avg_rank_ic:.6f}")
    logger.info(f"Average Sharpe: {avg_sharpe:.6f}")
    logger.info(f"\nOverall IC (all val): {overall_ic:.6f}")
    logger.info(f"Overall Rank IC (all val): {overall_rank_ic:.6f}")
    logger.info(f"Overall Sharpe (all val): {overall_sharpe:.6f}")

    # Save results
    result_summary = {
        "config": {
            "data_path": args.data_path,
            "target": args.target,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "n_features": len(feature_cols),
        },
        "fold_results": results,
        "average_metrics": {
            "ic": float(avg_ic),
            "rank_ic": float(avg_rank_ic),
            "sharpe": float(avg_sharpe),
        },
        "overall_metrics": {
            "ic": float(overall_ic),
            "rank_ic": float(overall_rank_ic),
            "sharpe": float(overall_sharpe),
            "n_samples": len(all_val_preds),
        },
    }

    result_path = output_dir / "lgbm_baseline_results.json"
    with open(result_path, "w") as f:
        json.dump(result_summary, f, indent=2)
    logger.info(f"\nResults saved: {result_path}")

    # Decision guidance
    logger.info(f"\n{'=' * 80}")
    logger.info("🎯 Decision Guidance")
    logger.info(f"{'=' * 80}")

    if overall_sharpe >= 0.15:
        logger.info(
            "✅ Sharpe >= 0.15: LightGBM is sufficient! Deep learning may be unnecessary."
        )
        logger.info("   Recommendation: Deploy LightGBM as production model.")
    elif overall_sharpe >= 0.10:
        logger.info("⚠️  Sharpe 0.10-0.15: Moderate performance.")
        logger.info(
            "   Recommendation: Try Hybrid (LightGBM + Lightweight ATFT) for improvement."
        )
    else:
        logger.info("❌ Sharpe < 0.10: Low performance.")
        logger.info(
            "   Recommendation: Feature engineering required or try innovative approaches."
        )

    logger.info("\n🎉 LightGBM baseline training complete!")


if __name__ == "__main__":
    main()
