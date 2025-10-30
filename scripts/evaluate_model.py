"""Evaluate trained ATFT-GAT-FAN model and compute RankIC."""
import sys

sys.path.insert(0, "/workspace/gogooku3")

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_rankic(predictions, targets):
    """Compute Rank IC (Spearman correlation)."""
    from scipy.stats import spearmanr

    # Remove NaNs
    mask = ~(np.isnan(predictions) | np.isnan(targets))
    if mask.sum() < 10:
        return np.nan

    pred_clean = predictions[mask]
    targ_clean = targets[mask]

    if len(np.unique(pred_clean)) < 2 or len(np.unique(targ_clean)) < 2:
        return np.nan

    corr, pval = spearmanr(pred_clean, targ_clean)
    return corr


def evaluate(model_path, data_path, device="cuda"):
    """Evaluate model on validation data."""
    logger.info(f"Loading model from {model_path}")

    # Load model checkpoint (PyTorch 2.6 compatibility)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Get config if available
    if "config" in checkpoint:
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(checkpoint["config"])
        logger.info("✅ Config loaded from checkpoint")
    else:
        # Load default config
        from hydra import compose, initialize_config_dir

        config_dir = Path("/workspace/gogooku3/configs/atft").absolute()
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            cfg = compose(config_name="config_production_optimized")
        logger.info("✅ Using default config")

    # Initialize model
    from src.atft_gat_fan.models.architectures.atft_gat_fan import ATFT_GAT_FAN

    model = ATFT_GAT_FAN(cfg).to(device)

    # Load state dict
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    logger.info(
        f"✅ Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)"
    )

    # Setup data module
    from src.gogooku3.training.atft.data_module import ProductionDataModuleV2

    data_module = ProductionDataModuleV2(cfg)
    try:
        data_module.setup(stage="fit")
    except TypeError:
        data_module.setup()  # No stage argument
    val_loader = data_module.val_dataloader()

    logger.info("✅ Validation dataloader ready")

    # Collect predictions and targets
    all_predictions = {f"horizon_{h}": [] for h in [1, 5, 10, 20]}
    all_targets = {f"horizon_{h}": [] for h in [1, 5, 10, 20]}

    logger.info("🔄 Running inference on validation set...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
            try:
                features = batch["features"].to(device)
                targets = batch.get("targets", {})
                valid_masks = batch.get("valid_mask", {})

                # Forward pass
                outputs = model(features)

                # Extract predictions
                if isinstance(outputs, dict) and "predictions" in outputs:
                    predictions = outputs["predictions"]
                else:
                    predictions = outputs

                # Collect by horizon
                for h in [1, 5, 10, 20]:
                    pred_key = f"horizon_{h}d"
                    targ_key = f"horizon_{h}"
                    mask = None

                    if pred_key in predictions and targ_key in targets:
                        pred_tensor = predictions[pred_key].detach().cpu()
                        targ_tensor = targets[targ_key].detach().cpu()

                        if pred_tensor.ndim == 3 and pred_tensor.size(-1) == 1:
                            pred_tensor = pred_tensor[:, -1, 0]
                        elif pred_tensor.ndim == 3:
                            pred_tensor = pred_tensor[:, -1, :]
                        elif pred_tensor.ndim == 2:
                            pred_tensor = pred_tensor[:, -1]

                        if targ_tensor.ndim == 3 and targ_tensor.size(-1) == 1:
                            targ_tensor = targ_tensor[:, -1, 0]
                        elif targ_tensor.ndim == 3:
                            targ_tensor = targ_tensor[:, -1, :]
                        elif targ_tensor.ndim == 2:
                            targ_tensor = targ_tensor[:, -1]

                        candidate_keys = [
                            f"horizon_{h}",
                            f"horizon_{h}d",
                            f"target_{h}",
                            f"target_{h}d",
                        ]
                        if isinstance(valid_masks, dict):
                            for key in candidate_keys:
                                if key in valid_masks:
                                    mask = valid_masks[key].detach().cpu()
                                    break

                        if mask is None:
                            mask = torch.isfinite(targ_tensor)
                        if mask.dim() == targ_tensor.dim():
                            pass
                        elif mask.dim() + 1 == targ_tensor.dim():
                            mask = mask.unsqueeze(-1)
                        mask = mask.to(dtype=torch.bool)
                        if (
                            mask.shape != targ_tensor.shape
                            and mask.numel() == targ_tensor.numel()
                        ):
                            mask = mask.view(targ_tensor.shape)

                        pred = pred_tensor.numpy()
                        targ = targ_tensor.numpy()

                        valid_mask = (
                            np.isfinite(pred) & np.isfinite(targ) & mask.numpy()
                        )
                        if valid_mask.ndim > 1:
                            valid_mask = valid_mask.reshape(-1)
                        pred = pred.reshape(-1)
                        targ = targ.reshape(-1)
                        pred = pred[valid_mask]
                        targ = targ[valid_mask]

                        if pred.size > 0 and targ.size > 0:
                            all_predictions[f"horizon_{h}"].append(pred)
                            all_targets[f"horizon_{h}"].append(targ)

            except Exception as e:
                logger.warning(f"Batch {batch_idx} error: {e}")
                continue

    # Concatenate and compute metrics
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 70)

    results = {}
    for h in [1, 5, 10, 20]:
        key = f"horizon_{h}"

        if all_predictions[key] and all_targets[key]:
            preds = np.concatenate(all_predictions[key])
            targs = np.concatenate(all_targets[key])

            # Compute RankIC
            rankic = compute_rankic(preds, targs)

            # Compute RMSE
            mask = ~(np.isnan(preds) | np.isnan(targs))
            if mask.sum() > 0:
                rmse = np.sqrt(np.mean((preds[mask] - targs[mask]) ** 2))
            else:
                rmse = np.nan

            results[key] = {
                "rank_ic": rankic,
                "rmse": rmse,
                "n_samples": len(preds),
                "n_valid": mask.sum() if "mask" in locals() else 0,
            }

            logger.info(f"\n{h}d Horizon:")
            logger.info(
                f"  Rank IC: {rankic:.4f}" if not np.isnan(rankic) else "  Rank IC: NaN"
            )
            logger.info(f"  RMSE: {rmse:.4f}" if not np.isnan(rmse) else "  RMSE: NaN")
            logger.info(f"  Samples: {len(preds):,} ({mask.sum():,} valid)")
        else:
            logger.info(f"\n{h}d Horizon: No data collected")
            results[key] = {"rank_ic": np.nan, "rmse": np.nan, "n_samples": 0}

    # Overall summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    valid_rankics = [
        v["rank_ic"] for v in results.values() if not np.isnan(v["rank_ic"])
    ]
    if valid_rankics:
        avg_rankic = np.mean(valid_rankics)
        logger.info(f"Average Rank IC: {avg_rankic:.4f}")
        logger.info(f"Best Rank IC: {max(valid_rankics):.4f}")

        # Phase 2 target
        target = 0.020
        if avg_rankic >= target:
            logger.info(f"✅ Phase 2 target achieved! (≥{target:.3f})")
        else:
            logger.info(f"⚠️  Phase 2 target not met ({avg_rankic:.4f} < {target:.3f})")
    else:
        logger.info("❌ No valid Rank IC computed")

    logger.info("=" * 70)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--data", default="output/ml_dataset_latest_full.parquet")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-rankic", action="store_true")

    args = parser.parse_args()

    results = evaluate(args.model, args.data, args.device)
