"""Simple RankIC computation from model checkpoint."""
import sys

sys.path.insert(0, "/workspace/gogooku3")

import logging

import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_rankic(predictions, targets):
    """Compute Rank IC (Spearman correlation)."""
    # Flatten and remove NaNs
    pred_flat = predictions.flatten()
    targ_flat = targets.flatten()

    mask = ~(np.isnan(pred_flat) | np.isnan(targ_flat))
    if mask.sum() < 10:
        return np.nan, 0

    pred_clean = pred_flat[mask]
    targ_clean = targ_flat[mask]

    if len(np.unique(pred_clean)) < 2 or len(np.unique(targ_clean)) < 2:
        return np.nan, mask.sum()

    corr, pval = spearmanr(pred_clean, targ_clean)
    return corr, mask.sum()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    logger.info("Loading model from models/checkpoints/atft_gat_fan_final.pt")
    checkpoint = torch.load(
        "models/checkpoints/atft_gat_fan_final.pt",
        map_location=device,
        weights_only=False,
    )

    # Get config
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(checkpoint["config"])

    # Initialize model
    from src.atft_gat_fan.models.architectures.atft_gat_fan import ATFT_GAT_FAN

    model = ATFT_GAT_FAN(cfg)

    # Load weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        raise KeyError("No model weights found in checkpoint")

    model.to(device)
    model.eval()
    logger.info(
        f"Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)"
    )

    # Setup data
    from src.gogooku3.training.atft.data_module import ProductionDataModuleV2

    data_module = ProductionDataModuleV2(
        data_dir="output/atft_data",
        batch_size=1,  # Single batch for simplicity
        num_workers=0,
        config=cfg.data,
    )
    data_module.setup("fit")

    val_loader = data_module.val_dataloader()
    logger.info("Dataloader ready")

    # Collect predictions and targets
    all_preds = {f"horizon_{h}": [] for h in [1, 5, 10, 20]}
    all_targs = {f"horizon_{h}": [] for h in [1, 5, 10, 20]}

    logger.info("Running inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
            if batch_idx > 0:  # Only process first batch for quick test
                break

            features = batch["features"].to(device)
            targets = batch.get("targets", {})

            # Forward pass
            outputs = model(features)

            # Extract predictions (assuming shape: (B, T, 1) or (B, T))
            if isinstance(outputs, dict) and "predictions" in outputs:
                predictions = outputs["predictions"]
            else:
                predictions = outputs

            # Collect by horizon
            for h in [1, 5, 10, 20]:
                pred_key = f"horizon_{h}d"
                targ_key = f"horizon_{h}"

                if pred_key in predictions and targ_key in targets:
                    # Predictions: (B, T, 1) or (B, T) → take last timestep
                    pred = predictions[pred_key].detach().cpu().numpy()
                    if pred.ndim == 3:  # (B, T, 1)
                        pred = pred[:, -1, 0]  # Take last timestep, remove feature dim
                    elif pred.ndim == 2:  # (B, T)
                        pred = pred[:, -1]  # Take last timestep

                    # Targets: (B,)
                    targ = targets[targ_key].detach().cpu().numpy()

                    all_preds[f"horizon_{h}"].append(pred)
                    all_targs[f"horizon_{h}"].append(targ)

    # Compute RankIC for each horizon
    logger.info("\n" + "=" * 70)
    logger.info("RANK IC RESULTS")
    logger.info("=" * 70)

    results = {}
    for h in [1, 5, 10, 20]:
        key = f"horizon_{h}"

        if all_preds[key] and all_targs[key]:
            preds = np.concatenate(all_preds[key])
            targs = np.concatenate(all_targs[key])

            rankic, n_valid = compute_rankic(preds, targs)

            results[key] = {
                "rank_ic": rankic,
                "n_valid": n_valid,
                "n_total": len(preds),
            }

            logger.info(f"\n{h}d Horizon:")
            logger.info(
                f"  Rank IC: {rankic:.4f}" if not np.isnan(rankic) else "  Rank IC: NaN"
            )
            logger.info(f"  Valid Samples: {n_valid:,} / {len(preds):,}")
        else:
            logger.info(f"\n{h}d Horizon: No data collected")
            results[key] = {"rank_ic": np.nan, "n_valid": 0, "n_total": 0}

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    valid_ics = [v["rank_ic"] for v in results.values() if not np.isnan(v["rank_ic"])]
    if valid_ics:
        avg_ic = np.mean(valid_ics)
        logger.info(f"Average Rank IC: {avg_ic:.4f}")
        logger.info("Phase 2 Target: ≥0.020")

        if avg_ic >= 0.020:
            logger.info("✅ TARGET MET")
        else:
            logger.info(f"⚠️  Below target by {0.020 - avg_ic:.4f}")
    else:
        logger.info("No valid Rank IC values computed")

    return results


if __name__ == "__main__":
    results = main()
