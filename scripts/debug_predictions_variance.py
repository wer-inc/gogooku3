"""Debug script to check prediction variance after quantile aggregation."""
import sys

sys.path.insert(0, "/workspace/gogooku3")

import logging

import numpy as np
import torch
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    logger.info("Loading model...")
    checkpoint = torch.load(
        "models/checkpoints/atft_gat_fan_final.pt",
        map_location=device,
        weights_only=False,
    )

    from omegaconf import OmegaConf

    cfg = OmegaConf.create(checkpoint["config"])

    from src.atft_gat_fan.models.architectures.atft_gat_fan import ATFT_GAT_FAN

    model = ATFT_GAT_FAN(cfg)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        raise KeyError("No model weights found")

    model.to(device)
    model.eval()
    logger.info(
        f"✅ Model loaded ({sum(p.numel() for p in model.parameters()):,} params)"
    )

    # Setup dataloader
    from src.gogooku3.training.atft.data_module import ProductionDataModuleV2

    data_module = ProductionDataModuleV2(
        data_dir="output/atft_data",
        batch_size=1,
        num_workers=0,
        config=cfg.data,
    )
    data_module.setup("fit")
    val_loader = data_module.val_dataloader()

    # Get first batch
    batch = next(iter(val_loader))
    features = batch["features"].to(device)
    targets = batch.get("targets", {})

    # Forward pass
    with torch.no_grad():
        outputs = model(features)

    logger.info("\n" + "=" * 70)
    logger.info("RAW QUANTILE PREDICTIONS (before aggregation)")
    logger.info("=" * 70)

    if isinstance(outputs, dict) and "predictions" in outputs:
        predictions = outputs["predictions"]
    else:
        predictions = outputs

    for h in [1, 5, 10, 20]:
        key = f"horizon_{h}d"
        if key in predictions:
            pred = predictions[key].detach().cpu().numpy()
            logger.info(f"\n{key}:")
            logger.info(f"  Shape: {pred.shape}")
            logger.info(f"  Mean: {pred.mean():.6f}")
            logger.info(f"  Std: {pred.std():.6f}")
            logger.info(f"  Min: {pred.min():.6f}")
            logger.info(f"  Max: {pred.max():.6f}")

            if pred.ndim == 2:
                logger.info("  Per-quantile stats:")
                for q_idx in range(pred.shape[1]):
                    q_std = pred[:, q_idx].std()
                    logger.info(f"    Quantile {q_idx}: std={q_std:.6f}")

    # Apply aggregation
    point_predictions = model.get_point_predictions(outputs, method="mean")

    logger.info("\n" + "=" * 70)
    logger.info("POINT PREDICTIONS (after mean aggregation)")
    logger.info("=" * 70)

    for h in [1, 5, 10, 20]:
        key = f"horizon_{h}d"
        if key in point_predictions:
            pred = point_predictions[key].detach().cpu().numpy()
            targ_key = f"horizon_{h}"
            targ = targets.get(targ_key, None)

            logger.info(f"\n{key}:")
            logger.info(f"  Shape: {pred.shape}")
            logger.info(f"  Mean: {pred.mean():.6f}")
            logger.info(f"  Std: {pred.std():.6f}")
            logger.info(f"  Min: {pred.min():.6f}")
            logger.info(f"  Max: {pred.max():.6f}")
            logger.info(f"  Unique values: {len(np.unique(pred))}")

            if targ is not None:
                targ_np = targ.detach().cpu().numpy()
                mask = np.isfinite(pred) & np.isfinite(targ_np)
                if mask.sum() > 10:
                    pred_clean = pred[mask]
                    targ_clean = targ_np[mask]

                    if len(np.unique(pred_clean)) >= 2:
                        corr, pval = spearmanr(pred_clean, targ_clean)
                        logger.info(f"  RankIC: {corr:.4f} (p={pval:.4f})")
                    else:
                        logger.info("  RankIC: NaN (not enough unique predictions)")
                else:
                    logger.info("  RankIC: NaN (not enough valid samples)")

    logger.info("\n" + "=" * 70)


if __name__ == "__main__":
    main()
