"""
Capture a representative batch from the dataloader for GAT gradient debugging.
"""
import logging
import pickle
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def capture_batch():
    """Capture first batch from training dataloader."""
    import sys

    sys.path.insert(0, "/workspace/gogooku3")

    from hydra import compose, initialize_config_dir

    from src.gogooku3.training.atft.data_module import ATFTDataModule

    # Initialize Hydra config
    config_dir = Path("/workspace/gogooku3/configs/atft").absolute()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config_production_optimized")

    # Override data path
    cfg.data.source.train_files = "output/atft_data/train/*.parquet"
    cfg.data.source.val_files = "output/atft_data/val/*.parquet"

    logger.info("Setting up data module...")
    data_module = ATFTDataModule(cfg)
    data_module.setup(stage="fit")

    logger.info("Creating train dataloader...")
    train_loader = data_module.train_dataloader()

    logger.info("Fetching first batch...")
    batch = next(iter(train_loader))

    # Save batch
    output_path = Path("_logs/training/captured_batch.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(batch, f)

    logger.info(f"✅ Batch saved to: {output_path}")
    logger.info(f"   Batch keys: {list(batch.keys())}")
    logger.info(f"   Features shape: {batch['features'].shape}")
    if "edge_index" in batch:
        logger.info(f"   Edge index shape: {batch['edge_index'].shape}")
    if "edge_attr" in batch:
        logger.info(f"   Edge attr shape: {batch['edge_attr'].shape}")

    return batch


if __name__ == "__main__":
    capture_batch()
