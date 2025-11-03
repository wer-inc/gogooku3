"""Placeholder training entrypoint for ATFT-GAT-FAN.

This will be replaced with the migrated SafeTrainingPipeline integration.
"""
from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
_logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ATFT-GAT-FAN model")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the training configuration file.",
    )
    parser.add_argument(
        "--dataset",
        default="../../data/output/ml_dataset_latest.parquet",
        help="Path to the pre-built dataset parquet file.",
    )
    return parser.parse_args()


def main() -> int:
    """Stub implementation that logs a warning instead of raising SystemExit.

    Phase 2 Bug #31 fix: Changed from NoReturn -> int to avoid import-time failure.
    """
    args = parse_args()
    _logger.warning(
        "ATFT-GAT-FAN training pipeline is not yet implemented. "
        f"Received config={args.config}, dataset={args.dataset}. "
        "Skipping training."
    )


if __name__ == "__main__":
    main()
