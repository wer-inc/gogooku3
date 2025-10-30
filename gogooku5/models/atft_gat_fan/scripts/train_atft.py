"""Placeholder training entrypoint for ATFT-GAT-FAN.

This will be replaced with the migrated SafeTrainingPipeline integration.
"""
from __future__ import annotations

import argparse
from typing import NoReturn


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


def main() -> NoReturn:
    args = parse_args()
    raise SystemExit(
        "ATFT-GAT-FAN training pipeline is not yet implemented. "
        f"Received config={args.config}, dataset={args.dataset}."
    )


if __name__ == "__main__":
    main()
