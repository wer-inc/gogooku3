"""Placeholder training entrypoint for APEX-Ranker."""
from __future__ import annotations

import argparse
from typing import NoReturn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the APEX-Ranker model")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the training configuration file.",
    )
    parser.add_argument(
        "--dataset",
        default="../../data/output/ml_dataset_latest.parquet",
        help="Path to the dataset parquet file.",
    )
    return parser.parse_args()


def main() -> NoReturn:
    args = parse_args()
    raise SystemExit(
        "APEX-Ranker training pipeline not implemented yet. "
        f"Received config={args.config}, dataset={args.dataset}."
    )


if __name__ == "__main__":
    main()
