#!/usr/bin/env python3
"""Split ML dataset into train/val/test splits for ATFT training."""

import argparse
import logging
from pathlib import Path

import polars as pl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_dataset(input_file: str, output_dir: str, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15):
    """Split dataset by time into train/val/test.

    Args:
        input_file: Path to input parquet file
        output_dir: Output directory for split files
        train_ratio: Fraction for training (default 0.7)
        val_ratio: Fraction for validation (default 0.15)
        test_ratio: Fraction for testing (default 0.15)
    """
    logger.info(f"Loading dataset from {input_file}...")
    df = pl.read_parquet(input_file)

    # Sort by date to ensure temporal order
    df = df.sort("Date")

    total_rows = len(df)
    train_end = int(total_rows * train_ratio)
    val_end = int(total_rows * (train_ratio + val_ratio))

    logger.info(f"Total samples: {total_rows:,}")
    logger.info(f"Train: {train_end:,} ({train_ratio:.1%})")
    logger.info(f"Val: {val_end - train_end:,} ({val_ratio:.1%})")
    logger.info(f"Test: {total_rows - val_end:,} ({test_ratio:.1%})")

    # Split data
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    # Create output directories
    output_path = Path(output_dir)
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    test_dir = output_path / "test"

    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Write split files
    logger.info("Writing train split...")
    train_df.write_parquet(train_dir / "data.parquet")

    logger.info("Writing val split...")
    val_df.write_parquet(val_dir / "data.parquet")

    logger.info("Writing test split...")
    test_df.write_parquet(test_dir / "data.parquet")

    logger.info("✅ Dataset split complete:")
    logger.info(f"   Train: {train_dir}/data.parquet")
    logger.info(f"   Val: {val_dir}/data.parquet")
    logger.info(f"   Test: {test_dir}/data.parquet")


def main():
    parser = argparse.ArgumentParser(description="Split ML dataset into train/val/test")
    parser.add_argument("--input-file", required=True, help="Input parquet file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio")

    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    split_dataset(
        input_file=args.input_file,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )


if __name__ == "__main__":
    main()
