#!/usr/bin/env python3
"""
Unified Feature Converter for gogooku3 → ATFT-GAT-FAN
Converts ML dataset to ATFT-compatible format with proper time-series structure
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class UnifiedFeatureConverter:
    """統合特徴量変換システム：gogooku3 → ATFT-GAT-FAN"""

    def __init__(
        self,
        sequence_length: int = 60,
        prediction_horizons: List[int] = None,
        min_samples_per_code: int = 100,
    ):
        """
        Initialize converter

        Args:
            sequence_length: Length of time series sequences (default: 60)
            prediction_horizons: Prediction horizons in days (default: [1, 5, 10, 20])
            min_samples_per_code: Minimum samples required per stock code
        """
        self.sequence_length = sequence_length
        self.prediction_horizons = prediction_horizons or [1, 5, 10, 20]
        self.min_samples_per_code = min_samples_per_code

        # Required columns
        self.required_columns = ["Code", "Date", "Close", "Volume"]

        # Feature columns to preserve (will be auto-detected)
        self.feature_columns = []

        # Target column mapping
        self.target_columns = {
            "feat_ret_1d": "target_1d",
            "returns_1d": "target_1d",
            "feat_ret_5d": "target_5d",
            "returns_5d": "target_5d",
            "feat_ret_10d": "target_10d",
            "returns_10d": "target_10d",
            "feat_ret_20d": "target_20d",
            "returns_20d": "target_20d",
        }

    def _validate_dataset(self, df: pl.DataFrame) -> None:
        """Validate input dataset"""
        missing = set(self.required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Check for at least one target column
        target_cols_present = [col for col in self.target_columns.keys() if col in df.columns]
        if not target_cols_present:
            raise ValueError(f"No target columns found. Need at least one of: {list(self.target_columns.keys())}")

        logger.info(f"Dataset validated: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.info(f"Target columns found: {target_cols_present}")

    def _detect_feature_columns(self, df: pl.DataFrame) -> List[str]:
        """Auto-detect feature columns (numeric columns excluding metadata)"""
        exclude_cols = {"Code", "Date", "code", "date", "index", "split_fold"}
        numeric_cols = [
            col for col in df.columns
            if col not in exclude_cols and df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
        ]

        logger.info(f"Detected {len(numeric_cols)} feature columns")
        return numeric_cols

    def _create_sequences(
        self,
        df: pl.DataFrame,
        code: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Create time-series sequences for a single stock

        Returns:
            features: Array of shape (n_samples, sequence_length, n_features)
            targets: Array of shape (n_samples, n_horizons)
            dates: List of dates for each sequence
        """
        code_df = df.filter(pl.col("Code") == code).sort("Date")

        if len(code_df) < self.sequence_length + max(self.prediction_horizons):
            return np.array([]), np.array([]), []

        # Get feature values
        feature_values = code_df.select(self.feature_columns).to_numpy()

        # Get target values (use first available target column)
        target_col = None
        for col in self.target_columns.keys():
            if col in code_df.columns:
                target_col = col
                break

        if target_col is None:
            return np.array([]), np.array([]), []

        target_values = code_df[target_col].to_numpy()
        dates = code_df["Date"].to_list()

        # Create sequences
        sequences = []
        targets = []
        seq_dates = []

        for i in range(len(feature_values) - self.sequence_length - max(self.prediction_horizons) + 1):
            # Feature sequence
            seq = feature_values[i:i + self.sequence_length]
            sequences.append(seq)

            # Multi-horizon targets
            horizon_targets = []
            for h in self.prediction_horizons:
                target_idx = i + self.sequence_length + h - 1
                if target_idx < len(target_values):
                    horizon_targets.append(target_values[target_idx])
                else:
                    horizon_targets.append(np.nan)
            targets.append(horizon_targets)

            # Date of the last point in the sequence
            seq_dates.append(dates[i + self.sequence_length - 1])

        return np.array(sequences), np.array(targets), seq_dates

    def convert_to_atft_format(
        self,
        df: pl.DataFrame,
        output_dir: str = "output/atft_data",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Dict[str, Union[List[str], Dict]]:
        """
        Convert ML dataset to ATFT format and save as parquet files

        Args:
            df: Input DataFrame from ML dataset
            output_dir: Output directory for converted files
            train_ratio: Training data ratio
            val_ratio: Validation data ratio
            test_ratio: Test data ratio

        Returns:
            Dictionary with file paths and metadata
        """
        # Validate
        self._validate_dataset(df)

        # Detect features
        self.feature_columns = self._detect_feature_columns(df)
        logger.info(f"Using {len(self.feature_columns)} feature columns")

        # Sort by date
        df = df.sort(["Code", "Date"])

        # Get unique codes
        codes = df["Code"].unique().to_list()
        logger.info(f"Processing {len(codes)} unique stocks")

        # Create output directories
        output_path = Path(output_dir)
        train_dir = output_path / "train"
        val_dir = output_path / "val"
        test_dir = output_path / "test"

        for dir_path in [train_dir, val_dir, test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Process each stock and collect sequences
        all_sequences = []
        all_targets = []
        all_codes = []
        all_dates = []

        for code in tqdm(codes, desc="Creating sequences"):
            sequences, targets, dates = self._create_sequences(df, code)

            if len(sequences) > 0:
                all_sequences.append(sequences)
                all_targets.append(targets)
                all_codes.extend([code] * len(sequences))
                all_dates.extend(dates)

        if not all_sequences:
            raise ValueError("No valid sequences created from dataset")

        # Concatenate all sequences
        all_sequences = np.vstack(all_sequences)
        all_targets = np.vstack(all_targets)

        logger.info(f"Created {len(all_sequences)} sequences of shape {all_sequences[0].shape}")

        # Split data chronologically
        n_total = len(all_sequences)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        # Create DataFrames for each split
        def create_split_df(sequences, targets, codes, dates, indices):
            """Helper to create DataFrame for a data split"""
            split_data = {
                "code": [codes[i] for i in indices],
                "date": [dates[i] for i in indices],
            }

            # Add feature columns (last value of each sequence)
            for i, feat_name in enumerate(self.feature_columns):
                split_data[feat_name] = sequences[indices, -1, i]

            # Add targets
            for i, horizon in enumerate(self.prediction_horizons):
                split_data[f"target_{horizon}d"] = targets[indices, i]

            return pl.DataFrame(split_data)

        # Create indices for splits
        indices = np.arange(n_total)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

        # Create and save DataFrames
        train_df = create_split_df(all_sequences, all_targets, all_codes, all_dates, train_indices)
        val_df = create_split_df(all_sequences, all_targets, all_codes, all_dates, val_indices)
        test_df = create_split_df(all_sequences, all_targets, all_codes, all_dates, test_indices)

        # Save as parquet files (chunked by date for efficiency)
        train_files = self._save_chunked_parquet(train_df, train_dir, "train")
        val_files = self._save_chunked_parquet(val_df, val_dir, "val")
        test_files = self._save_chunked_parquet(test_df, test_dir, "test")

        # Create metadata
        metadata = {
            "sequence_length": self.sequence_length,
            "prediction_horizons": self.prediction_horizons,
            "n_features": len(self.feature_columns),
            "feature_columns": self.feature_columns,
            "n_codes": len(codes),
            "n_sequences": n_total,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_total - n_train - n_val,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
        }

        logger.info(f"✅ Conversion complete")
        logger.info(f"  Train: {n_train} sequences → {len(train_files)} files")
        logger.info(f"  Val: {n_val} sequences → {len(val_files)} files")
        logger.info(f"  Test: {n_total - n_train - n_val} sequences → {len(test_files)} files")

        return {
            "train_files": train_files,
            "val_files": val_files,
            "test_files": test_files,
            "metadata": metadata,
            "output_dir": str(output_path),
        }

    def _save_chunked_parquet(
        self,
        df: pl.DataFrame,
        output_dir: Path,
        prefix: str,
        chunk_size: int = 100000
    ) -> List[str]:
        """Save DataFrame as chunked parquet files"""
        files = []
        n_chunks = (len(df) + chunk_size - 1) // chunk_size

        for i in range(n_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(df))
            chunk = df[start:end]

            file_path = output_dir / f"{prefix}_chunk_{i:04d}.parquet"
            chunk.write_parquet(file_path)
            files.append(str(file_path))

        return files

    def convert_from_file(
        self,
        input_path: str,
        output_dir: str = "output/atft_data",
        **kwargs
    ) -> Dict[str, Union[List[str], Dict]]:
        """
        Convenience method to convert from a parquet file

        Args:
            input_path: Path to input ML dataset parquet
            output_dir: Output directory for converted files
            **kwargs: Additional arguments for convert_to_atft_format

        Returns:
            Dictionary with file paths and metadata
        """
        logger.info(f"Loading dataset from {input_path}")
        df = pl.read_parquet(input_path)

        return self.convert_to_atft_format(df, output_dir, **kwargs)


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Convert ML dataset to ATFT format")
    parser.add_argument(
        "--input",
        type=str,
        default="output/ml_dataset_latest_full.parquet",
        help="Input ML dataset parquet file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/atft_data",
        help="Output directory for ATFT format files",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=60,
        help="Sequence length for time series",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Sample size for testing (optional)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load dataset
    df = pl.read_parquet(args.input)

    # Sample if requested
    if args.sample_size:
        df = df.sample(args.sample_size)
        logger.info(f"Sampled {args.sample_size} rows for testing")

    # Convert
    converter = UnifiedFeatureConverter(sequence_length=args.sequence_length)
    result = converter.convert_to_atft_format(df, args.output)

    logger.info("Conversion completed successfully")
    logger.info(f"Output directory: {result['output_dir']}")
    logger.info(f"Metadata: {result['metadata']}")


if __name__ == "__main__":
    main()