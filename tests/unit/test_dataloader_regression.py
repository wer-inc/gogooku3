"""
Regression test for DataLoader initialization to prevent hanging issues.

This test ensures that DataLoaders can be created without hanging,
addressing the issue where missing imports caused the training process to hang.
"""

import pytest
import torch
from pathlib import Path
import tempfile
import polars as pl
import numpy as np
from omegaconf import OmegaConf
import signal
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gogooku3.training.atft.data_module import ProductionDataModuleV2


def timeout_handler(signum, frame):
    """Handler for timeout signal."""
    raise TimeoutError("DataLoader creation timed out - possible hanging issue")


@pytest.mark.critical
class TestDataLoaderRegression:
    """Test suite to prevent DataLoader hanging issues."""

    @pytest.fixture
    def minimal_config(self):
        """Create minimal configuration for testing."""
        config = {
            "data": {
                "source": {
                    "data_dir": "test_data",
                    "file_pattern": "*.parquet"
                },
                "time_series": {
                    "sequence_length": 60,
                    "prediction_horizons": [1, 5, 10, 20]
                },
                "schema": {
                    "date_column": "date",
                    "code_column": "code",
                    "target_column": "target",
                    "feature_columns": None  # Auto-detect
                },
                "use_day_batch_sampler": False,
                "split": {
                    "train_ratio": 0.6,
                    "val_ratio": 0.2,
                    "test_ratio": 0.2
                }
            },
            "normalization": {
                "online_normalization": {
                    "enabled": False
                }
            },
            "train": {
                "batch": {
                    "train_batch_size": 2,
                    "val_batch_size": 2,
                    "test_batch_size": 2,
                    "num_workers": 0,
                    "pin_memory": False,
                    "prefetch_factor": None,
                    "persistent_workers": False
                }
            }
        }
        return OmegaConf.create(config)

    @pytest.fixture
    def sample_data_dir(self, tmp_path):
        """Create sample parquet files for testing."""
        data_dir = tmp_path / "test_data"
        data_dir.mkdir(exist_ok=True)

        # Create sample data
        dates = pl.date_range(
            start=pl.date(2024, 1, 1),
            end=pl.date(2024, 3, 31),
            interval="1d",
            eager=True
        )

        # Create train/val/test directories
        for split in ["train", "val", "test"]:
            split_dir = data_dir / split
            split_dir.mkdir(exist_ok=True)

            # Create a few small parquet files
            for i in range(2):
                df_data = {
                    "date": dates.to_list() * 2,  # 2 stocks
                    "code": [f"STOCK{j}" for _ in range(len(dates)) for j in range(2)],
                    "target_1d": np.random.randn(len(dates) * 2).tolist(),
                    "target_5d": np.random.randn(len(dates) * 2).tolist(),
                    "target_10d": np.random.randn(len(dates) * 2).tolist(),
                    "target_20d": np.random.randn(len(dates) * 2).tolist(),
                }

                # Add some feature columns
                for j in range(10):
                    df_data[f"feature_{j}"] = np.random.randn(len(dates) * 2).tolist()

                df = pl.DataFrame(df_data)
                df.write_parquet(split_dir / f"data_{i}.parquet")

        return data_dir

    def test_dataloader_creation_no_hang(self, minimal_config, sample_data_dir):
        """Test that DataLoader creation completes within timeout."""
        # Update config with test data directory
        minimal_config.data.source.data_dir = str(sample_data_dir)

        # Set timeout for dataloader creation (5 seconds should be plenty)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)

        try:
            # Create data module
            data_module = ProductionDataModuleV2(minimal_config)

            # Setup should complete without hanging
            data_module.setup()

            # Create train dataloader - this should not hang
            train_loader = data_module.train_dataloader()
            assert train_loader is not None

            # Create val dataloader - this should not hang
            val_loader = data_module.val_dataloader()
            assert val_loader is not None

            # Cancel the alarm
            signal.alarm(0)

        except TimeoutError as e:
            pytest.fail(f"DataLoader creation timed out: {e}")
        finally:
            # Always cancel the alarm
            signal.alarm(0)

    def test_dataloader_iteration(self, minimal_config, sample_data_dir):
        """Test that we can iterate through the DataLoader."""
        minimal_config.data.source.data_dir = str(sample_data_dir)

        # Create data module
        data_module = ProductionDataModuleV2(minimal_config)
        data_module.setup()

        # Get train dataloader
        train_loader = data_module.train_dataloader()

        # Set timeout for iteration
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)

        try:
            # Test that we can get at least one batch
            batch_count = 0
            for batch in train_loader:
                assert "features" in batch
                assert "targets" in batch
                batch_count += 1
                if batch_count >= 2:  # Test first 2 batches
                    break

            assert batch_count > 0, "No batches were retrieved from DataLoader"

            # Cancel the alarm
            signal.alarm(0)

        except TimeoutError as e:
            pytest.fail(f"DataLoader iteration timed out: {e}")
        finally:
            signal.alarm(0)

    def test_dataloader_with_empty_dataset(self, minimal_config, tmp_path):
        """Test that DataLoader handles empty datasets gracefully."""
        # Create empty data directory
        empty_dir = tmp_path / "empty_data"
        empty_dir.mkdir()
        for split in ["train", "val", "test"]:
            (empty_dir / split).mkdir()

        minimal_config.data.source.data_dir = str(empty_dir)

        # This should not hang even with no data
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)

        try:
            data_module = ProductionDataModuleV2(minimal_config)

            # Setup might fail or create empty datasets, but should not hang
            try:
                data_module.setup()
            except Exception as e:
                # It's OK if setup fails with no data, as long as it doesn't hang
                assert "No" in str(e) or "empty" in str(e).lower() or "found 0" in str(e).lower()

            signal.alarm(0)

        except TimeoutError as e:
            pytest.fail(f"DataLoader creation with empty dataset timed out: {e}")
        finally:
            signal.alarm(0)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])