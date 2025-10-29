import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from src.data.parquet_stock_dataset import (
    OnlineRobustScaler,
    ParquetStockIterableDataset,
)


def _write_parquet(path: Path, rows: list[dict]) -> None:
    df = pl.DataFrame(rows)
    df.write_parquet(path)


@pytest.mark.unit
def test_iterable_dataset_stitches_windows_across_shards(tmp_path: Path) -> None:
    """
    Ensure ParquetStockIterableDataset keeps per-code tails across shard boundaries.

    Without the buffers carry-over the dataset would drop windows that span files,
    reducing the effective training samples for long sequence lengths.
    """

    code = "JPX1337"
    feature_col = "feature_0"
    target_col = "target_1d"
    sequence_length = 4

    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(6)]
    all_rows = [
        {
            "Code": code,
            "Date": dates[i],
            feature_col: float(i),
            target_col: float(i),
        }
        for i in range(len(dates))
    ]

    shard_dir = Path(tmp_path)
    shard_paths = [
        shard_dir / "shard0.parquet",
        shard_dir / "shard1.parquet",
    ]

    # First shard has enough rows to form at least one window; second shard
    # provides additional rows that should stitch with the tail from shard0.
    _write_parquet(shard_paths[0], all_rows[:4])
    _write_parquet(shard_paths[1], all_rows[4:])

    dataset = ParquetStockIterableDataset(
        file_paths=shard_paths,
        feature_columns=[feature_col],
        target_columns=[target_col],
        code_column="Code",
        date_column="Date",
        sequence_length=sequence_length,
        scaler=OnlineRobustScaler(max_samples=128),
    )

    # Fit scaler (should see stitched windows during fitting as well).
    dataset.fit()

    samples = list(iter(dataset))

    # Total windows should include those spanning the cross-file boundary.
    # For 6 sequential rows with window size 4, expect 3 samples.
    assert len(samples) == 3

    # Window end-dates confirm that windows ending on dates from the second shard
    # were created, implying successful tail carry-over.
    assert [sample["date"] for sample in samples] == [
        "2024-01-04",
        "2024-01-05",
        "2024-01-06",
    ]

    # Sanity-check basic shapes/keys.
    for sample in samples:
        assert sample["features"].shape == (sequence_length, 1)
        assert sample["code"] == code
        assert "horizon_1" in sample["targets"]


@pytest.mark.unit
def test_iterable_dataset_worker_sharding(tmp_path: Path) -> None:
    """
    Verify ParquetStockIterableDataset correctly distributes shards across DataLoader workers.

    With num_workers=2, each worker should get a disjoint subset of file shards,
    and the total samples across all workers should match the expected count.
    """
    import torch.utils.data

    code = "JPX1337"
    feature_col = "feature_0"
    target_col = "target_1d"
    sequence_length = 4

    # Create 4 shards with enough data to form windows
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(20)]
    all_rows = [
        {
            "Code": code,
            "Date": dates[i],
            feature_col: float(i),
            target_col: float(i),
        }
        for i in range(len(dates))
    ]

    shard_dir = Path(tmp_path)
    shard_paths = [
        shard_dir / f"shard{i}.parquet" for i in range(4)
    ]

    # Distribute rows across 4 shards (5 rows each)
    for i, shard_path in enumerate(shard_paths):
        _write_parquet(shard_path, all_rows[i * 5 : (i + 1) * 5])

    dataset = ParquetStockIterableDataset(
        file_paths=shard_paths,
        feature_columns=[feature_col],
        target_columns=[target_col],
        code_column="Code",
        date_column="Date",
        sequence_length=sequence_length,
        scaler=OnlineRobustScaler(max_samples=128),
    )

    # Fit scaler first
    dataset.fit()

    # Create DataLoader with num_workers=2
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=2,
        persistent_workers=False,
    )

    samples = list(dataloader)

    # With 20 rows and sequence_length=4, expect 17 windows
    # (20 - 4 + 1 = 17)
    assert len(samples) == 17

    # Verify all samples have correct shape
    for batch in samples:
        assert batch["features"].shape == (1, sequence_length, 1)
        assert batch["code"][0] == code


@pytest.mark.unit
def test_iterable_dataset_handles_nan_windows(tmp_path: Path) -> None:
    """
    Verify ParquetStockIterableDataset correctly handles NaN values in features and targets.

    The scaler's partial_fit and transform should handle NaN gracefully,
    and targets should never be NaN after processing.
    """
    code = "JPX1337"
    feature_col = "feature_0"
    target_col = "target_1d"
    sequence_length = 4

    # Create data with some NaN values in features
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(10)]
    all_rows = [
        {
            "Code": code,
            "Date": dates[i],
            feature_col: float(i) if i % 3 != 0 else None,  # NaN every 3rd row
            target_col: float(i),
        }
        for i in range(len(dates))
    ]

    shard_path = tmp_path / "shard.parquet"
    _write_parquet(shard_path, all_rows)

    dataset = ParquetStockIterableDataset(
        file_paths=[shard_path],
        feature_columns=[feature_col],
        target_columns=[target_col],
        code_column="Code",
        date_column="Date",
        sequence_length=sequence_length,
        scaler=OnlineRobustScaler(max_samples=128),
    )

    # Fit scaler (should handle NaN values)
    dataset.fit()

    samples = list(iter(dataset))

    # Should still generate windows (7 windows from 10 rows with seq_len=4)
    assert len(samples) >= 1

    # Verify targets are never NaN
    import numpy as np

    for sample in samples:
        for horizon_key, horizon_val in sample["targets"].items():
            assert not np.isnan(horizon_val).any(), f"NaN found in {horizon_key}"
