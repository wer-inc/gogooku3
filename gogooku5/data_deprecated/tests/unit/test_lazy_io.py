"""Unit tests for lazy_io.py - Polars lazy scan and Arrow IPC helpers."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest
from builder.utils.lazy_io import (
    _apply_pushdown,
    get_format_info,
    lazy_load,
    save_with_cache,
)
from polars.testing import assert_frame_equal


@pytest.fixture
def sample_df() -> pl.DataFrame:
    """Create a sample DataFrame for testing."""
    return pl.DataFrame(
        {
            "date": [date(2024, 1, i) for i in range(1, 11)],
            "code": [f"A{i % 3}" for i in range(10)],
            "close": [100.0 + i for i in range(10)],
            "volume": [1000 + i * 100 for i in range(10)],
            "ret_1d": [0.01 + i * 0.001 for i in range(10)],
        }
    )


def test_save_with_cache_parquet_only(tmp_path: Path, sample_df: pl.DataFrame) -> None:
    """Test saving DataFrame as Parquet only (no IPC)."""
    output_path = tmp_path / "test.parquet"

    parquet_path, ipc_path = save_with_cache(
        sample_df,
        output_path,
        create_ipc=False,
    )

    # Check Parquet created
    assert parquet_path.exists()
    assert parquet_path == output_path

    # Check IPC not created
    assert ipc_path is None
    assert not (tmp_path / "test.arrow").exists()

    # Verify content
    loaded = pl.read_parquet(parquet_path)
    assert_frame_equal(loaded, sample_df)


def test_save_with_cache_dual_format(tmp_path: Path, sample_df: pl.DataFrame) -> None:
    """Test saving DataFrame in dual format (Parquet + IPC)."""
    output_path = tmp_path / "test.parquet"

    parquet_path, ipc_path = save_with_cache(
        sample_df,
        output_path,
        create_ipc=True,
    )

    # Check both formats created
    assert parquet_path.exists()
    assert ipc_path is not None
    assert ipc_path.exists()
    assert ipc_path == tmp_path / "test.arrow"

    # Verify both have same content
    parquet_df = pl.read_parquet(parquet_path)
    ipc_df = pl.read_ipc(ipc_path)

    assert_frame_equal(parquet_df, sample_df)
    assert_frame_equal(ipc_df, sample_df)


def test_lazy_load_parquet(tmp_path: Path, sample_df: pl.DataFrame) -> None:
    """Test lazy loading from Parquet file."""
    parquet_path = tmp_path / "test.parquet"
    sample_df.write_parquet(parquet_path)

    loaded = lazy_load(parquet_path, prefer_ipc=False)

    assert_frame_equal(loaded, sample_df)


def test_lazy_load_ipc_preferred(tmp_path: Path, sample_df: pl.DataFrame) -> None:
    """Test lazy loading with IPC preference (when IPC exists)."""
    parquet_path = tmp_path / "test.parquet"
    ipc_path = tmp_path / "test.arrow"

    # Save both formats
    sample_df.write_parquet(parquet_path)
    sample_df.write_ipc(ipc_path, compression="lz4")

    # Should load from IPC (faster)
    loaded = lazy_load(parquet_path, prefer_ipc=True)

    assert_frame_equal(loaded, sample_df)


def test_lazy_load_ipc_fallback(tmp_path: Path, sample_df: pl.DataFrame) -> None:
    """Test lazy loading falls back to Parquet when IPC missing."""
    parquet_path = tmp_path / "test.parquet"
    sample_df.write_parquet(parquet_path)

    # IPC file doesn't exist, should fallback to Parquet
    loaded = lazy_load(parquet_path, prefer_ipc=True)

    assert_frame_equal(loaded, sample_df)


def test_lazy_load_with_filters(tmp_path: Path, sample_df: pl.DataFrame) -> None:
    """Test lazy loading with predicate pushdown (date filter)."""
    parquet_path = tmp_path / "test.parquet"
    sample_df.write_parquet(parquet_path)

    # Filter to dates >= 2024-01-05
    loaded = lazy_load(
        parquet_path,
        filters=pl.col("date") >= date(2024, 1, 5),
        prefer_ipc=False,
    )

    expected = sample_df.filter(pl.col("date") >= date(2024, 1, 5))
    assert_frame_equal(loaded, expected)


def test_lazy_load_with_column_pruning(tmp_path: Path, sample_df: pl.DataFrame) -> None:
    """Test lazy loading with column selection."""
    parquet_path = tmp_path / "test.parquet"
    sample_df.write_parquet(parquet_path)

    # Select only date and close columns
    loaded = lazy_load(
        parquet_path,
        columns=["date", "close"],
        prefer_ipc=False,
    )

    expected = sample_df.select(["date", "close"])
    assert_frame_equal(loaded, expected)


def test_lazy_load_combined_optimization(tmp_path: Path, sample_df: pl.DataFrame) -> None:
    """Test lazy loading with both filters and column pruning."""
    parquet_path = tmp_path / "test.parquet"
    sample_df.write_parquet(parquet_path)

    loaded = lazy_load(
        parquet_path,
        filters=pl.col("close") > 105.0,
        columns=["date", "code", "close"],
        prefer_ipc=False,
    )

    expected = sample_df.filter(pl.col("close") > 105.0).select(["date", "code", "close"])
    assert_frame_equal(loaded, expected)


def test_lazy_load_multi_file(tmp_path: Path, sample_df: pl.DataFrame) -> None:
    """Test lazy loading from multiple Parquet files."""
    # Split into two files
    df1 = sample_df.slice(0, 5)
    df2 = sample_df.slice(5, 5)

    path1 = tmp_path / "part1.parquet"
    path2 = tmp_path / "part2.parquet"

    df1.write_parquet(path1)
    df2.write_parquet(path2)

    # Load both files
    loaded = lazy_load([path1, path2], prefer_ipc=False)

    # Should concatenate both files
    expected = pl.concat([df1, df2])
    assert_frame_equal(loaded.sort("date"), expected.sort("date"))


def test_lazy_load_file_not_found(tmp_path: Path) -> None:
    """Test lazy loading raises FileNotFoundError when file missing."""
    missing_path = tmp_path / "nonexistent.parquet"

    with pytest.raises(FileNotFoundError, match="Dataset file"):
        lazy_load(missing_path, prefer_ipc=False)


def test_apply_pushdown_filters_only() -> None:
    """Test _apply_pushdown with filters only."""
    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [10, 20, 30, 40, 50],
        }
    )

    lf = pl.LazyFrame(df)
    result = _apply_pushdown(lf, filters=pl.col("x") > 2, columns=None).collect()

    expected = df.filter(pl.col("x") > 2)
    assert_frame_equal(result, expected)


def test_apply_pushdown_columns_only() -> None:
    """Test _apply_pushdown with column selection only."""
    df = pl.DataFrame(
        {
            "x": [1, 2, 3],
            "y": [10, 20, 30],
            "z": [100, 200, 300],
        }
    )

    lf = pl.LazyFrame(df)
    result = _apply_pushdown(lf, filters=None, columns=["x", "y"]).collect()

    expected = df.select(["x", "y"])
    assert_frame_equal(result, expected)


def test_apply_pushdown_both() -> None:
    """Test _apply_pushdown with filters and columns."""
    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [10, 20, 30, 40, 50],
            "z": [100, 200, 300, 400, 500],
        }
    )

    lf = pl.LazyFrame(df)
    result = _apply_pushdown(
        lf,
        filters=pl.col("x") >= 3,
        columns=["x", "y"],
    ).collect()

    expected = df.filter(pl.col("x") >= 3).select(["x", "y"])
    assert_frame_equal(result, expected)


def test_get_format_info_parquet_only(tmp_path: Path, sample_df: pl.DataFrame) -> None:
    """Test get_format_info when only Parquet exists."""
    parquet_path = tmp_path / "test.parquet"
    sample_df.write_parquet(parquet_path)

    info = get_format_info(parquet_path)

    assert info["parquet_exists"] is True
    assert info["parquet_size_mb"] > 0
    assert info["ipc_exists"] is False
    assert info["ipc_size_mb"] == 0.0
    assert info["speedup_estimate"] == "N/A"


def test_get_format_info_dual_format(tmp_path: Path, sample_df: pl.DataFrame) -> None:
    """Test get_format_info when both formats exist."""
    parquet_path = tmp_path / "test.parquet"
    ipc_path = tmp_path / "test.arrow"

    sample_df.write_parquet(parquet_path)
    sample_df.write_ipc(ipc_path, compression="lz4")

    info = get_format_info(parquet_path)

    assert info["parquet_exists"] is True
    assert info["parquet_size_mb"] > 0
    assert info["ipc_exists"] is True
    assert info["ipc_size_mb"] > 0
    assert info["speedup_estimate"] == "3-5x"


def test_get_format_info_neither_format(tmp_path: Path) -> None:
    """Test get_format_info when neither format exists."""
    nonexistent = tmp_path / "missing.parquet"

    info = get_format_info(nonexistent)

    assert info["parquet_exists"] is False
    assert info["parquet_size_mb"] == 0.0
    assert info["ipc_exists"] is False
    assert info["ipc_size_mb"] == 0.0
    assert info["speedup_estimate"] == "N/A"


def test_save_with_cache_compression(tmp_path: Path, sample_df: pl.DataFrame) -> None:
    """Test save_with_cache with custom Parquet compression."""
    output_path = tmp_path / "test.parquet"

    parquet_path, ipc_path = save_with_cache(
        sample_df,
        output_path,
        create_ipc=True,
        parquet_kwargs={"compression": "snappy"},
    )

    assert parquet_path.exists()
    assert ipc_path is not None
    assert ipc_path.exists()

    # Verify content matches
    loaded = pl.read_parquet(parquet_path)
    assert_frame_equal(loaded, sample_df)
