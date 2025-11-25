"""Unit tests for cache.py Arrow IPC support."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from builder.utils.cache import CacheManager
from polars.testing import assert_frame_equal


@pytest.fixture
def sample_df() -> pl.DataFrame:
    """Create a sample DataFrame for cache testing."""
    return pl.DataFrame(
        {
            "date": [date(2024, 1, i) for i in range(1, 6)],
            "code": [f"A{i % 2}" for i in range(5)],
            "close": [100.0 + i for i in range(5)],
            "volume": [1000 + i * 100 for i in range(5)],
        }
    )


@pytest.fixture
def cache_manager(tmp_path: Path) -> CacheManager:
    """Create a CacheManager instance with temporary directory."""
    from unittest.mock import Mock, PropertyMock

    from builder.config.settings import DatasetBuilderSettings

    settings = Mock(spec=DatasetBuilderSettings)
    settings.data_cache_dir = tmp_path
    settings.data_output_dir = tmp_path
    settings.latest_dataset_symlink = "ml_dataset_latest.parquet"
    settings.cache_ttl_days_default = 1

    # Mock default_cache_index_path property to return a real Path
    type(settings).default_cache_index_path = PropertyMock(return_value=tmp_path / "cache_index.json")

    manager = CacheManager(settings=settings)
    return manager


def test_cache_file_parquet_format(cache_manager: CacheManager) -> None:
    """Test cache_file returns correct path for Parquet format."""
    path = cache_manager.cache_file("test_key", format="parquet")

    assert path.name == "test_key.parquet"
    assert path.parent == cache_manager.cache_dir


def test_cache_file_ipc_format(cache_manager: CacheManager) -> None:
    """Test cache_file returns correct path for IPC format."""
    path = cache_manager.cache_file("test_key", format="ipc")

    assert path.name == "test_key.arrow"
    assert path.parent == cache_manager.cache_dir


def test_save_dataframe_ipc_format(cache_manager: CacheManager, sample_df: pl.DataFrame) -> None:
    """Test saving DataFrame in IPC format."""
    path = cache_manager.save_dataframe(
        "test_key",
        sample_df,
        format="ipc",
        dual_format=False,
    )

    # Check IPC file created
    assert path.exists()
    assert path.suffix == ".arrow"

    # Verify content
    loaded = pl.read_ipc(path)
    assert_frame_equal(loaded, sample_df)

    # Check Parquet not created (dual_format=False)
    parquet_path = cache_manager.cache_file("test_key", format="parquet")
    assert not parquet_path.exists()


def test_save_dataframe_parquet_format(cache_manager: CacheManager, sample_df: pl.DataFrame) -> None:
    """Test saving DataFrame in Parquet format."""
    path = cache_manager.save_dataframe(
        "test_key",
        sample_df,
        format="parquet",
        dual_format=False,
    )

    # Check Parquet file created
    assert path.exists()
    assert path.suffix == ".parquet"

    # Verify content
    loaded = pl.read_parquet(path)
    assert_frame_equal(loaded, sample_df)

    # Check IPC not created
    ipc_path = cache_manager.cache_file("test_key", format="ipc")
    assert not ipc_path.exists()


def test_save_dataframe_dual_format(cache_manager: CacheManager, sample_df: pl.DataFrame) -> None:
    """Test saving DataFrame in dual format (IPC + Parquet)."""
    path = cache_manager.save_dataframe(
        "test_key",
        sample_df,
        format="ipc",
        dual_format=True,
    )

    # Check both formats created
    ipc_path = cache_manager.cache_file("test_key", format="ipc")
    parquet_path = cache_manager.cache_file("test_key", format="parquet")

    assert ipc_path.exists()
    assert parquet_path.exists()

    # Verify both have same content
    ipc_df = pl.read_ipc(ipc_path)
    parquet_df = pl.read_parquet(parquet_path)

    assert_frame_equal(ipc_df, sample_df)
    assert_frame_equal(parquet_df, sample_df)


def test_save_dataframe_updates_index(cache_manager: CacheManager, sample_df: pl.DataFrame) -> None:
    """Test save_dataframe updates cache index with metadata."""
    cache_manager.save_dataframe(
        "test_key",
        sample_df,
        format="ipc",
        dual_format=True,
    )

    index = cache_manager.load_index()
    assert "test_key" in index

    entry = index["test_key"]
    assert entry["rows"] == len(sample_df)
    assert entry["format"] == "ipc"
    assert entry["dual_format"] is True
    assert "updated_at" in entry


def test_load_dataframe_ipc_preferred(cache_manager: CacheManager, sample_df: pl.DataFrame) -> None:
    """Test load_dataframe prefers IPC format when available."""
    # Save both formats
    ipc_path = cache_manager.cache_file("test_key", format="ipc")
    parquet_path = cache_manager.cache_file("test_key", format="parquet")

    sample_df.write_ipc(ipc_path, compression="lz4")
    sample_df.write_parquet(parquet_path)

    # Load with IPC preference
    loaded = cache_manager.load_dataframe("test_key", prefer_ipc=True)

    assert loaded is not None
    assert_frame_equal(loaded, sample_df)


def test_load_dataframe_ipc_fallback(cache_manager: CacheManager, sample_df: pl.DataFrame) -> None:
    """Test load_dataframe falls back to Parquet when IPC missing."""
    # Save only Parquet
    parquet_path = cache_manager.cache_file("test_key", format="parquet")
    sample_df.write_parquet(parquet_path)

    # Load with IPC preference (should fallback to Parquet)
    loaded = cache_manager.load_dataframe("test_key", prefer_ipc=True)

    assert loaded is not None
    assert_frame_equal(loaded, sample_df)


def test_load_dataframe_parquet_only(cache_manager: CacheManager, sample_df: pl.DataFrame) -> None:
    """Test load_dataframe with prefer_ipc=False loads Parquet."""
    # Save only Parquet
    parquet_path = cache_manager.cache_file("test_key", format="parquet")
    sample_df.write_parquet(parquet_path)

    loaded = cache_manager.load_dataframe("test_key", prefer_ipc=False)

    assert loaded is not None
    assert_frame_equal(loaded, sample_df)


def test_load_dataframe_not_found(cache_manager: CacheManager) -> None:
    """Test load_dataframe returns None when cache doesn't exist."""
    loaded = cache_manager.load_dataframe("nonexistent_key")

    assert loaded is None


def test_has_cache_any_format(cache_manager: CacheManager, sample_df: pl.DataFrame) -> None:
    """Test has_cache with any_format=True detects either format."""
    # Save only IPC
    ipc_path = cache_manager.cache_file("test_key", format="ipc")
    sample_df.write_ipc(ipc_path, compression="lz4")

    assert cache_manager.has_cache("test_key", any_format=True) is True


def test_has_cache_parquet_only_mode(cache_manager: CacheManager, sample_df: pl.DataFrame) -> None:
    """Test has_cache with any_format=False checks Parquet only."""
    # Save only IPC
    ipc_path = cache_manager.cache_file("test_key", format="ipc")
    sample_df.write_ipc(ipc_path, compression="lz4")

    # Should return False because Parquet doesn't exist
    assert cache_manager.has_cache("test_key", any_format=False) is False

    # Save Parquet
    parquet_path = cache_manager.cache_file("test_key", format="parquet")
    sample_df.write_parquet(parquet_path)

    # Should return True now
    assert cache_manager.has_cache("test_key", any_format=False) is True


def test_has_cache_both_formats(cache_manager: CacheManager, sample_df: pl.DataFrame) -> None:
    """Test has_cache when both formats exist."""
    ipc_path = cache_manager.cache_file("test_key", format="ipc")
    parquet_path = cache_manager.cache_file("test_key", format="parquet")

    sample_df.write_ipc(ipc_path, compression="lz4")
    sample_df.write_parquet(parquet_path)

    assert cache_manager.has_cache("test_key", any_format=True) is True
    assert cache_manager.has_cache("test_key", any_format=False) is True


def test_invalidate_single_key(cache_manager: CacheManager, sample_df: pl.DataFrame) -> None:
    """Test invalidate removes both IPC and Parquet for single key."""
    # Save both formats
    cache_manager.save_dataframe(
        "test_key",
        sample_df,
        format="ipc",
        dual_format=True,
    )

    ipc_path = cache_manager.cache_file("test_key", format="ipc")
    parquet_path = cache_manager.cache_file("test_key", format="parquet")

    assert ipc_path.exists()
    assert parquet_path.exists()

    # Invalidate
    cache_manager.invalidate("test_key")

    # Both formats should be removed
    assert not ipc_path.exists()
    assert not parquet_path.exists()


def test_invalidate_entire_cache(cache_manager: CacheManager, sample_df: pl.DataFrame) -> None:
    """Test invalidate with key=None clears entire cache."""
    # Save multiple keys
    cache_manager.save_dataframe("key1", sample_df, format="ipc", dual_format=True)
    cache_manager.save_dataframe("key2", sample_df, format="parquet", dual_format=False)

    # Verify files exist
    assert cache_manager.has_cache("key1")
    assert cache_manager.has_cache("key2")

    # Clear entire cache
    cache_manager.invalidate(key=None)

    # All cache files should be removed
    assert not cache_manager.has_cache("key1")
    assert not cache_manager.has_cache("key2")


def test_get_or_fetch_dataframe_cache_hit(cache_manager: CacheManager, sample_df: pl.DataFrame) -> None:
    """Test get_or_fetch_dataframe returns cached data on hit."""
    # Pre-populate cache
    cache_manager.save_dataframe(
        "test_key",
        sample_df,
        format="ipc",
        dual_format=True,
    )

    # Mock fetch function (should not be called)
    fetch_called = False

    def mock_fetch() -> pl.DataFrame:
        nonlocal fetch_called
        fetch_called = True
        return sample_df

    # Get from cache
    df, hit = cache_manager.get_or_fetch_dataframe(
        "test_key",
        mock_fetch,
        ttl_days=1,
        prefer_ipc=True,
    )

    assert hit is True
    assert fetch_called is False
    assert_frame_equal(df, sample_df)


def test_get_or_fetch_dataframe_cache_miss(cache_manager: CacheManager, sample_df: pl.DataFrame) -> None:
    """Test get_or_fetch_dataframe fetches and caches on miss."""
    fetch_called = False

    def mock_fetch() -> pl.DataFrame:
        nonlocal fetch_called
        fetch_called = True
        return sample_df

    # Get with cache miss
    df, hit = cache_manager.get_or_fetch_dataframe(
        "test_key",
        mock_fetch,
        ttl_days=1,
        save_format="ipc",
        dual_format=True,
    )

    assert hit is False
    assert fetch_called is True
    assert_frame_equal(df, sample_df)

    # Verify cache was saved
    assert cache_manager.has_cache("test_key")


def test_get_or_fetch_dataframe_expired_ttl(cache_manager: CacheManager, sample_df: pl.DataFrame) -> None:
    """Test get_or_fetch_dataframe refetches when cache expired."""
    # Save cache with old timestamp
    cache_manager.save_dataframe(
        "test_key",
        sample_df,
        format="ipc",
        dual_format=False,
    )

    # Manually update timestamp to expired
    index = cache_manager.load_index()
    old_time = datetime.utcnow() - timedelta(days=10)
    index["test_key"]["updated_at"] = old_time.isoformat()
    cache_manager.save_index(index)

    fetch_called = False

    def mock_fetch() -> pl.DataFrame:
        nonlocal fetch_called
        fetch_called = True
        return sample_df

    # Should refetch due to expired TTL
    df, hit = cache_manager.get_or_fetch_dataframe(
        "test_key",
        mock_fetch,
        ttl_days=1,
    )

    assert hit is False
    assert fetch_called is True


def test_is_valid_within_ttl(cache_manager: CacheManager, sample_df: pl.DataFrame) -> None:
    """Test is_valid returns True when cache within TTL."""
    cache_manager.save_dataframe("test_key", sample_df, format="ipc")

    assert cache_manager.is_valid("test_key", ttl_days=1) is True


def test_is_valid_expired_ttl(cache_manager: CacheManager, sample_df: pl.DataFrame) -> None:
    """Test is_valid returns False when cache expired."""
    cache_manager.save_dataframe("test_key", sample_df, format="ipc")

    # Manually expire the cache
    index = cache_manager.load_index()
    old_time = datetime.utcnow() - timedelta(days=10)
    index["test_key"]["updated_at"] = old_time.isoformat()
    cache_manager.save_index(index)

    assert cache_manager.is_valid("test_key", ttl_days=1) is False


def test_is_valid_no_ttl(cache_manager: CacheManager, sample_df: pl.DataFrame) -> None:
    """Test is_valid with ttl_days=0 always returns True."""
    cache_manager.save_dataframe("test_key", sample_df, format="ipc")

    # Even with expired timestamp, ttl=0 should return True
    index = cache_manager.load_index()
    old_time = datetime.utcnow() - timedelta(days=365)
    index["test_key"]["updated_at"] = old_time.isoformat()
    cache_manager.save_index(index)

    assert cache_manager.is_valid("test_key", ttl_days=0) is True


def test_cache_roundtrip_ipc(cache_manager: CacheManager, sample_df: pl.DataFrame) -> None:
    """Test full roundtrip: save IPC → load IPC."""
    cache_manager.save_dataframe(
        "test_key",
        sample_df,
        format="ipc",
        dual_format=False,
    )

    loaded = cache_manager.load_dataframe("test_key", prefer_ipc=True)

    assert loaded is not None
    assert_frame_equal(loaded, sample_df)


def test_cache_roundtrip_dual_format(cache_manager: CacheManager, sample_df: pl.DataFrame) -> None:
    """Test full roundtrip: save dual → load with IPC preferred."""
    cache_manager.save_dataframe(
        "test_key",
        sample_df,
        format="ipc",
        dual_format=True,
    )

    loaded = cache_manager.load_dataframe("test_key", prefer_ipc=True)

    assert loaded is not None
    assert_frame_equal(loaded, sample_df)

    # Also verify Parquet fallback works
    loaded_parquet = cache_manager.load_dataframe("test_key", prefer_ipc=False)
    assert loaded_parquet is not None
    assert_frame_equal(loaded_parquet, sample_df)
