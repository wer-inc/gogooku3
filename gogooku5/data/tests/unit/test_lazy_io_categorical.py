#!/usr/bin/env python3
"""Unit tests for categorical optimization in lazy_io.save_with_cache."""

import os
import tempfile
from pathlib import Path

import polars as pl
import pytest


def test_categorical_encoding_explicit():
    """Test that categorical encoding is applied when explicitly specified."""
    from gogooku5.data.src.builder.utils.lazy_io import save_with_cache

    # Create sample data with low-cardinality string columns
    sample_data = pl.DataFrame({
        "Code": ["1301", "1332", "1333", "1301", "1332"],
        "Date": ["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
        "sector_code": ["0050", "0050", "0050", "0050", "0050"],
        "market_code": ["0101", "0101", "0101", "0101", "0101"],
        "Close": [100.0, 200.0, 300.0, 105.0, 210.0],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_categorical.parquet"

        # Save with categorical encoding
        parquet_path, _ = save_with_cache(
            sample_data,
            output_path,
            create_ipc=False,
            categorical_columns=["Code", "sector_code", "market_code"]
        )

        # Load back and verify types
        loaded = pl.read_parquet(parquet_path)

        # Check that categorical columns were encoded
        assert loaded["Code"].dtype == pl.Categorical, "Code should be Categorical"
        assert loaded["sector_code"].dtype == pl.Categorical, "sector_code should be Categorical"
        assert loaded["market_code"].dtype == pl.Categorical, "market_code should be Categorical"

        # Check that other columns are unchanged
        assert loaded["Date"].dtype == pl.String, "Date should remain String"
        assert loaded["Close"].dtype == pl.Float64, "Close should remain Float64"

        # Check data integrity
        assert loaded["Code"].to_list() == sample_data["Code"].to_list(), "Code values should match"


def test_categorical_encoding_env_variable():
    """Test that categorical encoding respects CATEGORICAL_COLUMNS environment variable."""
    from gogooku5.data.src.builder.utils.lazy_io import save_with_cache

    sample_data = pl.DataFrame({
        "Code": ["1301", "1332", "1333"],
        "sector_code": ["0050", "0050", "0051"],
        "Close": [100.0, 200.0, 300.0],
    })

    # Set environment variable
    os.environ["CATEGORICAL_COLUMNS"] = "Code,sector_code"

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_env_categorical.parquet"

            # Save without explicit categorical_columns (should use env var)
            parquet_path, _ = save_with_cache(
                sample_data,
                output_path,
                create_ipc=False
            )

            # Load back and verify
            loaded = pl.read_parquet(parquet_path)
            assert loaded["Code"].dtype == pl.Categorical, "Code should be Categorical (from env var)"
            assert loaded["sector_code"].dtype == pl.Categorical, "sector_code should be Categorical (from env var)"
            assert loaded["Close"].dtype == pl.Float64, "Close should remain Float64"
    finally:
        # Clean up environment variable
        os.environ.pop("CATEGORICAL_COLUMNS", None)


def test_categorical_encoding_invalid_columns():
    """Test that invalid categorical columns are handled gracefully."""
    from gogooku5.data.src.builder.utils.lazy_io import save_with_cache

    sample_data = pl.DataFrame({
        "Code": ["1301", "1332", "1333"],
        "Close": [100.0, 200.0, 300.0],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_invalid_categorical.parquet"

        # Request categorical encoding for columns that don't exist
        parquet_path, _ = save_with_cache(
            sample_data,
            output_path,
            create_ipc=False,
            categorical_columns=["Code", "sector_code", "market_code"]  # sector_code, market_code don't exist
        )

        # Load back and verify
        loaded = pl.read_parquet(parquet_path)

        # Only Code should be categorical (the only valid column)
        assert loaded["Code"].dtype == pl.Categorical, "Code should be Categorical"
        assert loaded["Close"].dtype == pl.Float64, "Close should remain Float64"


def test_categorical_encoding_no_columns():
    """Test that save_with_cache works when no categorical columns specified."""
    from gogooku5.data.src.builder.utils.lazy_io import save_with_cache

    sample_data = pl.DataFrame({
        "Code": ["1301", "1332", "1333"],
        "Close": [100.0, 200.0, 300.0],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_no_categorical.parquet"

        # Save without categorical encoding
        parquet_path, _ = save_with_cache(
            sample_data,
            output_path,
            create_ipc=False,
            categorical_columns=None
        )

        # Load back and verify (all columns should remain original types)
        loaded = pl.read_parquet(parquet_path)
        assert loaded["Code"].dtype == pl.String, "Code should remain String"
        assert loaded["Close"].dtype == pl.Float64, "Close should remain Float64"


def test_categorical_encoding_size_reduction():
    """Test that categorical encoding reduces parquet file size."""
    from gogooku5.data.src.builder.utils.lazy_io import save_with_cache

    # Create data with high repetition (ideal for categorical compression)
    sample_data = pl.DataFrame({
        "Code": ["1301"] * 500 + ["1332"] * 500 + ["1333"] * 500,
        "sector_code": ["0050"] * 1500,
        "market_code": ["0101"] * 1500,
        "Close": list(range(1500)),
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save without categorical encoding
        path_no_cat = Path(tmpdir) / "no_categorical.parquet"
        save_with_cache(
            sample_data,
            path_no_cat,
            create_ipc=False,
            categorical_columns=None
        )

        # Save with categorical encoding
        path_with_cat = Path(tmpdir) / "with_categorical.parquet"
        save_with_cache(
            sample_data,
            path_with_cat,
            create_ipc=False,
            categorical_columns=["Code", "sector_code", "market_code"]
        )

        # Compare file sizes
        size_no_cat = path_no_cat.stat().st_size
        size_with_cat = path_with_cat.stat().st_size

        # Categorical encoding should reduce size by at least 5%
        reduction_pct = (1 - size_with_cat / size_no_cat) * 100
        assert size_with_cat < size_no_cat, "Categorical encoding should reduce file size"
        assert reduction_pct >= 5, f"Expected at least 5% reduction, got {reduction_pct:.1f}%"


def test_categorical_encoding_data_integrity():
    """Test that categorical encoding preserves data integrity."""
    from gogooku5.data.src.builder.utils.lazy_io import save_with_cache

    sample_data = pl.DataFrame({
        "Code": ["1301", "1332", "1333", "1301", "1332", "1333"],
        "Date": ["2024-01-01"] * 6,
        "sector_code": ["0050", "0051", "0052", "0050", "0051", "0052"],
        "Close": [100.0, 200.0, 300.0, 105.0, 210.0, 315.0],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_integrity.parquet"

        # Save with categorical encoding
        parquet_path, _ = save_with_cache(
            sample_data,
            output_path,
            create_ipc=False,
            categorical_columns=["Code", "sector_code"]
        )

        # Load back and verify exact match
        loaded = pl.read_parquet(parquet_path)

        # Cast categorical back to string for comparison
        loaded_str = loaded.with_columns([
            pl.col("Code").cast(pl.String),
            pl.col("sector_code").cast(pl.String),
        ])

        assert loaded_str.equals(sample_data), "Data should be identical after categorical encoding"


def test_categorical_encoding_with_ipc():
    """Test that categorical encoding works with IPC cache creation."""
    from gogooku5.data.src.builder.utils.lazy_io import save_with_cache

    sample_data = pl.DataFrame({
        "Code": ["1301", "1332", "1333"],
        "sector_code": ["0050", "0050", "0051"],
        "Close": [100.0, 200.0, 300.0],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_ipc_categorical.parquet"

        # Save with both categorical encoding and IPC cache
        parquet_path, ipc_path = save_with_cache(
            sample_data,
            output_path,
            create_ipc=True,
            categorical_columns=["Code", "sector_code"]
        )

        # Verify both files exist
        assert parquet_path.exists(), "Parquet file should exist"
        assert ipc_path is not None and ipc_path.exists(), "IPC file should exist"

        # Load from both and verify consistency
        loaded_parquet = pl.read_parquet(parquet_path)
        loaded_ipc = pl.read_ipc(ipc_path)

        # Both should have categorical columns
        assert loaded_parquet["Code"].dtype == pl.Categorical
        assert loaded_ipc["Code"].dtype == pl.Categorical

        # Data should be identical
        assert loaded_parquet.with_columns([
            pl.col("Code").cast(pl.String),
            pl.col("sector_code").cast(pl.String),
        ]).equals(loaded_ipc.with_columns([
            pl.col("Code").cast(pl.String),
            pl.col("sector_code").cast(pl.String),
        ])), "Parquet and IPC data should match"


@pytest.mark.skipif(
    not Path("output_g5/dim_security.parquet").exists(),
    reason="dim_security.parquet not found (run build_dim_security.py first)"
)
def test_categorical_encoding_production():
    """Test categorical encoding with production-like data."""
    from gogooku5.data.src.builder.utils.lazy_io import save_with_cache

    # Load dim_security as sample production data
    dim_security = pl.read_parquet("output_g5/dim_security.parquet")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_production.parquet"

        # Save with categorical encoding
        parquet_path, _ = save_with_cache(
            dim_security,
            output_path,
            create_ipc=False,
            categorical_columns=["code", "sector_code", "market_code"]
        )

        # Load back and verify
        loaded = pl.read_parquet(parquet_path)

        # Check categorical types
        assert loaded["code"].dtype == pl.Categorical, "code should be Categorical"
        assert loaded["sector_code"].dtype == pl.Categorical, "sector_code should be Categorical"
        assert loaded["market_code"].dtype == pl.Categorical, "market_code should be Categorical"

        # Check data integrity (cast back to string for comparison)
        loaded_str = loaded.with_columns([
            pl.col("code").cast(pl.String),
            pl.col("sector_code").cast(pl.String),
            pl.col("market_code").cast(pl.String),
        ])
        assert loaded_str.equals(dim_security), "Production data should be identical after categorical encoding"
