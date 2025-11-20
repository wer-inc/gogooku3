"""Unit tests for dim_security table generation."""

from pathlib import Path

import polars as pl
import pytest


def test_dim_security_deterministic():
    """Test that dim_security generation is deterministic."""
    # Create sample listed_info data
    sample_data = pl.DataFrame(
        {
            "Code": ["13010", "13050", "13060"],
            "Date": ["2024-01-01", "2024-01-01", "2024-01-01"],
            "MarketCode": ["0101", "0109", "0109"],
            "MarketCodeName": ["東証一部", "その他", "その他"],
            "Sector33Code": ["0050", "9999", "9999"],
            "Sector33CodeName": ["水産・農林業", "その他", "その他"],
        }
    )

    # Build dim_security twice
    dim1 = _build_dim_security_from_df(sample_data)
    dim2 = _build_dim_security_from_df(sample_data)

    # Should be identical
    assert dim1.equals(dim2), "dim_security generation should be deterministic"


def test_dim_security_sec_id_sequential():
    """Test that sec_id is sequential and 1-based."""
    sample_data = pl.DataFrame(
        {
            "Code": ["A", "B", "C", "D", "E"],
            "Date": ["2024-01-01"] * 5,
            "MarketCode": ["0101"] * 5,
            "MarketCodeName": ["Market"] * 5,
            "Sector33Code": ["0050"] * 5,
            "Sector33CodeName": ["Sector"] * 5,
        }
    )

    dim = _build_dim_security_from_df(sample_data)

    # Check sec_id range
    assert dim["sec_id"].min() == 1, "sec_id should start at 1"
    assert dim["sec_id"].max() == 5, "sec_id should end at row count"

    # Check no gaps
    expected_ids = list(range(1, 6))
    actual_ids = sorted(dim["sec_id"].to_list())
    assert actual_ids == expected_ids, "sec_id should be sequential"


def test_dim_security_no_duplicate_codes():
    """Test that dim_security has no duplicate codes."""
    # Sample with duplicate codes (same code, different dates)
    sample_data = pl.DataFrame(
        {
            "Code": ["13010", "13010", "13050"],
            "Date": ["2024-01-01", "2024-01-02", "2024-01-01"],
            "MarketCode": ["0101", "0101", "0109"],
            "MarketCodeName": ["東証一部", "東証一部", "その他"],
            "Sector33Code": ["0050", "0050", "9999"],
            "Sector33CodeName": ["水産・農林業", "水産・農林業", "その他"],
        }
    )

    dim = _build_dim_security_from_df(sample_data)

    # Should have unique codes
    assert dim["code"].n_unique() == 2, "Should have 2 unique codes"
    assert len(dim) == 2, "Should have 2 rows (one per code)"


def test_dim_security_sorted_by_code():
    """Test that dim_security is sorted by code."""
    sample_data = pl.DataFrame(
        {
            "Code": ["Z", "A", "M", "C", "B"],
            "Date": ["2024-01-01"] * 5,
            "MarketCode": ["0101"] * 5,
            "MarketCodeName": ["Market"] * 5,
            "Sector33Code": ["0050"] * 5,
            "Sector33CodeName": ["Sector"] * 5,
        }
    )

    dim = _build_dim_security_from_df(sample_data)

    # Check sorting
    assert dim["code"].to_list() == ["A", "B", "C", "M", "Z"], "Codes should be sorted"


def test_dim_security_schema():
    """Test that dim_security has correct schema."""
    sample_data = pl.DataFrame(
        {
            "Code": ["13010"],
            "Date": ["2024-01-01"],
            "MarketCode": ["0101"],
            "MarketCodeName": ["東証一部"],
            "Sector33Code": ["0050"],
            "Sector33CodeName": ["水産・農林業"],
        }
    )

    dim = _build_dim_security_from_df(sample_data)

    # Check columns
    expected_columns = [
        "sec_id",
        "code",
        "market_code",
        "market_name",
        "sector_code",
        "sector_name",
        "effective_date",
        "is_active",
    ]
    assert dim.columns == expected_columns, f"Expected columns: {expected_columns}, got: {dim.columns}"

    # Check types
    assert dim["sec_id"].dtype == pl.Int32, "sec_id should be Int32"
    assert dim["code"].dtype == pl.String, "code should be String"
    assert dim["effective_date"].dtype == pl.Date, "effective_date should be Date"
    assert dim["is_active"].dtype == pl.Boolean, "is_active should be Boolean"


def test_dim_security_normalization():
    """Test that dim_security normalizes whitespace."""
    sample_data = pl.DataFrame(
        {
            "Code": ["  13010  ", "13050"],
            "Date": ["2024-01-01", "2024-01-01"],
            "MarketCode": [" 0101 ", "0109"],
            "MarketCodeName": ["  東証一部  ", "その他"],
            "Sector33Code": [" 0050 ", "9999"],
            "Sector33CodeName": ["  水産・農林業  ", "その他"],
        }
    )

    dim = _build_dim_security_from_df(sample_data)

    # Check trimming
    assert dim["code"][0] == "13010", "Code should be trimmed"
    assert dim["market_code"][0] == "0101", "Market code should be trimmed"


def test_dim_security_effective_date():
    """Test that effective_date is the earliest date for each code."""
    sample_data = pl.DataFrame(
        {
            "Code": ["13010", "13010", "13050"],
            "Date": ["2024-01-05", "2024-01-01", "2024-02-01"],
            "MarketCode": ["0101", "0101", "0109"],
            "MarketCodeName": ["東証一部", "東証一部", "その他"],
            "Sector33Code": ["0050", "0050", "9999"],
            "Sector33CodeName": ["水産・農林業", "水産・農林業", "その他"],
        }
    )

    dim = _build_dim_security_from_df(sample_data)

    # Check effective_date
    code_13010 = dim.filter(pl.col("code") == "13010")
    assert str(code_13010["effective_date"][0]) == "2024-01-01", "effective_date should be min date"


def test_dim_security_is_active_default():
    """Test that is_active defaults to True."""
    sample_data = pl.DataFrame(
        {
            "Code": ["13010"],
            "Date": ["2024-01-01"],
            "MarketCode": ["0101"],
            "MarketCodeName": ["東証一部"],
            "Sector33Code": ["0050"],
            "Sector33CodeName": ["水産・農林業"],
        }
    )

    dim = _build_dim_security_from_df(sample_data)

    # Check is_active
    assert dim["is_active"][0] is True, "is_active should default to True"


# Helper function to build dim_security from DataFrame
def _build_dim_security_from_df(df: pl.DataFrame) -> pl.DataFrame:
    """
    Build dim_security from normalized listed_info DataFrame.

    This replicates the logic from build_dim_security.py.
    """
    # Normalize
    normalized = df.select(
        [
            pl.col("Code").str.strip_chars().alias("code"),
            pl.col("Date").str.strptime(pl.Date, "%Y-%m-%d").alias("date"),
            pl.col("MarketCode").str.strip_chars().alias("market_code"),
            pl.col("MarketCodeName").str.strip_chars().alias("market_name"),
            pl.col("Sector33Code").str.strip_chars().alias("sector_code"),
            pl.col("Sector33CodeName").str.strip_chars().alias("sector_name"),
        ]
    )

    # Group and build dim_security
    dim = (
        normalized.group_by("code")
        .agg(
            [
                pl.col("market_code").first().alias("market_code"),
                pl.col("market_name").first().alias("market_name"),
                pl.col("sector_code").first().alias("sector_code"),
                pl.col("sector_name").first().alias("sector_name"),
                pl.col("date").min().alias("effective_date"),
            ]
        )
        .sort("code")
        .with_row_index("sec_id", offset=1)
        .with_columns(pl.col("sec_id").cast(pl.Int32))
        .with_columns(pl.lit(True).alias("is_active"))
        .select(
            [
                "sec_id",
                "code",
                "market_code",
                "market_name",
                "sector_code",
                "sector_name",
                "effective_date",
                "is_active",
            ]
        )
    )

    return dim


@pytest.mark.skipif(
    not Path("output_g5/dim_security.parquet").exists(),
    reason="dim_security.parquet not found (run build_dim_security.py first)",
)
def test_dim_security_file_exists():
    """Test that generated dim_security file exists and is valid."""
    dim_path = Path("output_g5/dim_security.parquet")
    dim = pl.read_parquet(dim_path)

    # Basic validation
    assert len(dim) > 0, "dim_security should not be empty"
    assert "sec_id" in dim.columns, "dim_security should have sec_id column"
    assert "code" in dim.columns, "dim_security should have code column"
    assert dim["code"].n_unique() == len(dim), "All codes should be unique"
    assert dim["sec_id"].is_sorted(), "sec_id should be sorted"
    assert dim["code"].is_sorted(), "code should be sorted"
