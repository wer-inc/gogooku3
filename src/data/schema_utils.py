#!/usr/bin/env python3
"""
Schema Normalization Utilities

Purpose: Handle column name variations and ensure consistent schema
- Maps common aliases to canonical column names
- Handles case sensitivity issues (code → Code, date → Date)
- Part of P0-3 Phase 3 fixes

Author: Phase 3 Implementation (2025-11-03)
"""

import logging

import polars as pl

logger = logging.getLogger(__name__)

# Canonical column name mapping
# Maps various aliases to the standard column name used in the system
ALIAS_TO_CANON: dict[str, str] = {
    # Core identifiers
    "code": "Code",
    "CODE": "Code",
    "stock_code": "Code",
    "localcode": "LocalCode",
    "LOCALCODE": "LocalCode",
    "local_code": "LocalCode",

    # Date/Time
    "date": "Date",
    "DATE": "Date",
    "trading_date": "Date",

    # Market identifiers
    "marketcode": "MarketCode",
    "MARKETCODE": "MarketCode",
    "market_code": "MarketCode",
    "section": "Section",
    "SECTION": "Section",
    "section_norm": "section_norm",

    # Index
    "row_idx": "row_idx",
    "ROW_IDX": "row_idx",
    "index": "row_idx",
}


def normalize_schema(
    df: pl.DataFrame,
    alias_map: dict[str, str] | None = None,
    strict: bool = False,
) -> pl.DataFrame:
    """
    Normalize DataFrame schema by mapping alias column names to canonical names.

    Args:
        df: Input DataFrame with potentially non-standard column names
        alias_map: Custom alias mapping (uses ALIAS_TO_CANON by default)
        strict: If True, raise error on unrecognized columns

    Returns:
        DataFrame with normalized column names

    Example:
        >>> df = pl.DataFrame({"code": [1301], "date": ["2025-01-01"]})
        >>> df_norm = normalize_schema(df)
        >>> df_norm.columns  # ['Code', 'Date']
    """
    if alias_map is None:
        alias_map = ALIAS_TO_CANON

    # Build rename mapping for columns that need normalization
    rename_map = {}
    for col in df.columns:
        if col in alias_map:
            canonical = alias_map[col]
            if canonical != col:
                rename_map[col] = canonical
                logger.debug(f"Normalizing column: {col} → {canonical}")

    # Apply renaming
    if rename_map:
        df = df.rename(rename_map)
        logger.info(f"Normalized {len(rename_map)} column names: {list(rename_map.keys())}")
    else:
        logger.debug("No column normalization needed")

    # Strict mode: check for unrecognized columns
    if strict:
        expected_prefixes = ["target_", "returns_", "adv_", "volume", "Close", "Open"]
        metadata_cols = set(ALIAS_TO_CANON.values())

        unrecognized = []
        for col in df.columns:
            if col not in metadata_cols and not any(col.startswith(p) for p in expected_prefixes):
                # Check if it ends with common suffixes
                if not (col.endswith("_cs_z") or col.endswith("_lag") or "_rank" in col):
                    unrecognized.append(col)

        if unrecognized:
            logger.warning(f"Unrecognized columns (strict mode): {unrecognized[:10]}")
            if len(unrecognized) > 10:
                logger.warning(f"... and {len(unrecognized) - 10} more")

    return df


def validate_required_columns(
    df: pl.DataFrame,
    required: list[str] | None = None,
) -> None:
    """
    Validate that DataFrame contains required columns.

    Args:
        df: DataFrame to validate
        required: List of required column names (uses defaults if None)

    Raises:
        ValueError: If any required columns are missing
    """
    if required is None:
        required = ["Code", "Date"]  # Minimum required

    missing = [col for col in required if col not in df.columns]

    if missing:
        available = df.columns[:20]  # Show first 20 for debugging
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {available}"
            f"{' ...' if len(df.columns) > 20 else ''}"
        )

    logger.debug(f"Validated required columns: {required}")


def infer_column_types(df: pl.DataFrame) -> dict[str, list[str]]:
    """
    Infer semantic types of columns based on naming patterns.

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary mapping category to list of column names

    Categories:
        - metadata: Code, Date, Section, etc.
        - targets: target_* columns
        - cs_z: Cross-sectional Z-score features (*_cs_z)
        - features: Other numeric features
    """
    cols = df.columns

    metadata_names = {"Code", "Date", "Section", "MarketCode", "LocalCode",
                      "section_norm", "row_idx"}

    categorized = {
        "metadata": [c for c in cols if c in metadata_names],
        "targets": [c for c in cols if c.startswith("target_")],
        "cs_z": [c for c in cols if c.endswith("_cs_z")],
        "features": []
    }

    # Remaining columns are features
    assigned = set(categorized["metadata"] + categorized["targets"] + categorized["cs_z"])
    categorized["features"] = [c for c in cols if c not in assigned]

    # Log summary
    logger.info("Column categorization:")
    for cat, col_list in categorized.items():
        logger.info(f"  {cat}: {len(col_list)} columns")

    return categorized


def add_missing_metadata(
    df: pl.DataFrame,
    code_col: str = "Code",
    date_col: str = "Date",
) -> pl.DataFrame:
    """
    Add missing metadata columns with default values if needed.

    Args:
        df: Input DataFrame
        code_col: Name of stock code column
        date_col: Name of date column

    Returns:
        DataFrame with guaranteed metadata columns
    """
    # Ensure Code and Date exist
    if code_col not in df.columns:
        raise ValueError(f"Cannot proceed without {code_col} column")
    if date_col not in df.columns:
        raise ValueError(f"Cannot proceed without {date_col} column")

    # Add optional metadata with defaults
    exprs = []

    if "section_norm" not in df.columns:
        exprs.append(pl.lit(0).alias("section_norm"))
        logger.debug("Added default section_norm=0")

    if "row_idx" not in df.columns:
        exprs.append(pl.lit(None).cast(pl.Int64).alias("row_idx"))
        logger.debug("Added default row_idx=null")

    if exprs:
        df = df.with_columns(exprs)

    return df


if __name__ == "__main__":
    # Test normalization
    import sys

    logging.basicConfig(level=logging.INFO)

    # Create test DataFrame with non-standard names
    test_df = pl.DataFrame({
        "code": [1301, 1332, 1605],
        "date": ["2025-01-01", "2025-01-01", "2025-01-01"],
        "returns_1d": [0.01, -0.005, 0.02],
        "volume": [1000000, 500000, 2000000],
    })

    print("Original columns:", test_df.columns)

    # Normalize
    normalized = normalize_schema(test_df)
    print("Normalized columns:", normalized.columns)

    # Validate
    try:
        validate_required_columns(normalized)
        print("✅ Validation passed")
    except ValueError as e:
        print(f"❌ Validation failed: {e}")
        sys.exit(1)

    # Infer types
    types = infer_column_types(normalized)
    print("\nColumn types:")
    for cat, cols in types.items():
        print(f"  {cat}: {cols}")

    print("\n✅ Schema normalization test passed")
