#!/usr/bin/env python3
"""
dim_security テーブル生成スクリプト

グローバルに安定な sec_id を持つ証券マスタテーブルを生成します。

Usage:
    python gogooku5/data/scripts/build_dim_security.py [--output-dir OUTPUT_DIR]

Output:
    - {output_dir}/dim_security.parquet: メインのマスタテーブル
    - {output_dir}/dim_security.csv: デバッグ用 CSV

Schema:
    - sec_id: Int32 (1-based, グローバルに安定)
    - code: String (正規化済み証券コード)
    - market_code: String (市場コード)
    - market_name: String (市場名)
    - sector_code: String (33業種コード)
    - sector_name: String (33業種名)
    - effective_date: Date (初出日)
    - is_active: Boolean (上場中フラグ、常に True)

Design:
    - sec_id の決定論性を保証するため、code でソートしてから row_number() を適用
    - 同じ listed_info から生成すれば、常に同じ sec_id が割り当てられる
    - グローバルに安定なマスターテーブルとして、全チャンク・全ビルドで共有
"""

import argparse
import logging
from pathlib import Path

import polars as pl

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def find_listed_info_files(search_dirs: list[Path]) -> list[Path]:
    """
    Find all listed_info parquet files in search directories.

    Args:
        search_dirs: Directories to search

    Returns:
        List of parquet file paths
    """
    files = []
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        # Search in cache/ and raw/ subdirectories
        for subdir in ["cache", "raw", "raw/listed_info"]:
            pattern_dir = search_dir / subdir
            if pattern_dir.exists():
                matched = list(pattern_dir.glob("*listed*.parquet"))
                files.extend(matched)

    LOGGER.info(f"Found {len(files)} listed_info files")
    return files


def load_listed_info(files: list[Path]) -> pl.DataFrame:
    """
    Load and combine all listed_info files.

    Args:
        files: List of parquet files

    Returns:
        Combined DataFrame with all listed info
    """
    if not files:
        raise FileNotFoundError(
            "No listed_info files found. " "Please run dataset build at least once to generate listed_info cache."
        )

    LOGGER.info(f"Loading {len(files)} listed_info files...")

    # Read all files lazily and combine
    lazy_frames = [pl.scan_parquet(f) for f in files]
    combined = pl.concat(lazy_frames)

    # Collect to eager DataFrame
    df = combined.collect()

    LOGGER.info(f"Loaded {len(df):,} rows from listed_info")
    return df


def normalize_listed_info(df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize listed_info for dim_security creation.

    Normalization steps:
    - Trim whitespace from string columns
    - Handle null values
    - Select relevant columns

    Args:
        df: Raw listed_info DataFrame

    Returns:
        Normalized DataFrame
    """
    LOGGER.info("Normalizing listed_info...")

    # Select and normalize columns
    normalized = df.select(
        [
            pl.col("Code").str.strip_chars().alias("code"),
            pl.col("Date").alias("date"),
            pl.col("MarketCode").str.strip_chars().alias("market_code"),
            pl.col("MarketCodeName").str.strip_chars().alias("market_name"),
            pl.col("Sector33Code").str.strip_chars().alias("sector_code"),
            pl.col("Sector33CodeName").str.strip_chars().alias("sector_name"),
        ]
    )

    # Remove rows with null code (should not happen, but defensive)
    null_codes = normalized.filter(pl.col("code").is_null())
    if len(null_codes) > 0:
        LOGGER.warning(f"Removing {len(null_codes)} rows with null code")
        normalized = normalized.filter(pl.col("code").is_not_null())

    LOGGER.info(f"Normalized to {len(normalized):,} rows")
    return normalized


def build_dim_security(df: pl.DataFrame) -> pl.DataFrame:
    """
    Build dim_security table from normalized listed_info.

    Strategy:
    - Group by code to get unique codes
    - Take first market/sector (most codes don't change sector/market)
    - Use min(date) as effective_date (first appearance)
    - Sort by code to ensure deterministic sec_id assignment
    - Add row_number() as sec_id (1-based)

    Args:
        df: Normalized listed_info

    Returns:
        dim_security DataFrame with sec_id
    """
    LOGGER.info("Building dim_security table...")

    # Group by code to get unique securities
    dim = (
        df.group_by("code")
        .agg(
            [
                pl.col("market_code").first().alias("market_code"),
                pl.col("market_name").first().alias("market_name"),
                pl.col("sector_code").first().alias("sector_code"),
                pl.col("sector_name").first().alias("sector_name"),
                pl.col("date").min().alias("effective_date"),
            ]
        )
        # CRITICAL: Sort by code for deterministic sec_id
        .sort("code")
        # Add row_number as sec_id (1-based)
        .with_row_index("sec_id", offset=1)
        # Cast sec_id to Int32 (more efficient than Int64)
        .with_columns(pl.col("sec_id").cast(pl.Int32))
        # Add is_active flag (always True for now)
        .with_columns(pl.lit(True).alias("is_active"))
        # Reorder columns for readability
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

    LOGGER.info(f"Created dim_security with {len(dim):,} securities")

    # Validation
    unique_codes = dim["code"].n_unique()
    if unique_codes != len(dim):
        raise ValueError(f"Duplicate codes detected: {unique_codes} unique codes but {len(dim)} rows")

    LOGGER.info(f"✅ Validation passed: {unique_codes} unique codes")

    # Statistics
    LOGGER.info(f"   Markets: {dim['market_code'].n_unique()} unique")
    LOGGER.info(f"   Sectors: {dim['sector_code'].n_unique()} unique")
    LOGGER.info(f"   Date range: {dim['effective_date'].min()} to {dim['effective_date'].max()}")

    return dim


def save_dim_security(dim: pl.DataFrame, output_dir: Path) -> None:
    """
    Save dim_security to parquet and CSV.

    Args:
        dim: dim_security DataFrame
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / "dim_security.parquet"
    csv_path = output_dir / "dim_security.csv"

    LOGGER.info(f"Saving dim_security to {parquet_path}...")
    dim.write_parquet(
        parquet_path,
        compression="zstd",
        use_pyarrow=True,
    )

    LOGGER.info(f"Saving debug CSV to {csv_path}...")
    dim.write_csv(csv_path)

    # Report file sizes
    parquet_size = parquet_path.stat().st_size
    csv_size = csv_path.stat().st_size

    LOGGER.info("✅ Saved dim_security:")
    LOGGER.info(f"   Parquet: {parquet_path} ({parquet_size / 1024:.1f} KB)")
    LOGGER.info(f"   CSV: {csv_path} ({csv_size / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Build dim_security master table from listed_info")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_g5"),
        help="Output directory (default: output_g5)",
    )
    parser.add_argument(
        "--search-dirs",
        type=Path,
        nargs="+",
        default=[Path("output_g5"), Path("output")],
        help="Directories to search for listed_info files",
    )

    args = parser.parse_args()

    LOGGER.info("=" * 80)
    LOGGER.info("dim_security Table Builder")
    LOGGER.info("=" * 80)

    # Find listed_info files
    files = find_listed_info_files(args.search_dirs)

    # Load and normalize
    df = load_listed_info(files)
    df = normalize_listed_info(df)

    # Build dim_security
    dim = build_dim_security(df)

    # Save outputs
    save_dim_security(dim, args.output_dir)

    LOGGER.info("=" * 80)
    LOGGER.info("✅ dim_security build completed successfully")
    LOGGER.info("=" * 80)
    LOGGER.info("")
    LOGGER.info("Next steps:")
    LOGGER.info(f"  1. Verify: cat {args.output_dir / 'dim_security.csv'} | head -20")
    LOGGER.info(f"  2. Add to .env: DIM_SECURITY_PATH={args.output_dir / 'dim_security.parquet'}")
    LOGGER.info("  3. Run dataset build with sec_id support")


if __name__ == "__main__":
    main()
