#!/usr/bin/env python3
"""
CS-Z Dataset Builder - Phase 3 Implementation

Purpose: Rebuild dataset with Cross-Sectional Z-score features
- Loads current 389-feature dataset
- Applies schema normalization
- Generates ~78 CS-Z features (date-grouped Z-scores)
- Target: ~467 total columns (389 base + 78 CS-Z)

Part of: ATFT P0 Phase 3 (Option B)
Author: Phase 3 Implementation (2025-11-03)
"""

import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

# Add src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.schema_utils import infer_column_types, normalize_schema, validate_required_columns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def identify_csz_candidates(df: pl.DataFrame) -> list[str]:
    """
    Identify columns that should have CS-Z versions generated.

    CS-Z candidates are numeric features that benefit from cross-sectional
    normalization (Z-score computed across all stocks at each date).

    Categories:
    - Returns: returns_*d (1d, 5d, 10d, 20d)
    - Volume: adv_*d, volume_*, turnover*
    - Volatility: volatility_*d, vol_*
    - Price-based: Close, Open, High, Low (relative to date mean)
    - Technical: rsi_*, macd_*, bb_*, atr_*
    - Financial: profit_margin, roe, roa, debt_ratio, etc.

    Returns:
        List of column names to generate CS-Z features for
    """
    candidates = []

    # Define patterns for CS-Z candidates
    patterns = [
        # Returns (most important for ranking)
        r"^returns_\d+d$",
        r"^ret_\d+d(_vs_\w+)?$",

        # Volume features
        r"^adv_\d+d$",
        r"^volume_",
        r"^turnover",
        r"^dollar_volume",

        # Volatility
        r"^volatility_\d+d$",
        r"^vol_\d+d$",
        r"^realized_vol",

        # Technical indicators
        r"^rsi_",
        r"^macd_",
        r"^bb_",
        r"^atr_",
        r"^obv$",
        r"^mfi$",

        # Momentum
        r"^momentum_\d+d$",
        r"^roc_\d+d$",

        # Price levels (for cross-sectional comparison)
        r"^(Close|Open|High|Low)$",

        # Financial ratios
        r"^(profit_margin|roe|roa|debt_ratio|current_ratio)$",
        r"^(pe_ratio|pb_ratio|ps_ratio)$",

        # Flow features
        r"^flow_",
        r"^net_buy_",

        # Market structure
        r"^spread_",
        r"^depth_",

        # Peer/sector features
        r"^peer_",
        r"^sector_",
    ]

    # Compile patterns
    compiled = [re.compile(p) for p in patterns]

    # Check each column
    for col in df.columns:
        # Skip if already a CS-Z column
        if col.endswith("_cs_z"):
            continue

        # Skip metadata and targets
        if col in {"Code", "Date", "Section", "MarketCode", "LocalCode",
                   "section_norm", "row_idx"}:
            continue

        if col.startswith("target_"):
            continue

        # Check if column matches any pattern
        for pattern in compiled:
            if pattern.match(col):
                # Verify it's numeric
                dtype = df[col].dtype
                if dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64):
                    candidates.append(col)
                    break

    logger.info(f"Identified {len(candidates)} CS-Z candidates")
    return sorted(candidates)


def generate_csz_features(
    df: pl.DataFrame,
    target_cols: list[str],
    epsilon: float = 1e-9,
    batch_size: int = 50,
) -> pl.DataFrame:
    """
    Generate Cross-Sectional Z-score features.

    For each target column, computes:
        cs_z = (value - date_mean) / (date_std + epsilon)

    Where date_mean and date_std are computed across all stocks
    at the same date.

    Args:
        df: Input DataFrame
        target_cols: Columns to generate CS-Z for
        epsilon: Small value to prevent division by zero
        batch_size: Process columns in batches to manage memory

    Returns:
        DataFrame with CS-Z columns added
    """
    logger.info(f"Generating CS-Z for {len(target_cols)} columns...")

    # Process in batches to manage memory
    n_batches = (len(target_cols) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(target_cols))
        batch_cols = target_cols[start:end]

        logger.info(f"Processing batch {batch_idx + 1}/{n_batches} "
                   f"({len(batch_cols)} columns)")

        # Generate CS-Z expressions for this batch
        exprs = []
        for col in batch_cols:
            cs_z_name = f"{col}_cs_z"

            # Skip if already exists
            if cs_z_name in df.columns:
                logger.debug(f"Skipping {cs_z_name} (already exists)")
                continue

            # Compute: (value - date_mean) / (date_std + epsilon)
            # Using .over("Date") for date-grouped operations
            expr = (
                (pl.col(col) - pl.col(col).mean().over("Date"))
                / (pl.col(col).std().over("Date") + epsilon)
            ).alias(cs_z_name)

            exprs.append(expr)

        # Apply batch
        if exprs:
            df = df.with_columns(exprs)
            logger.info(f"Added {len(exprs)} CS-Z columns in batch {batch_idx + 1}")

    # Count total CS-Z columns
    cs_z_count = len([c for c in df.columns if c.endswith("_cs_z")])
    logger.info(f"✅ Total CS-Z columns: {cs_z_count}")

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build dataset with CS-Z features")
    parser.add_argument(
        "--input",
        default="output/ml_dataset_latest_clean.parquet",
        help="Input dataset path (default: ml_dataset_latest_clean.parquet)"
    )
    parser.add_argument(
        "--output",
        default="output/ml_dataset_with_csz.parquet",
        help="Output dataset path (default: ml_dataset_with_csz.parquet)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without writing"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for CS-Z generation (default: 50)"
    )

    args = parser.parse_args()

    # Validate input exists
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    logger.info(f"Loading dataset: {input_path}")
    start_time = datetime.now()

    # Load dataset
    df = pl.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Get initial statistics
    types = infer_column_types(df)
    logger.info("Initial column breakdown:")
    for cat, cols in types.items():
        logger.info(f"  {cat}: {len(cols)} columns")

    # Step 1: Schema normalization
    logger.info("\n" + "="*80)
    logger.info("Step 1: Schema Normalization")
    logger.info("="*80)

    df = normalize_schema(df)
    validate_required_columns(df)

    # Step 2: Identify CS-Z candidates
    logger.info("\n" + "="*80)
    logger.info("Step 2: Identify CS-Z Candidates")
    logger.info("="*80)

    candidates = identify_csz_candidates(df)
    logger.info(f"\nCS-Z candidates ({len(candidates)} total):")

    # Group by pattern for display
    groups = {
        "returns": [c for c in candidates if "return" in c],
        "volume": [c for c in candidates if "volume" in c or "turnover" in c or "adv_" in c],
        "volatility": [c for c in candidates if "vol" in c],
        "technical": [c for c in candidates if any(x in c for x in ["rsi", "macd", "bb", "atr"])],
        "price": [c for c in candidates if c in ["Close", "Open", "High", "Low"]],
        "other": []
    }

    # Assign remaining to "other"
    assigned = set()
    for g in groups.values():
        assigned.update(g)
    groups["other"] = [c for c in candidates if c not in assigned]

    for group, cols in groups.items():
        if cols:
            logger.info(f"  {group}: {len(cols)} columns")
            logger.info(f"    Sample: {cols[:5]}")

    if args.dry_run:
        logger.info("\n" + "="*80)
        logger.info("DRY RUN - Would generate:")
        logger.info(f"  Input: {len(df.columns)} columns")
        logger.info(f"  CS-Z candidates: {len(candidates)} columns")
        logger.info(f"  Output: {len(df.columns) + len(candidates)} columns")
        logger.info("="*80)
        return

    # Step 3: Generate CS-Z features
    logger.info("\n" + "="*80)
    logger.info("Step 3: Generate CS-Z Features")
    logger.info("="*80)

    df = generate_csz_features(df, candidates, batch_size=args.batch_size)

    # Step 4: Save output
    logger.info("\n" + "="*80)
    logger.info("Step 4: Save Output")
    logger.info("="*80)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing to: {output_path}")
    df.write_parquet(output_path, compression="zstd")

    # Get final statistics
    file_size = output_path.stat().st_size / (1024**3)  # GB
    elapsed = (datetime.now() - start_time).total_seconds()

    types_final = infer_column_types(df)

    logger.info("\n" + "="*80)
    logger.info("✅ COMPLETE")
    logger.info("="*80)
    logger.info(f"Output: {output_path}")
    logger.info(f"Rows: {len(df):,}")
    logger.info(f"Columns: {len(df.columns)} (added {len(df.columns) - len(pl.read_parquet(input_path).columns)})")
    logger.info(f"File size: {file_size:.2f} GB")
    logger.info(f"Time: {elapsed:.1f}s")
    logger.info("\nFinal column breakdown:")
    for cat, cols in types_final.items():
        logger.info(f"  {cat}: {len(cols)} columns")
    logger.info("="*80)

    # Update symlink
    symlink_path = output_path.parent / "ml_dataset_latest_with_csz.parquet"
    if symlink_path.exists() or symlink_path.is_symlink():
        symlink_path.unlink()
    symlink_path.symlink_to(output_path.name)
    logger.info(f"✅ Updated symlink: {symlink_path.name} → {output_path.name}")


if __name__ == "__main__":
    main()
