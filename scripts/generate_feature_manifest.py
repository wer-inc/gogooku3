#!/usr/bin/env python3
"""
Feature Manifest Generator - Phase 3 Implementation

Purpose: Generate feature manifest from dataset for Feature-ABI validation
- Reads dataset and extracts feature columns
- Groups features by category (VSN/FAN/SAN)
- Generates YAML manifest with SHA1 hash
- Ensures model-data compatibility

Part of: ATFT P0 Phase 3 (Option B)
Author: Phase 3 Implementation (2025-11-03)
"""

import hashlib
import logging
import sys
from datetime import datetime
from pathlib import Path

import polars as pl
import yaml

# Add src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.schema_utils import infer_column_types

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def categorize_features(features: list[str]) -> dict[str, list[str]]:
    """
    Categorize features into groups for VSN/FAN/SAN architecture.

    Categories:
    - temporal: Time-series features (returns, momentum, technical)
    - cross_sectional: CS-Z normalized features
    - flow: Flow and volume features
    - macro: Market-wide features (indices, rates)
    - graph: Graph-based features (peer, sector)
    - fundamental: Financial statement features

    Args:
        features: List of feature column names

    Returns:
        Dictionary mapping category to list of features
    """
    categories = {
        "temporal": [],
        "cross_sectional": [],
        "flow": [],
        "macro": [],
        "graph": [],
        "fundamental": [],
        "other": []
    }

    for feat in features:
        # CS-Z features
        if feat.endswith("_cs_z"):
            categories["cross_sectional"].append(feat)

        # Temporal features (returns, momentum, technical)
        elif any(x in feat for x in ["returns_", "momentum_", "rsi_", "macd_", "bb_", "atr_", "volatility_"]):
            categories["temporal"].append(feat)

        # Flow features
        elif any(x in feat for x in ["flow_", "volume", "turnover", "adv_", "dollar_volume"]):
            categories["flow"].append(feat)

        # Graph features
        elif any(x in feat for x in ["peer_", "sector_", "graph_"]):
            categories["graph"].append(feat)

        # Macro features
        elif any(x in feat for x in ["macro_", "index_", "rate_", "topix", "nikkei"]):
            categories["macro"].append(feat)

        # Fundamental features
        elif any(x in feat for x in ["profit_", "roe", "roa", "debt_", "eps_", "bps_", "dps_"]):
            categories["fundamental"].append(feat)

        # Price levels
        elif feat in ["Close", "Open", "High", "Low"]:
            categories["temporal"].append(feat)

        else:
            categories["other"].append(feat)

    return categories


def compute_feature_hash(features: list[str]) -> str:
    """
    Compute SHA1 hash of feature list for Feature-ABI validation.

    Args:
        features: Sorted list of feature names

    Returns:
        SHA1 hash (first 16 chars)
    """
    # Join sorted features with newlines
    features_str = "\n".join(sorted(features))
    hash_obj = hashlib.sha1(features_str.encode('utf-8'))
    return hash_obj.hexdigest()[:16]


def generate_manifest(
    dataset_path: Path,
    output_path: Path,
    include_metadata: bool = True,
) -> dict:
    """
    Generate feature manifest from dataset.

    Args:
        dataset_path: Path to parquet dataset
        output_path: Path to save YAML manifest
        include_metadata: Include dataset metadata in manifest

    Returns:
        Manifest dictionary
    """
    logger.info(f"Loading dataset: {dataset_path}")

    # Load schema only (no data read)
    df = pl.read_parquet(dataset_path, n_rows=1)
    logger.info(f"Dataset has {len(df.columns)} columns")

    # Identify column types
    types = infer_column_types(df)

    features = types["features"]
    logger.info(f"Identified {len(features)} feature columns")

    # Categorize features
    categories = categorize_features(features)

    # Log category breakdown
    logger.info("Feature categorization:")
    for cat, feats in categories.items():
        if feats:
            logger.info(f"  {cat}: {len(feats)} features")

    # Compute feature hash
    feature_hash = compute_feature_hash(features)
    logger.info(f"Feature hash: {feature_hash}")

    # Build manifest
    manifest = {
        "version": "1.0",
        "generated": datetime.now().isoformat(),
        "dataset": str(dataset_path.name),
        "total_features": len(features),
        "feature_hash": feature_hash,
        "features": {
            "all": sorted(features),
            "by_category": {
                cat: sorted(feats)
                for cat, feats in categories.items()
                if feats
            }
        }
    }

    if include_metadata:
        manifest["metadata_columns"] = sorted(types["metadata"])
        manifest["target_columns"] = sorted(types["targets"])
        manifest["cs_z_columns"] = sorted(types["cs_z"])

        # Add statistics
        manifest["statistics"] = {
            "total_columns": len(df.columns),
            "metadata": len(types["metadata"]),
            "targets": len(types["targets"]),
            "features": len(features),
            "cs_z_features": len(types["cs_z"]),
        }

    # Save manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

    logger.info(f"✅ Saved manifest: {output_path}")

    return manifest


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate feature manifest from dataset")
    parser.add_argument(
        "--input",
        default="output/ml_dataset_with_csz.parquet",
        help="Input dataset path (default: ml_dataset_with_csz.parquet)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output manifest path (default: configs/atft/features/manifest_<N>feat.yaml)"
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Exclude metadata from manifest"
    )

    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Auto-generate output path based on feature count
        df = pl.read_parquet(input_path, n_rows=1)
        types = infer_column_types(df)
        n_features = len(types["features"])

        output_path = Path(f"configs/atft/features/manifest_{n_features}feat.yaml")

    # Generate manifest
    logger.info("="*80)
    logger.info("Feature Manifest Generation")
    logger.info("="*80)

    manifest = generate_manifest(
        input_path,
        output_path,
        include_metadata=not args.no_metadata
    )

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("✅ MANIFEST GENERATED")
    logger.info("="*80)
    logger.info(f"Output: {output_path}")
    logger.info(f"Features: {manifest['total_features']}")
    logger.info(f"Hash: {manifest['feature_hash']}")
    logger.info(f"Categories: {len(manifest['features']['by_category'])}")

    # Print category summary
    logger.info("\nCategory breakdown:")
    for cat, feats in manifest['features']['by_category'].items():
        logger.info(f"  {cat}: {len(feats)} features")

    logger.info("="*80)


if __name__ == "__main__":
    main()
