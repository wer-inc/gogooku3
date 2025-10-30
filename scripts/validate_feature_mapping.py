#!/usr/bin/env python3
"""
Feature Category Mapping Validator

Validates that actual dataset features can be mapped to model's expected categories.
Identifies missing features and provides corrected category mappings.
"""

import sys
from pathlib import Path

import polars as pl
import yaml


def load_actual_features(data_path: str) -> list[str]:
    """Load actual feature columns from dataset."""
    df = pl.read_parquet(data_path)

    # Exclude target and metadata columns
    exclude = {
        "date",
        "Date",
        "code",
        "Code",
        "target",
        "target_1d",
        "target_5d",
        "target_10d",
        "target_20d",
    }

    features = [c for c in df.columns if c not in exclude]
    return features


def load_feature_categories(config_path: str) -> dict[str, list[str]]:
    """Load feature categories from config."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config["feature_categories"]


def analyze_mapping(
    actual_features: list[str], category_config: dict[str, list[str]]
) -> dict:
    """Analyze mapping between actual features and categories."""

    # Normalize feature names (lowercase for matching)
    actual_set = {f.lower() for f in actual_features}

    results = {
        "mapped": {},
        "unmapped": [],
        "missing_from_data": {},
        "category_counts": {},
    }

    # Check each category
    for category, expected_features in category_config.items():
        if not expected_features:
            continue

        mapped = []
        missing = []

        for feat in expected_features:
            feat_lower = feat.lower()

            # Try exact match
            if feat_lower in actual_set:
                mapped.append(feat)
            else:
                # Try partial match (e.g., 'returns_1d' matches 'returns_1d')
                found = False
                for actual_feat in actual_features:
                    if (
                        feat_lower in actual_feat.lower()
                        or actual_feat.lower() in feat_lower
                    ):
                        mapped.append(actual_feat)
                        found = True
                        break

                if not found:
                    missing.append(feat)

        results["mapped"][category] = mapped
        results["missing_from_data"][category] = missing
        results["category_counts"][category] = len(mapped)

    # Find unmapped features
    all_mapped = set()
    for mapped_list in results["mapped"].values():
        all_mapped.update(f.lower() for f in mapped_list)

    results["unmapped"] = [f for f in actual_features if f.lower() not in all_mapped]

    return results


def print_analysis(results: dict, actual_count: int):
    """Print analysis results."""
    print("=" * 80)
    print("📊 FEATURE MAPPING ANALYSIS")
    print("=" * 80)
    print()

    print(f"Total features in dataset: {actual_count}")
    print()

    print("📋 Category Mapping Summary:")
    print("-" * 80)
    total_mapped = 0
    for category in sorted(results["category_counts"].keys()):
        count = results["category_counts"][category]
        total_mapped += count
        missing_count = len(results["missing_from_data"].get(category, []))

        status = "✅" if missing_count == 0 else "⚠️"
        print(
            f"  {status} {category:15s}: {count:3d} mapped, {missing_count:3d} missing from data"
        )

    print("-" * 80)
    print(f"  Total mapped: {total_mapped}")
    print(f"  Total unmapped: {len(results['unmapped'])}")
    print()

    # Show unmapped features
    if results["unmapped"]:
        print("⚠️  UNMAPPED FEATURES (not in any category):")
        print("-" * 80)
        for feat in sorted(results["unmapped"]):
            print(f"  - {feat}")
        print()

    # Show missing features (in config but not in data)
    print("❌ MISSING FROM DATA (defined in config but not found):")
    print("-" * 80)
    has_missing = False
    for category, missing_list in results["missing_from_data"].items():
        if missing_list:
            has_missing = True
            print(f"\n  {category}:")
            for feat in sorted(missing_list):
                print(f"    - {feat}")

    if not has_missing:
        print("  None! All configured features exist in data.")
    print()


def generate_corrected_config(
    results: dict, actual_features: list[str], output_path: str
):
    """Generate corrected feature category config."""

    corrected = {"feature_categories": {}}

    # Add mapped features
    for category, mapped_list in results["mapped"].items():
        if mapped_list:
            corrected["feature_categories"][category] = sorted(mapped_list)

    # Add unmapped to 'additional' category
    if results["unmapped"]:
        if "additional" not in corrected["feature_categories"]:
            corrected["feature_categories"]["additional"] = []
        corrected["feature_categories"]["additional"] = sorted(results["unmapped"])

    # Add expected counts
    corrected["expected_counts"] = {
        category: len(features)
        for category, features in corrected["feature_categories"].items()
    }
    corrected["expected_counts"]["total"] = len(actual_features)

    # Write to file
    with open(output_path, "w") as f:
        yaml.dump(corrected, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Corrected config saved to: {output_path}")
    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate feature category mapping")
    parser.add_argument(
        "--data",
        default="output/ml_dataset_20251001_154821_full.parquet",
        help="Path to ML dataset parquet file",
    )
    parser.add_argument(
        "--config",
        default="configs/atft/feature_categories.yaml",
        help="Path to feature categories config",
    )
    parser.add_argument(
        "--output",
        default="configs/atft/feature_categories_corrected.yaml",
        help="Path to save corrected config",
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.data).exists():
        print(f"❌ Data file not found: {args.data}", file=sys.stderr)
        sys.exit(1)

    if not Path(args.config).exists():
        print(f"❌ Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    # Load data
    print("Loading actual features from data...")
    actual_features = load_actual_features(args.data)

    print("Loading category config...")
    category_config = load_feature_categories(args.config)

    # Analyze
    print("Analyzing mapping...\n")
    results = analyze_mapping(actual_features, category_config)

    # Print results
    print_analysis(results, len(actual_features))

    # Generate corrected config
    generate_corrected_config(results, actual_features, args.output)

    print("=" * 80)
    print("✅ VALIDATION COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print(f"  1. Review corrected config: {args.output}")
    print("  2. Update model config to use corrected feature counts")
    print("  3. Re-run training with correct feature mapping")
    print()


if __name__ == "__main__":
    main()
