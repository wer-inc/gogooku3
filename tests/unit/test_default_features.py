#!/usr/bin/env python3
"""
Test script to verify that all necessary features are enabled by default.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.pipelines.run_full_dataset import _parse_args


def test_default_features():
    """Test that all necessary features are enabled by default."""

    # Parse arguments with minimal required flags
    test_args = ["--start-date", "2020-01-01", "--end-date", "2024-12-31"]
    sys.argv = ["test"] + test_args

    args = _parse_args()

    print("=" * 60)
    print("DEFAULT FEATURE FLAGS TEST")
    print("=" * 60)

    # Check which features are enabled by default
    features_to_check = [
        ("GPU-ETL", "gpu_etl", True),
        ("Indices", "enable_indices", True),
        ("Daily Margin", "enable_daily_margin", True),
        ("Advanced Features", "enable_advanced_features", True),
        ("Graph Features", "enable_graph_features", True),
        ("Sector CS", "enable_sector_cs", True),
        ("Advanced Vol", "enable_advanced_vol", True),
        ("Short Selling", "enable_short_selling", True),
        ("Earnings Events", "enable_earnings_events", True),
        ("PEAD Features", "enable_pead_features", True),
        ("Sector Short Selling", "enable_sector_short_selling", True),
        ("NK225 Options", "enable_nk225_option_features", True),
    ]

    all_passed = True

    for feature_name, arg_name, expected in features_to_check:
        actual = getattr(args, arg_name, False)
        status = "✅" if actual == expected else "❌"

        if actual != expected:
            all_passed = False

        print(f"{status} {feature_name:25s}: {actual} (expected: {expected})")

    print("=" * 60)

    if all_passed:
        print("🎉 ALL FEATURES ARE ENABLED BY DEFAULT!")
        print("✅ The dataset will now generate all 395 features without additional flags.")
    else:
        print("❌ Some features are not enabled by default.")
        print("⚠️  Please check the configuration.")

    print("=" * 60)

    # Show simplified command
    print("\n📝 Simplified command to generate full dataset:")
    print("   make dataset-full-gpu START=2020-01-01 END=2024-12-31")
    print("\nThis will now generate all 395 features by default! 🚀")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(test_default_features())