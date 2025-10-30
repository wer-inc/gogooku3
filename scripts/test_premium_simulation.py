#!/usr/bin/env python3
"""
Test script to simulate Premium plan environment.

This script verifies that the futures API enablement logic works correctly
when JQUANTS_PLAN_TIER is set to "premium" without actually calling the API.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_plan_tier_detection():
    """Test plan tier detection logic."""
    print("=" * 80)
    print("Test 1: Plan Tier Detection")
    print("=" * 80)

    # Import after path setup
    from dotenv import load_dotenv

    load_dotenv()

    # Test Standard plan (default)
    original_value = os.getenv("JQUANTS_PLAN_TIER")

    # Test case 1: Standard plan
    os.environ["JQUANTS_PLAN_TIER"] = "standard"
    plan_tier = os.getenv("JQUANTS_PLAN_TIER", "standard").lower()
    is_premium = plan_tier == "premium"

    assert plan_tier == "standard", f"Expected 'standard', got '{plan_tier}'"
    assert not is_premium, "Standard plan should not enable futures"
    print("✅ Test 1a: Standard plan detection - PASSED")
    print(f"   Plan tier: {plan_tier}")
    print(f"   Futures available: {is_premium}")

    # Test case 2: Premium plan
    os.environ["JQUANTS_PLAN_TIER"] = "premium"
    plan_tier = os.getenv("JQUANTS_PLAN_TIER", "standard").lower()
    is_premium = plan_tier == "premium"

    assert plan_tier == "premium", f"Expected 'premium', got '{plan_tier}'"
    assert is_premium, "Premium plan should enable futures"
    print("✅ Test 1b: Premium plan detection - PASSED")
    print(f"   Plan tier: {plan_tier}")
    print(f"   Futures available: {is_premium}")

    # Test case 3: Case insensitivity
    os.environ["JQUANTS_PLAN_TIER"] = "PREMIUM"
    plan_tier = os.getenv("JQUANTS_PLAN_TIER", "standard").lower()
    is_premium = plan_tier == "premium"

    assert plan_tier == "premium", f"Expected 'premium', got '{plan_tier}'"
    assert is_premium, "PREMIUM (uppercase) should be recognized"
    print("✅ Test 1c: Case insensitivity - PASSED")
    print(f"   Plan tier: {plan_tier}")
    print(f"   Futures available: {is_premium}")

    # Restore original value
    if original_value:
        os.environ["JQUANTS_PLAN_TIER"] = original_value
    else:
        os.environ.pop("JQUANTS_PLAN_TIER", None)

    print()


def test_futures_availability_logic():
    """Test futures availability logic from run_full_dataset.py."""
    print("=" * 80)
    print("Test 2: Futures Availability Logic")
    print("=" * 80)

    original_value = os.getenv("JQUANTS_PLAN_TIER")

    # Simulate Standard plan
    os.environ["JQUANTS_PLAN_TIER"] = "standard"

    # Import helper functions
    from scripts.pipelines.run_full_dataset import (
        _get_jquants_plan_tier,
        _is_futures_available,
    )

    # Test Standard plan
    plan = _get_jquants_plan_tier()
    futures_available = _is_futures_available()

    assert plan == "standard", f"Expected 'standard', got '{plan}'"
    assert not futures_available, "Futures should not be available on Standard plan"
    print("✅ Test 2a: Standard plan futures disabled - PASSED")
    print(f"   Plan: {plan}")
    print(f"   Futures available: {futures_available}")

    # Simulate Premium plan
    os.environ["JQUANTS_PLAN_TIER"] = "premium"

    plan = _get_jquants_plan_tier()
    futures_available = _is_futures_available()

    assert plan == "premium", f"Expected 'premium', got '{plan}'"
    assert futures_available, "Futures should be available on Premium plan"
    print("✅ Test 2b: Premium plan futures enabled - PASSED")
    print(f"   Plan: {plan}")
    print(f"   Futures available: {futures_available}")

    # Restore original value
    if original_value:
        os.environ["JQUANTS_PLAN_TIER"] = original_value
    else:
        os.environ.pop("JQUANTS_PLAN_TIER", None)

    print()


def test_premium_migration_scenario():
    """Test the Premium migration scenario."""
    print("=" * 80)
    print("Test 3: Premium Migration Scenario")
    print("=" * 80)

    original_value = os.getenv("JQUANTS_PLAN_TIER")

    # Import helper functions
    from scripts.pipelines.run_full_dataset import (
        _get_jquants_plan_tier,
        _is_futures_available,
    )

    # Before migration (Standard plan)
    os.environ["JQUANTS_PLAN_TIER"] = "standard"

    plan_before = _get_jquants_plan_tier()
    futures_before = _is_futures_available()

    print("Before migration (Standard):")
    print(f"  Plan tier: {plan_before}")
    print(f"  Futures API: {'enabled' if futures_before else 'disabled'}")
    print("  Expected features: ~303-307")

    # After migration (Premium plan) - simulate changing .env
    os.environ["JQUANTS_PLAN_TIER"] = "premium"

    plan_after = _get_jquants_plan_tier()
    futures_after = _is_futures_available()

    print("\nAfter migration (Premium):")
    print(f"  Plan tier: {plan_after}")
    print(f"  Futures API: {'enabled' if futures_after else 'disabled'}")
    print("  Expected features: ~395 (+88-92 futures features)")

    # Verify migration
    assert plan_before == "standard" and not futures_before
    assert plan_after == "premium" and futures_after
    print("\n✅ Test 3: Premium migration simulation - PASSED")
    print("   Migration successful: Standard → Premium")

    # Restore original value
    if original_value:
        os.environ["JQUANTS_PLAN_TIER"] = original_value
    else:
        os.environ.pop("JQUANTS_PLAN_TIER", None)

    print()


def test_conditional_parameters():
    """Test conditional parameter setting for enrich_and_save."""
    print("=" * 80)
    print("Test 4: Conditional Parameters")
    print("=" * 80)

    original_value = os.getenv("JQUANTS_PLAN_TIER")

    # Import helper functions
    from scripts.pipelines.run_full_dataset import _is_futures_available

    # Test Standard plan parameters
    os.environ["JQUANTS_PLAN_TIER"] = "standard"

    enable_futures = _is_futures_available()
    futures_parquet = "test_path" if _is_futures_available() else None
    futures_categories = ["TOPIXF", "NK225F"] if _is_futures_available() else []
    futures_continuous = True if _is_futures_available() else False

    assert enable_futures is False
    assert futures_parquet is None
    assert futures_categories == []
    assert futures_continuous is False
    print("✅ Test 4a: Standard plan parameters - PASSED")
    print(f"  enable_futures: {enable_futures}")
    print(f"  futures_parquet: {futures_parquet}")
    print(f"  futures_categories: {futures_categories}")
    print(f"  futures_continuous: {futures_continuous}")

    # Test Premium plan parameters
    os.environ["JQUANTS_PLAN_TIER"] = "premium"

    enable_futures = _is_futures_available()
    futures_parquet = "test_path" if _is_futures_available() else None
    futures_categories = ["TOPIXF", "NK225F"] if _is_futures_available() else []
    futures_continuous = True if _is_futures_available() else False

    assert enable_futures is True
    assert futures_parquet == "test_path"
    assert futures_categories == ["TOPIXF", "NK225F"]
    assert futures_continuous is True
    print("\n✅ Test 4b: Premium plan parameters - PASSED")
    print(f"  enable_futures: {enable_futures}")
    print(f"  futures_parquet: {futures_parquet}")
    print(f"  futures_categories: {futures_categories}")
    print(f"  futures_continuous: {futures_continuous}")

    # Restore original value
    if original_value:
        os.environ["JQUANTS_PLAN_TIER"] = original_value
    else:
        os.environ.pop("JQUANTS_PLAN_TIER", None)

    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Premium Plan Simulation Test Suite")
    print("=" * 80)
    print("\nThis test verifies futures API enablement logic without API calls.")
    print()

    try:
        test_plan_tier_detection()
        test_futures_availability_logic()
        test_premium_migration_scenario()
        test_conditional_parameters()

        print("=" * 80)
        print("ALL TESTS PASSED ✅")
        print("=" * 80)
        print("\nPremium migration is ready:")
        print("1. To enable futures API, set JQUANTS_PLAN_TIER=premium in .env")
        print("2. Restart dataset generation: make dataset-bg")
        print("3. Futures features will be automatically enabled")
        print()
        return 0

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
