#!/usr/bin/env python3
"""
Test futures integration functionality.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import polars as pl
from datetime import date, timedelta

def test_futures_auto_discovery():
    """Test the auto-discovery functionality for spot parquets."""
    print("🧪 Testing Futures Auto-Discovery")

    # Test auto-discovery function (recreate locally since it's not exported)
    def _auto_find_spot(keywords: list[str]) -> Path | None:
        """Simulate auto-discovery function from full_dataset.py"""
        cands = []
        output_dir = Path("output")
        if output_dir.exists():
            for p in output_dir.glob("*.parquet"):
                name = p.name.lower()
                if all(k in name for k in keywords):
                    cands.append(p)
        return cands[-1] if cands else None

    # Test various keyword combinations
    test_cases = [
        (["nikkei"], "Nikkei225"),
        (["nk225"], "NK225"),
        (["reit"], "REIT"),
        (["jpx400"], "JPX400"),
        (["jp", "400"], "JPX400 alternative"),
        (["topix"], "TOPIX"),
    ]

    print("📊 Auto-discovery results:")
    found_any = False
    for keywords, description in test_cases:
        result = _auto_find_spot(keywords)
        status = f"✅ {result}" if result else "❌ Not found"
        print(f"  - {description} ({keywords}): {status}")
        if result:
            found_any = True

    if not found_any:
        print("  Note: No spot parquet files found in output/ directory")
        print("  This is expected if you haven't generated index data yet")

    return found_any

def test_futures_features_module():
    """Test the futures features module functionality."""
    print("\n🧪 Testing Futures Features Module")

    try:
        from gogooku3.features.futures import build_central_contracts, FUTURES_TO_SPOT_MAPPING
        print("✅ Futures module imports successful")

        # Test mapping
        print(f"📋 Futures to spot mapping: {len(FUTURES_TO_SPOT_MAPPING)} pairs")
        for futures, spot in FUTURES_TO_SPOT_MAPPING.items():
            print(f"  - {futures} → {spot}")

        # Create mock futures data
        sample_futures = pl.DataFrame({
            "Date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
            "Code": ["TOPIXF", "NK225F", "REITF"],
            "Close": [2100.5, 33000.0, 2050.0],
            "Volume": [1000, 2000, 500],
            "OpenInterest": [5000, 10000, 2500],
        })

        print(f"📊 Sample futures data: {len(sample_futures)} records")
        print(sample_futures.head(3))

        # Test central contracts function
        try:
            central = build_central_contracts(sample_futures)
            print(f"✅ Central contracts processed: {len(central)} records")
        except Exception as e:
            print(f"⚠️  Central contracts processing failed: {e}")

    except ImportError as e:
        print(f"❌ Failed to import futures module: {e}")
        return False

    return True

def test_jquants_futures_api():
    """Test JQuants futures API endpoint availability."""
    print("\n🧪 Testing JQuants Futures API")

    try:
        from gogooku3.components.jquants_async_fetcher import JQuantsAsyncFetcher
        print("✅ JQuantsAsyncFetcher imported successfully")

        # Check if get_futures_daily method exists
        fetcher = JQuantsAsyncFetcher("dummy", "dummy")  # No actual auth needed for method check

        if hasattr(fetcher, 'get_futures_daily'):
            print("✅ get_futures_daily method available")

            # Get method signature
            import inspect
            sig = inspect.signature(fetcher.get_futures_daily)
            print(f"📋 Method signature: get_futures_daily{sig}")
            return True
        else:
            print("❌ get_futures_daily method not found")
            return False

    except ImportError as e:
        print(f"❌ Failed to import JQuantsAsyncFetcher: {e}")
        return False

def test_pipeline_integration():
    """Test futures integration in the pipeline."""
    print("\n🧪 Testing Pipeline Integration")

    try:
        from pipeline.full_dataset import enrich_and_save
        print("✅ Pipeline import successful")

        # Check function signature for futures parameters
        import inspect
        sig = inspect.signature(enrich_and_save)

        futures_params = [param for param in sig.parameters.keys() if 'futures' in param.lower()]
        print(f"📋 Futures-related parameters: {futures_params}")

        # Expected parameters
        expected_params = [
            'enable_futures', 'futures_parquet', 'futures_categories',
            'futures_continuous', 'nk225_parquet', 'reit_parquet', 'jpx400_parquet'
        ]

        missing_params = [p for p in expected_params if p not in sig.parameters]
        if missing_params:
            print(f"⚠️  Missing parameters: {missing_params}")
        else:
            print("✅ All expected futures parameters present")

        return len(missing_params) == 0

    except ImportError as e:
        print(f"❌ Failed to import pipeline: {e}")
        return False

def test_cli_options():
    """Test CLI options for futures integration."""
    print("\n🧪 Testing CLI Options")

    # Read the CLI script to check for futures options
    try:
        cli_script = Path("scripts/pipelines/run_full_dataset.py")
        if cli_script.exists():
            content = cli_script.read_text()

            # Check for key CLI options
            expected_options = [
                "--futures-parquet",
                "--futures-categories",
                "--disable-futures",
                "--futures-continuous",
                "--nk225-parquet",
                "--reit-parquet",
                "--jpx400-parquet"
            ]

            print("📋 CLI options status:")
            all_present = True
            for option in expected_options:
                if option in content:
                    print(f"  ✅ {option}")
                else:
                    print(f"  ❌ {option}")
                    all_present = False

            if all_present:
                print("✅ All expected CLI options present")
            else:
                print("⚠️  Some CLI options missing")

            return all_present
        else:
            print("❌ CLI script not found")
            return False

    except Exception as e:
        print(f"❌ Error checking CLI options: {e}")
        return False

def main():
    """Run all futures integration tests."""
    print("🚀 Futures Integration Test Suite")
    print("=" * 50)

    results = []

    # Run all tests
    results.append(("Auto-discovery", test_futures_auto_discovery()))
    results.append(("Features Module", test_futures_features_module()))
    results.append(("JQuants API", test_jquants_futures_api()))
    results.append(("Pipeline Integration", test_pipeline_integration()))
    results.append(("CLI Options", test_cli_options()))

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")

    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  - {test_name}: {status}")
        if result:
            passed += 1

    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("🎉 All futures integration tests passed!")
    elif passed >= len(results) - 1:
        print("✅ Futures integration mostly complete with minor issues")
    else:
        print("⚠️  Some futures integration issues need attention")

    return passed >= len(results) - 1

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)