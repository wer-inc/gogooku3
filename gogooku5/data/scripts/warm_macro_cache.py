#!/usr/bin/env python3
"""
Macro Cache Warmer - Pre-populate macro feature caches with validation.

Purpose:
    - Fetch all required macro data (VIX, VVMD global regime) upfront
    - Verify yfinance availability and network connectivity
    - Cache data for use in chunk builds
    - Fail fast if dependencies or network issues detected

Usage:
    # Warm cache for full date range
    python scripts/warm_macro_cache.py --start 2020-01-01 --end 2025-12-31

    # Quick validation (2-day window)
    python scripts/warm_macro_cache.py --validate

    # Force refresh (ignore existing cache)
    python scripts/warm_macro_cache.py --start 2020-01-01 --end 2025-12-31 --force-refresh

Exit codes:
    0: Success (all macro data fetched and cached)
    1: yfinance not available
    2: Network/API error (Yahoo Finance unreachable)
    3: Empty data returned (possible API issue)
    4: Cache write error

Author: gogooku5 migration team
Date: 2025-11-15
"""

import sys
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import polars as pl
    from builder.features.macro.vix import load_vix_history
    from builder.features.macro.global_regime import load_global_regime_data
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Run from gogooku5/data directory or install package: pip install -e .")
    sys.exit(1)


def check_yfinance_available() -> bool:
    """Check if yfinance is installed and importable."""
    try:
        import yfinance
        print(f"✅ yfinance v{yfinance.__version__} detected")
        return True
    except ImportError:
        print("❌ yfinance not available")
        print("   Install with: pip install yfinance")
        print("   Or: pip install -e gogooku5/data")
        return False


def warm_vix_cache(start_date: str, end_date: str, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Warm VIX feature cache.

    Returns:
        Dict with keys: success (bool), rows (int), cache_path (str), error (str|None)
    """
    print(f"\n📊 Warming VIX cache ({start_date} to {end_date})...")

    try:
        vix_df = load_vix_history(
            start_date=start_date,
            end_date=end_date,
            force_refresh=force_refresh
        )

        if vix_df is None or len(vix_df) == 0:
            return {
                "success": False,
                "rows": 0,
                "cache_path": None,
                "error": "Empty DataFrame returned (possible API issue or date range mismatch)"
            }

        # Verify expected columns
        expected_cols = ["Date", "Close"]
        missing_cols = set(expected_cols) - set(vix_df.columns)
        if missing_cols:
            return {
                "success": False,
                "rows": len(vix_df),
                "cache_path": None,
                "error": f"Missing columns: {missing_cols}"
            }

        # Cache path (inferred from typical cache structure)
        cache_path = Path("output/macro/vix_history.parquet")

        print(f"   ✅ {len(vix_df):,} rows fetched")
        print(f"   📁 Cache: {cache_path}")

        return {
            "success": True,
            "rows": len(vix_df),
            "cache_path": str(cache_path),
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "rows": 0,
            "cache_path": None,
            "error": str(e)
        }


def warm_global_regime_cache(start_date: str, end_date: str, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Warm VVMD global regime cache.

    Returns:
        Dict with keys: success (bool), rows (int), cache_path (str), error (str|None)
    """
    print(f"\n🌍 Warming Global Regime cache ({start_date} to {end_date})...")

    try:
        regime_df = load_global_regime_data(
            start_date=start_date,
            end_date=end_date,
            force_refresh=force_refresh
        )

        if regime_df is None or len(regime_df) == 0:
            return {
                "success": False,
                "rows": 0,
                "cache_path": None,
                "error": "Empty DataFrame returned (possible API issue or date range mismatch)"
            }

        # Verify expected columns (at least one indicator)
        if len(regime_df.columns) < 2:  # Date + at least 1 feature
            return {
                "success": False,
                "rows": len(regime_df),
                "cache_path": None,
                "error": f"Insufficient columns: {len(regime_df.columns)} (expected >1)"
            }

        cache_path = Path("output/macro/global_regime.parquet")

        print(f"   ✅ {len(regime_df):,} rows fetched")
        print(f"   📁 Cache: {cache_path}")

        return {
            "success": True,
            "rows": len(regime_df),
            "cache_path": str(cache_path),
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "rows": 0,
            "cache_path": None,
            "error": str(e)
        }


def save_health_marker(results: Dict[str, Any], output_path: Path):
    """Save success marker for subsequent builds to skip preflight check."""
    health_data = {
        "timestamp": datetime.now().isoformat(),
        "vix": results["vix"],
        "global_regime": results["global_regime"],
        "overall_success": results["vix"]["success"] and results["global_regime"]["success"]
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(health_data, f, indent=2)

    print(f"\n💾 Health marker saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Warm macro feature caches with validation")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--validate", action="store_true",
                       help="Quick validation mode (2-day window)")
    parser.add_argument("--force-refresh", action="store_true",
                       help="Force refresh (ignore existing cache)")
    parser.add_argument("--health-marker", type=str,
                       default="output/cache/macro/vix_health.json",
                       help="Path to save health marker")

    args = parser.parse_args()

    # Determine date range
    if args.validate:
        # Validation mode: 2-day window
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        print("🔍 VALIDATION MODE: 2-day window")
    else:
        if not args.start or not args.end:
            print("❌ Error: --start and --end required (or use --validate)")
            parser.print_help()
            sys.exit(1)
        start_str = args.start
        end_str = args.end

    print("=" * 80)
    print("🔥 Macro Cache Warmer")
    print("=" * 80)
    print(f"Date range: {start_str} to {end_str}")
    print(f"Force refresh: {args.force_refresh}")
    print()

    # Step 1: Check yfinance
    if not check_yfinance_available():
        sys.exit(1)

    # Step 2: Warm VIX cache
    vix_result = warm_vix_cache(start_str, end_str, args.force_refresh)

    # Step 3: Warm Global Regime cache
    regime_result = warm_global_regime_cache(start_str, end_str, args.force_refresh)

    # Step 4: Save health marker
    results = {
        "vix": vix_result,
        "global_regime": regime_result
    }
    save_health_marker(results, Path(args.health_marker))

    # Step 5: Report results
    print("\n" + "=" * 80)
    print("📋 Summary")
    print("=" * 80)

    if vix_result["success"] and regime_result["success"]:
        print("✅ ALL MACRO CACHES WARMED SUCCESSFULLY")
        print(f"\n   VIX:           {vix_result['rows']:,} rows")
        print(f"   Global Regime: {regime_result['rows']:,} rows")
        print(f"\n   Total features: 40 (10 VIX + 30 VVMD)")
        print("\n✅ Build can proceed with full macro feature set")
        sys.exit(0)
    else:
        print("❌ MACRO CACHE WARMING FAILED")
        if not vix_result["success"]:
            print(f"\n   VIX Error: {vix_result['error']}")
        if not regime_result["success"]:
            print(f"\n   Global Regime Error: {regime_result['error']}")

        print("\n⚠️  Build will produce 2727 columns (missing 40 macro features)")
        print("   Fix errors above before running dataset build")

        # Determine appropriate exit code
        if "Empty DataFrame" in str(vix_result.get("error", "")) or \
           "Empty DataFrame" in str(regime_result.get("error", "")):
            sys.exit(3)  # Empty data
        else:
            sys.exit(2)  # Network/API error


if __name__ == "__main__":
    main()
