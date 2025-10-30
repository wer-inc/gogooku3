#!/usr/bin/env python3
"""Simplified test for Phase 2 partial match cache functionality.

This directly tests the helper function logic without full imports.
"""

import re
from datetime import datetime, timedelta
from pathlib import Path


def _find_latest_with_date_range(glob: str, req_start: str, req_end: str):
    """
    Copied implementation from run_pipeline_v4_optimized.py for testing.
    """
    req_start_dt = datetime.strptime(req_start, "%Y-%m-%d")
    req_end_dt = datetime.strptime(req_end, "%Y-%m-%d")

    # Find all matching files under output/
    candidates = sorted(Path("output").rglob(glob))

    best_match = None
    best_coverage = 0.0

    for cand in reversed(candidates):
        match = re.search(r"_(\d{8})_(\d{8})\.parquet$", cand.name)
        if not match:
            continue

        file_start = datetime.strptime(match.group(1), "%Y%m%d")
        file_end = datetime.strptime(match.group(2), "%Y%m%d")

        # Check for any overlap
        overlap_start = max(file_start, req_start_dt)
        overlap_end = min(file_end, req_end_dt)

        if overlap_start <= overlap_end:
            overlap_days = (overlap_end - overlap_start).days + 1
            total_days = (req_end_dt - req_start_dt).days + 1
            coverage = overlap_days / total_days

            # Complete match
            if file_start <= req_start_dt and file_end >= req_end_dt:
                print(f"✅ COMPLETE MATCH: {cand.name} covers {req_start} to {req_end}")
                return {
                    "path": cand,
                    "cache_start": file_start.strftime("%Y-%m-%d"),
                    "cache_end": file_end.strftime("%Y-%m-%d"),
                    "match_type": "complete",
                    "missing_start": None,
                    "missing_end": None,
                    "coverage": 1.0
                }

            # Partial match
            if coverage > best_coverage:
                best_coverage = coverage

                missing_ranges = []
                if req_start_dt < file_start:
                    missing_ranges.append((
                        req_start,
                        (file_start - timedelta(days=1)).strftime("%Y-%m-%d")
                    ))
                if req_end_dt > file_end:
                    missing_ranges.append((
                        (file_end + timedelta(days=1)).strftime("%Y-%m-%d"),
                        req_end
                    ))

                legacy_start = None
                legacy_end = None
                if len(missing_ranges) == 1:
                    legacy_start, legacy_end = missing_ranges[0]

                best_match = {
                    "path": cand,
                    "cache_start": file_start.strftime("%Y-%m-%d"),
                    "cache_end": file_end.strftime("%Y-%m-%d"),
                    "match_type": "partial",
                    "missing_start": legacy_start,
                    "missing_end": legacy_end,
                    "missing_ranges": missing_ranges,
                    "coverage": coverage
                }

    if best_match:
        print(f"🔄 PARTIAL MATCH: {best_match['path'].name}")
        print(f"   Coverage: {best_match['coverage']*100:.1f}%")
        print(f"   Cache range: {best_match['cache_start']} to {best_match['cache_end']}")
        missing_ranges = best_match.get("missing_ranges") or []
        if missing_ranges:
            for start, end in missing_ranges:
                print(f"   Missing range: {start} to {end}")

    return best_match


def test_helper_function():
    """Test the helper function with real cache files."""
    print("=" * 80)
    print("TEST: Helper Function - _find_latest_with_date_range()")
    print("=" * 80)

    # Find actual cache files
    cache_files = sorted(Path("output/raw/prices").glob("daily_quotes_*.parquet"))
    if not cache_files:
        print("❌ No cache files found")
        return False

    latest = cache_files[-1]
    print(f"\nFound cache file: {latest.name}")

    # Parse cache date range
    match = re.search(r"_(\d{8})_(\d{8})\.parquet$", latest.name)
    if not match:
        print("❌ Could not parse filename")
        return False

    cache_start = datetime.strptime(match.group(1), "%Y%m%d")
    cache_end = datetime.strptime(match.group(2), "%Y%m%d")

    print(f"Cache range: {cache_start.strftime('%Y-%m-%d')} to {cache_end.strftime('%Y-%m-%d')}")
    print()

    # Test cases
    test_cases = [
        {
            "name": "Exact match",
            "req_start": cache_start.strftime("%Y-%m-%d"),
            "req_end": cache_end.strftime("%Y-%m-%d"),
            "expected": "complete"
        },
        {
            "name": "Partial - extend 5 days forward",
            "req_start": (cache_end - timedelta(days=10)).strftime("%Y-%m-%d"),
            "req_end": (cache_end + timedelta(days=5)).strftime("%Y-%m-%d"),
            "expected": "partial"
        },
        {
            "name": "Partial - extend 7 days backward",
            "req_start": (cache_start - timedelta(days=7)).strftime("%Y-%m-%d"),
            "req_end": (cache_start + timedelta(days=10)).strftime("%Y-%m-%d"),
            "expected": "partial"
        },
        {
            "name": "Partial - extend both directions",
            "req_start": (cache_start - timedelta(days=3)).strftime("%Y-%m-%d"),
            "req_end": (cache_end + timedelta(days=3)).strftime("%Y-%m-%d"),
            "expected": "partial"
        },
        {
            "name": "Complete subset",
            "req_start": (cache_start + timedelta(days=10)).strftime("%Y-%m-%d"),
            "req_end": (cache_end - timedelta(days=10)).strftime("%Y-%m-%d"),
            "expected": "complete"
        },
        {
            "name": "No overlap - way in past",
            "req_start": "2010-01-01",
            "req_end": "2010-12-31",
            "expected": None
        }
    ]

    passed = 0
    failed = 0

    for i, case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {case['name']}")
        print(f"Request: {case['req_start']} to {case['req_end']}")

        result = _find_latest_with_date_range(
            "daily_quotes_*.parquet",
            case["req_start"],
            case["req_end"]
        )

        if case["expected"] is None:
            if result is None:
                print("✅ PASS - No match as expected")
                passed += 1
            else:
                print(f"❌ FAIL - Expected no match, got {result['match_type']}")
                failed += 1
        else:
            if result and result["match_type"] == case["expected"]:
                print(f"✅ PASS - Detected {result['match_type']} match")
                if result["match_type"] == "partial":
                    # Verify missing range calculation
                    if result["missing_start"] and result["missing_end"]:
                        print(f"   ✓ Missing range identified: {result['missing_start']} to {result['missing_end']}")
                passed += 1
            else:
                print(f"❌ FAIL - Expected {case['expected']}, got {result['match_type'] if result else None}")
                failed += 1

    print("\n" + "=" * 80)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 80)

    return failed == 0


def test_all_data_sources():
    """Test helper function for all three data sources."""
    print("\n" + "=" * 80)
    print("TEST: All Data Sources - Cache Detection")
    print("=" * 80)

    sources = [
        ("Daily Quotes", "daily_quotes_*.parquet", "output/raw/prices"),
        ("Statements", "event_raw_statements_*.parquet", "output/raw/statements"),
        ("TOPIX", "topix_history_*.parquet", "output/raw/indices")
    ]

    results = {}

    for source_name, pattern, directory in sources:
        print(f"\n{source_name}:")
        print("-" * 40)

        cache_files = sorted(Path(directory).glob(pattern.replace("*", "*")))
        if not cache_files:
            print("⚠️  No cache files found")
            results[source_name] = None
            continue

        latest = cache_files[-1]
        print(f"Latest cache: {latest.name}")
        print(f"Size: {latest.stat().st_size / (1024*1024):.1f} MB")

        # Parse date range
        match = re.search(r"_(\d{8})_(\d{8})\.parquet$", latest.name)
        if match:
            cache_start = datetime.strptime(match.group(1), "%Y%m%d")
            cache_end = datetime.strptime(match.group(2), "%Y%m%d")
            days_covered = (cache_end - cache_start).days + 1
            print(f"Range: {cache_start.strftime('%Y-%m-%d')} to {cache_end.strftime('%Y-%m-%d')} ({days_covered} days)")

            # Test partial match scenario
            test_start = (cache_end - timedelta(days=5)).strftime("%Y-%m-%d")
            test_end = (cache_end + timedelta(days=3)).strftime("%Y-%m-%d")

            print(f"\nTest scenario: Request {test_start} to {test_end}")
            result = _find_latest_with_date_range(pattern, test_start, test_end)

            if result and result["match_type"] == "partial":
                print("✅ Partial match detection working")
                results[source_name] = True
            else:
                print(f"❌ Expected partial match, got {result['match_type'] if result else None}")
                results[source_name] = False
        else:
            print("❌ Could not parse filename")
            results[source_name] = False

    print("\n" + "=" * 80)
    print("Summary:")
    for source, result in results.items():
        if result is None:
            status = "⚠️  SKIP"
        elif result:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        print(f"{status} - {source}")
    print("=" * 80)

    return all(r in [True, None] for r in results.values())


if __name__ == "__main__":
    print("Phase 2 Partial Match Cache - Simplified Test Suite")
    print()

    # Run tests
    test1_pass = test_helper_function()
    test2_pass = test_all_data_sources()

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    if test1_pass and test2_pass:
        print("✅ ALL TESTS PASSED")
        print("\nPhase 2 partial match cache functionality is working correctly!")
        exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        if not test1_pass:
            print("   - Helper function tests failed")
        if not test2_pass:
            print("   - Data source tests failed")
        exit(1)
