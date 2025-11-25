#!/usr/bin/env python3
"""
Phase 1 Complete Validation: 2025年全四半期 (Q1-Q4)

Validates that all 2025 chunks are built with schema v1.4.0
"""

import json
import os
from pathlib import Path
from typing import Dict, List


def _default_chunks_dir() -> Path:
    """Resolve default chunks directory relative to DATA_OUTPUT_DIR or repo root."""
    env_dir = os.getenv("DATA_OUTPUT_DIR")
    if env_dir:
        return Path(env_dir) / "chunks"
    # Fallback: gogooku5/data/output/chunks
    return Path(__file__).resolve().parents[2] / "data" / "output" / "chunks"


def validate_2025_complete() -> bool:
    """Validate all 2025 chunks with manifest v1.4.0"""

    base_dir = _default_chunks_dir()
    expected_hash = "81c029b120e9c5e2"
    expected_version = "1.4.0"
    expected_quarters = ["2025Q1", "2025Q2", "2025Q3", "2025Q4"]

    results: List[Dict[str, any]] = []
    all_passed = True

    print("=" * 80)
    print("Phase 1 Complete Validation: 2025年全四半期 (manifest v1.4.0)")
    print("=" * 80)
    print()

    for quarter in expected_quarters:
        chunk_dir = base_dir / quarter
        status_file = chunk_dir / "status.json"

        if not status_file.exists():
            print(f"❌ {quarter}: Status file not found")
            results.append({"quarter": quarter, "status": "missing", "passed": False})
            all_passed = False
            continue

        with open(status_file) as f:
            status = json.load(f)

        # Validate state
        if status.get("state") != "completed":
            print(f"❌ {quarter}: Build not completed (state={status.get('state')})")
            results.append({"quarter": quarter, "status": "incomplete", "passed": False})
            all_passed = False
            continue

        # Validate schema version
        schema_version = status.get("feature_schema_version")
        if schema_version != expected_version:
            print(f"❌ {quarter}: Wrong schema version (got {schema_version}, expected {expected_version})")
            results.append(
                {"quarter": quarter, "status": "wrong_version", "schema_version": schema_version, "passed": False}
            )
            all_passed = False
            continue

        # Validate schema hash
        schema_hash = status.get("feature_schema_hash")
        if schema_hash != expected_hash:
            print(f"❌ {quarter}: Wrong schema hash (got {schema_hash}, expected {expected_hash})")
            results.append({"quarter": quarter, "status": "wrong_hash", "schema_hash": schema_hash, "passed": False})
            all_passed = False
            continue

        # All checks passed
        rows = status.get("rows", 0)
        duration = status.get("build_duration_seconds", 0)
        timestamp = status.get("timestamp", "unknown")

        print(f"✅ {quarter}: Valid")
        print(f"   - Rows: {rows:,}")
        print(f"   - Build time: {duration:.1f}s")
        print(f"   - Timestamp: {timestamp}")
        print(f"   - Schema: v{schema_version} (hash: {schema_hash})")
        print()

        results.append(
            {
                "quarter": quarter,
                "status": "valid",
                "rows": rows,
                "duration_seconds": duration,
                "timestamp": timestamp,
                "schema_version": schema_version,
                "schema_hash": schema_hash,
                "passed": True,
            }
        )

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)

    total_rows = sum(r.get("rows", 0) for r in results)
    total_duration = sum(r.get("duration_seconds", 0) for r in results)
    passed_count = sum(1 for r in results if r["passed"])

    print(f"Total quarters: {len(expected_quarters)}")
    print(f"Passed: {passed_count}/{len(expected_quarters)}")
    print(f"Total rows: {total_rows:,}")
    print(f"Total build time: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print()

    if all_passed:
        print("✅ All 2025 chunks validated successfully with manifest v1.4.0!")
        print()
        print("🎉 Option 1 (2025年全再ビルド) COMPLETED!")
        print()
        print("Next steps:")
        print("  1. Test with Apex Ranker")
        print("  2. If successful, consider Option 2 (2020-2024 rebuild)")
        return True
    else:
        print("❌ Validation failed for some chunks")
        return False


if __name__ == "__main__":
    success = validate_2025_complete()
    exit(0 if success else 1)
