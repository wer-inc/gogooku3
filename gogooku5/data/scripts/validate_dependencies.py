#!/usr/bin/env python3
"""
Dependency validation script for gogooku5 data builder.
Checks for all required and optional dependencies before starting builds.

Usage:
    python scripts/validate_dependencies.py [--strict]

    --strict: Fail on missing optional dependencies (default: warn only)
"""

import sys
import importlib
from typing import List, Tuple, Dict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Dependency:
    """Dependency specification."""
    name: str
    import_name: str
    required: bool
    impact: str
    min_version: str | None = None


# Define all dependencies with their impact
DEPENDENCIES: List[Dependency] = [
    # Core dependencies (required)
    Dependency("polars", "polars", required=True, impact="Core DataFrame operations", min_version="0.20.0"),
    Dependency("pyarrow", "pyarrow", required=True, impact="Parquet I/O", min_version="12.0.0"),
    Dependency("pydantic", "pydantic", required=True, impact="Config validation", min_version="2.5.0"),
    Dependency("requests", "requests", required=True, impact="HTTP requests", min_version="2.31.0"),
    Dependency("numpy", "numpy", required=True, impact="Numeric operations", min_version="1.26.0"),
    Dependency("aiohttp", "aiohttp", required=True, impact="Async HTTP", min_version="3.9.0"),

    # Feature generation dependencies (optional but important)
    Dependency("yfinance", "yfinance", required=False,
               impact="40 macro/VIX features (macro_vix_*, macro_vvmd_*)", min_version="0.2.0"),
]


def check_dependency(dep: Dependency) -> Tuple[bool, str | None]:
    """
    Check if a dependency is available and meets version requirements.

    Returns:
        (available, version_or_error)
    """
    try:
        module = importlib.import_module(dep.import_name)
        version = getattr(module, "__version__", "unknown")

        # Version check (if specified and available)
        if dep.min_version and version != "unknown":
            # Simple version comparison (works for most cases)
            if version < dep.min_version:
                return False, f"version {version} < required {dep.min_version}"

        return True, version
    except ImportError as e:
        return False, str(e)


def validate_all_dependencies(strict: bool = False) -> Dict[str, dict]:
    """
    Validate all dependencies.

    Args:
        strict: If True, fail on missing optional dependencies

    Returns:
        Results dictionary with status for each dependency
    """
    results = {
        "required": {"passed": [], "failed": []},
        "optional": {"passed": [], "failed": []}
    }

    print("=" * 80)
    print("🔍 Dependency Validation Check")
    print("=" * 80)
    print()

    # Check each dependency
    for dep in DEPENDENCIES:
        available, version_or_error = check_dependency(dep)

        category = "required" if dep.required else "optional"
        status = "passed" if available else "failed"

        result = {
            "name": dep.name,
            "available": available,
            "version": version_or_error if available else None,
            "error": version_or_error if not available else None,
            "impact": dep.impact
        }

        results[category][status].append(result)

        # Print status
        if available:
            icon = "✅" if dep.required else "✅"
            print(f"{icon} {dep.name:20s} v{version_or_error:15s} ({dep.impact})")
        else:
            icon = "❌" if dep.required else "⚠️ "
            print(f"{icon} {dep.name:20s} {'MISSING':15s} ({dep.impact})")
            if not available:
                print(f"   └─ Impact: {dep.impact}")

    print()
    print("=" * 80)
    print("📊 Summary")
    print("=" * 80)

    # Required dependencies summary
    req_passed = len(results["required"]["passed"])
    req_failed = len(results["required"]["failed"])
    print(f"Required:  {req_passed} passed, {req_failed} failed")

    # Optional dependencies summary
    opt_passed = len(results["optional"]["passed"])
    opt_failed = len(results["optional"]["failed"])
    print(f"Optional:  {opt_passed} passed, {opt_failed} failed")

    # Overall status
    print()
    if req_failed > 0:
        print("❌ VALIDATION FAILED: Missing required dependencies")
        print("\nTo install missing dependencies:")
        print("  pip install -e gogooku5/data")
        return results

    if opt_failed > 0:
        print("⚠️  VALIDATION WARNING: Missing optional dependencies")
        print("\nMissing optional features:")
        for result in results["optional"]["failed"]:
            print(f"  - {result['name']}: {result['impact']}")
        print("\nTo install all dependencies:")
        print("  pip install -e gogooku5/data")

        if strict:
            print("\n❌ STRICT MODE: Failing due to missing optional dependencies")
            return results

    if req_failed == 0 and opt_failed == 0:
        print("✅ ALL DEPENDENCIES VALIDATED")
    elif req_failed == 0:
        print("✅ REQUIRED DEPENDENCIES VALIDATED (optional warnings can be ignored)")

    return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate build dependencies")
    parser.add_argument("--strict", action="store_true",
                       help="Fail on missing optional dependencies")
    args = parser.parse_args()

    results = validate_all_dependencies(strict=args.strict)

    # Exit codes
    req_failed = len(results["required"]["failed"])
    opt_failed = len(results["optional"]["failed"])

    if req_failed > 0:
        sys.exit(1)  # Required dependencies missing
    elif args.strict and opt_failed > 0:
        sys.exit(2)  # Strict mode + optional dependencies missing
    else:
        sys.exit(0)  # All good


if __name__ == "__main__":
    main()
