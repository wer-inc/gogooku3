#!/usr/bin/env python3
"""
Cache monitoring and reporting utility.

Provides detailed statistics about graph cache usage,
helps identify large/old files, and suggests cleanup actions.

Usage:
    python scripts/maintenance/cache_monitor.py [--json] [--verbose]
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def format_size(bytes_val: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f}TB"


def get_cache_stats(cache_dir: Path) -> dict:
    """Collect comprehensive cache statistics."""
    if not cache_dir.exists():
        return {
            "exists": False,
            "path": str(cache_dir),
            "total_size_bytes": 0,
            "total_files": 0,
        }

    stats = {
        "exists": True,
        "path": str(cache_dir),
        "total_size_bytes": 0,
        "total_files": 0,
        "by_month": defaultdict(lambda: {"size_bytes": 0, "files": 0}),
        "by_age": {
            "0-7_days": {"size_bytes": 0, "files": 0},
            "8-30_days": {"size_bytes": 0, "files": 0},
            "31-90_days": {"size_bytes": 0, "files": 0},
            "90+_days": {"size_bytes": 0, "files": 0},
        },
        "largest_files": [],
        "oldest_files": [],
    }

    now = datetime.now()
    all_files: list[dict] = []

    # Walk through cache directory
    for file_path in cache_dir.rglob("*"):
        if not file_path.is_file():
            continue

        # Get file stats
        file_stat = file_path.stat()
        file_size = file_stat.st_size
        file_mtime = datetime.fromtimestamp(file_stat.st_mtime)
        file_age = (now - file_mtime).days

        # Update totals
        stats["total_size_bytes"] += file_size
        stats["total_files"] += 1

        # Track by month (from path if available)
        month_dir = None
        for part in file_path.parts:
            if part.startswith("20") and len(part) == 6:  # YYYYMM format
                month_dir = part
                break

        if month_dir:
            stats["by_month"][month_dir]["size_bytes"] += file_size
            stats["by_month"][month_dir]["files"] += 1

        # Track by age
        if file_age <= 7:
            age_bucket = "0-7_days"
        elif file_age <= 30:
            age_bucket = "8-30_days"
        elif file_age <= 90:
            age_bucket = "31-90_days"
        else:
            age_bucket = "90+_days"

        stats["by_age"][age_bucket]["size_bytes"] += file_size
        stats["by_age"][age_bucket]["files"] += 1

        # Track file details for reporting
        all_files.append({
            "path": str(file_path.relative_to(cache_dir)),
            "size_bytes": file_size,
            "age_days": file_age,
            "modified": file_mtime.isoformat(),
        })

    # Sort and get top files
    all_files.sort(key=lambda x: x["size_bytes"], reverse=True)
    stats["largest_files"] = all_files[:10]

    all_files.sort(key=lambda x: x["age_days"], reverse=True)
    stats["oldest_files"] = all_files[:10]

    return stats


def print_report(stats: dict, verbose: bool = False):
    """Print human-readable cache report."""
    if not stats["exists"]:
        print(f"Cache directory does not exist: {stats['path']}")
        return

    print("=" * 70)
    print("GRAPH CACHE STATISTICS")
    print("=" * 70)
    print(f"Cache Directory: {stats['path']}")
    print(f"Total Size:      {format_size(stats['total_size_bytes'])}")
    print(f"Total Files:     {stats['total_files']:,}")
    print()

    # By month
    if stats["by_month"]:
        print("-" * 70)
        print("CACHE BY MONTH")
        print("-" * 70)
        months = sorted(stats["by_month"].keys(), reverse=True)
        for month in months:
            month_stats = stats["by_month"][month]
            size_str = format_size(month_stats["size_bytes"])
            pct = (month_stats["size_bytes"] / stats["total_size_bytes"] * 100) if stats["total_size_bytes"] > 0 else 0
            print(f"  {month}: {size_str:>10} ({month_stats['files']:>5} files) [{pct:5.1f}%]")
        print()

    # By age
    print("-" * 70)
    print("CACHE BY AGE")
    print("-" * 70)
    for age_bucket in ["0-7_days", "8-30_days", "31-90_days", "90+_days"]:
        age_stats = stats["by_age"][age_bucket]
        size_str = format_size(age_stats["size_bytes"])
        pct = (age_stats["size_bytes"] / stats["total_size_bytes"] * 100) if stats["total_size_bytes"] > 0 else 0
        print(f"  {age_bucket:>12}: {size_str:>10} ({age_stats['files']:>5} files) [{pct:5.1f}%]")
    print()

    # Largest files
    if stats["largest_files"]:
        print("-" * 70)
        print("TOP 10 LARGEST FILES")
        print("-" * 70)
        for i, file_info in enumerate(stats["largest_files"], 1):
            size_str = format_size(file_info["size_bytes"])
            path_display = file_info["path"]
            if len(path_display) > 50:
                path_display = "..." + path_display[-47:]
            print(f"  {i:2}. {size_str:>10}  {path_display}")
        print()

    # Oldest files
    if verbose and stats["oldest_files"]:
        print("-" * 70)
        print("TOP 10 OLDEST FILES")
        print("-" * 70)
        for i, file_info in enumerate(stats["oldest_files"], 1):
            age_str = f"{file_info['age_days']} days"
            path_display = file_info["path"]
            if len(path_display) > 50:
                path_display = "..." + path_display[-47:]
            print(f"  {i:2}. {age_str:>10}  {path_display}")
        print()

    # Recommendations
    print("=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    total_mb = stats["total_size_bytes"] / (1024 * 1024)
    old_size = stats["by_age"]["90+_days"]["size_bytes"]
    old_mb = old_size / (1024 * 1024)

    if total_mb > 500:
        print(f"⚠️  Cache size ({format_size(stats['total_size_bytes'])}) exceeds recommended limit (500MB)")
        print("   Consider running: bash scripts/maintenance/cleanup_cache.sh")
    else:
        print(f"✓  Cache size ({format_size(stats['total_size_bytes'])}) is within recommended limits")

    if old_mb > 100:
        print(f"⚠️  {format_size(old_size)} in files older than 90 days")
        print(f"   Can free ~{format_size(old_size)} with: bash scripts/maintenance/cleanup_cache.sh --days 90")
    elif stats["by_age"]["90+_days"]["files"] > 0:
        print(f"✓  Only {format_size(old_size)} in old files (>90 days)")

    if stats["total_files"] == 0:
        print("✓  Cache is empty")

    print()


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Monitor graph cache usage and provide statistics"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output statistics in JSON format",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Include additional details (oldest files, etc.)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("output/graph_cache"),
        help="Path to cache directory (default: output/graph_cache)",
    )

    args = parser.parse_args()

    # Collect statistics
    stats = get_cache_stats(args.cache_dir)

    # Output
    if args.json:
        # Convert defaultdict to regular dict for JSON serialization
        stats["by_month"] = dict(stats["by_month"])
        print(json.dumps(stats, indent=2))
    else:
        print_report(stats, verbose=args.verbose)

    # Exit code based on recommendations
    if stats["total_size_bytes"] > 500 * 1024 * 1024:  # >500MB
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
