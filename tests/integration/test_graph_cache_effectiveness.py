#!/usr/bin/env python3
"""
Graph Cache Effectiveness Test
==============================

Tests the graph caching mechanism by:
1. Extracting a small sample (3 days) from existing dataset
2. Building graph features with cache generation (INITIAL RUN)
3. Re-running to test cache hit (CACHED RUN)
4. Comparing execution times and generating report

Expected result: 1000x+ speedup on cached run
"""

import os
import time
from pathlib import Path
import polars as pl
import sys

# Fix CuDF/Numba compatibility BEFORE any imports
os.environ["NUMBA_CUDA_ENABLE_PYNVJITLINK"] = "0"

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.utils.graph_builder_gpu import FinancialGraphBuilder


def extract_sample_data(input_parquet: Path, n_days: int = 3) -> pl.DataFrame:
    """Extract the last N trading days from existing dataset."""
    print(f"📊 Loading existing dataset: {input_parquet}")
    df = pl.read_parquet(input_parquet)

    # Get last N unique dates
    dates = df.select("Date").unique().sort("Date").tail(n_days)
    last_dates = dates["Date"].to_list()

    # Filter to those dates
    sample = df.filter(pl.col("Date").is_in(last_dates))

    print(f"✅ Extracted {len(sample):,} rows across {n_days} days")
    print(f"   Date range: {min(last_dates)} to {max(last_dates)}")
    print(f"   Unique stocks: {sample.select('Code').n_unique()}")

    return sample


def run_graph_build_test(
    df: pl.DataFrame,
    cache_dir: Path,
    run_name: str,
    verbose: bool = True
) -> dict:
    """Run graph building and measure time."""
    print(f"\n{'='*60}")
    print(f"🚀 {run_name}")
    print(f"{'='*60}")

    # Prepare data for graph builder
    df_for_graph = df.select(["Code", "Date", "returns_1d"]).rename({
        "Code": "code",
        "Date": "date",
        "returns_1d": "return_1d"
    })

    # Convert to pandas (graph builder expects pandas)
    pdf = df_for_graph.to_pandas()

    # Initialize builder with cache
    builder = FinancialGraphBuilder(
        correlation_window=60,
        min_observations=40,
        correlation_threshold=0.5,
        max_edges_per_node=4,
        include_negative_correlation=True,
        correlation_method='pearson',
        cache_dir=str(cache_dir),
        verbose=verbose
    )

    dates = sorted(pdf["date"].unique())
    codes_per_date = {}
    times_per_date = {}

    total_start = time.time()

    for date in dates:
        day_df = pdf[pdf["date"] == date]
        codes = day_df["code"].unique().tolist()
        codes_per_date[str(date.date())] = len(codes)

        date_start = time.time()
        result = builder.build_graph(pdf, codes, date_end=str(date.date()))
        date_elapsed = time.time() - date_start

        times_per_date[str(date.date())] = date_elapsed

        print(f"  {date.date()}: {len(codes):4d} stocks, "
              f"{result.get('n_edges', 0):5d} edges, "
              f"{date_elapsed:.3f}s")

    total_elapsed = time.time() - total_start

    print(f"\n⏱️  Total time: {total_elapsed:.3f}s")
    print(f"   Average per day: {total_elapsed / len(dates):.3f}s")

    return {
        "run_name": run_name,
        "total_time": total_elapsed,
        "n_days": len(dates),
        "times_per_date": times_per_date,
        "codes_per_date": codes_per_date,
        "avg_time_per_day": total_elapsed / len(dates)
    }


def check_cache_files(cache_dir: Path) -> dict:
    """Check cache directory contents."""
    if not cache_dir.exists():
        return {"exists": False, "n_files": 0, "total_size_mb": 0}

    pkl_files = list(cache_dir.rglob("*.pkl"))
    total_size = sum(f.stat().st_size for f in pkl_files)

    return {
        "exists": True,
        "n_files": len(pkl_files),
        "total_size_mb": total_size / (1024 * 1024),
        "files": [f.name for f in pkl_files[:10]]  # First 10 files
    }


def generate_report(initial_result: dict, cached_result: dict, cache_info: dict):
    """Generate benchmark report."""
    print(f"\n{'='*60}")
    print(f"📊 CACHE EFFECTIVENESS REPORT")
    print(f"{'='*60}")

    speedup = initial_result["total_time"] / cached_result["total_time"] if cached_result["total_time"] > 0 else float('inf')

    print(f"\n🔹 Initial Run (Cache Generation):")
    print(f"   Total time: {initial_result['total_time']:.3f}s")
    print(f"   Avg per day: {initial_result['avg_time_per_day']:.3f}s")

    print(f"\n🔹 Cached Run (Cache Hit):")
    print(f"   Total time: {cached_result['total_time']:.3f}s")
    print(f"   Avg per day: {cached_result['avg_time_per_day']:.3f}s")

    print(f"\n🚀 Speedup: {speedup:.1f}x")
    print(f"   Time saved: {initial_result['total_time'] - cached_result['total_time']:.3f}s")
    print(f"   Percentage: {(1 - cached_result['total_time']/initial_result['total_time'])*100:.1f}% faster")

    print(f"\n💾 Cache Info:")
    print(f"   Files created: {cache_info['n_files']}")
    print(f"   Total size: {cache_info['total_size_mb']:.2f} MB")
    print(f"   Avg size per file: {cache_info['total_size_mb']/cache_info['n_files']:.2f} MB")

    print(f"\n✅ Cache is {'WORKING' if speedup > 10 else 'NOT EFFECTIVE'}")
    print(f"   Expected: >1000x speedup for pure cache hits")
    print(f"   Actual: {speedup:.1f}x speedup")

    if speedup > 100:
        print(f"   ✅ Excellent! Cache is highly effective.")
    elif speedup > 10:
        print(f"   🟡 Good. Cache is working but with some overhead.")
    else:
        print(f"   ❌ Poor. Cache may not be loading correctly.")

    print(f"\n{'='*60}\n")


def main():
    # Configuration
    INPUT_PARQUET = Path("output/ml_dataset_latest_full.parquet")
    CACHE_DIR = Path("output/graph_cache_test")
    N_DAYS = 3

    # Ensure cache directory is clean
    if CACHE_DIR.exists():
        import shutil
        shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("🧪 Graph Cache Effectiveness Test")
    print("="*60)

    # Step 1: Extract sample
    sample_df = extract_sample_data(INPUT_PARQUET, n_days=N_DAYS)

    # Check cache before (should be empty)
    cache_before = check_cache_files(CACHE_DIR)
    print(f"\n💾 Cache before: {cache_before['n_files']} files")

    # Step 2: Initial run (cache generation)
    initial_result = run_graph_build_test(
        sample_df,
        CACHE_DIR,
        "INITIAL RUN (Cache Generation)",
        verbose=True
    )

    # Check cache after initial run
    cache_after_initial = check_cache_files(CACHE_DIR)
    print(f"\n💾 Cache after initial: {cache_after_initial['n_files']} files "
          f"({cache_after_initial['total_size_mb']:.2f} MB)")

    # Step 3: Cached run (should hit cache)
    cached_result = run_graph_build_test(
        sample_df,
        CACHE_DIR,
        "CACHED RUN (Cache Hit Expected)",
        verbose=True
    )

    # Check cache after cached run (should be unchanged)
    cache_after_cached = check_cache_files(CACHE_DIR)

    # Step 4: Generate report
    generate_report(initial_result, cached_result, cache_after_initial)

    # Save detailed results
    results_file = Path("output/graph_cache_benchmark.txt")
    with open(results_file, "w") as f:
        f.write("Graph Cache Effectiveness Test Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"Initial run time: {initial_result['total_time']:.3f}s\n")
        f.write(f"Cached run time: {cached_result['total_time']:.3f}s\n")
        f.write(f"Speedup: {initial_result['total_time'] / cached_result['total_time']:.1f}x\n")
        f.write(f"\nCache files: {cache_after_initial['n_files']}\n")
        f.write(f"Cache size: {cache_after_initial['total_size_mb']:.2f} MB\n")

    print(f"📝 Results saved to: {results_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
