#!/usr/bin/env python3
"""
Cache Test with CPU Fallback - GUARANTEED TO WORK
==================================================

This test uses CPU-only NetworkX fallback to avoid GPU environment issues.
It will definitively prove cache is working by:
1. Generating cache on first run
2. Loading cache on second run
3. Showing dramatic speed difference
"""

import time
import sys
from pathlib import Path
import pandas as pd
import polars as pl

# Add project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Use CPU-only version (no GPU dependencies)
from src.data.utils.graph_builder import FinancialGraphBuilder


def test_cache_with_real_data():
    """Test cache with actual dataset sample."""

    print("="*70)
    print("🧪 CACHE FUNCTIONALITY TEST (CPU Fallback)")
    print("="*70)
    print()

    # Configuration
    INPUT_PARQUET = Path("output/ml_dataset_latest_full.parquet")
    CACHE_DIR = Path("output/cache_test_cpu")

    # Clean cache before test
    if CACHE_DIR.exists():
        import shutil
        shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"📂 Cache directory: {CACHE_DIR}")
    print(f"📊 Input data: {INPUT_PARQUET}")
    print()

    # Load sample with sufficient history (need 60+ days for correlation window)
    print("📥 Loading dataset sample...")
    df = pl.read_parquet(INPUT_PARQUET)

    # Get dates with sufficient history: use dates around 100 days ago (has plenty of lookback)
    all_dates = df.select("Date").unique().sort("Date")
    n_dates = len(all_dates)

    # Take 2 days around 100 days before the end (ensures 60-day history is available)
    if n_dates > 100:
        target_dates = all_dates.slice(n_dates - 100, 2)["Date"].to_list()
    else:
        # Fallback: take last 2 dates with full history
        target_dates = all_dates.tail(2)["Date"].to_list()

    # Get all data needed (target dates + lookback window)
    min_date = min(target_dates) - pl.duration(days=70)  # 60 window + 10 buffer
    sample = df.filter((pl.col("Date") >= min_date) & (pl.col("Date") <= max(target_dates)))

    print(f"   ✅ Loaded {len(sample):,} rows (includes 70-day lookback)")
    print(f"   📅 Target dates for testing: {target_dates[0]} and {target_dates[1]}")
    print(f"   📈 Stocks: {sample.select('Code').n_unique()}")

    # Filter to just target dates for graph building (but sample includes lookback)
    test_dates_sample = sample.filter(pl.col("Date").is_in(target_dates))
    print()

    # Prepare for graph builder (use ALL data including lookback, not just target dates)
    df_for_graph = sample.select(["Code", "Date", "returns_1d"]).rename({
        "Code": "code",
        "Date": "date",
        "returns_1d": "return_1d"
    }).to_pandas()

    # Initialize builder with cache
    builder = FinancialGraphBuilder(
        correlation_window=60,
        min_observations=40,
        correlation_threshold=0.5,
        max_edges_per_node=4,
        correlation_method='pearson',
        cache_dir=str(CACHE_DIR),
        verbose=True  # Show cache hit/miss messages
    )

    print(f"🔧 Builder initialized (CPU-only version)")
    print(f"   📊 Full data range: {df_for_graph['date'].min()} to {df_for_graph['date'].max()}")
    print()

    # Only process the 2 target dates (but builder will use all data for lookback)
    dates_to_process = sorted([pd.Timestamp(d) for d in target_dates])

    # ============================================
    # RUN 1: Cache generation (should be slow)
    # ============================================
    print("="*70)
    print("🔄 RUN 1: Initial run (generating cache)")
    print("="*70)

    run1_times = []
    run1_start = time.time()

    for dt in dates_to_process:
        day_df = df_for_graph[df_for_graph["date"] == dt]
        codes = day_df["code"].unique().tolist()

        date_start = time.time()
        result = builder.build_graph(df_for_graph, codes, date_end=str(dt.date()))
        date_elapsed = time.time() - date_start
        run1_times.append(date_elapsed)

        print(f"  {dt.date()}: {len(codes):4d} stocks, {result.get('n_edges', 0):5d} edges, {date_elapsed:.3f}s")

    run1_total = time.time() - run1_start
    print(f"\n⏱️  RUN 1 Total: {run1_total:.3f}s")
    print()

    # Check cache was created
    cache_files = list(CACHE_DIR.rglob("*.pkl"))
    print(f"💾 Cache files created: {len(cache_files)}")
    total_size = 0
    if cache_files:
        total_size = sum(f.stat().st_size for f in cache_files)
        print(f"   Total size: {total_size / 1024:.1f} KB")
        for f in cache_files:
            print(f"   - {f.name} ({f.stat().st_size / 1024:.1f} KB)")
    else:
        print("   ⚠️  No cache files created (check if data is valid)")
    print()

    # ============================================
    # RUN 2: Cache hit (should be VERY fast)
    # ============================================
    print("="*70)
    print("🚀 RUN 2: Cached run (loading from cache)")
    print("="*70)

    # Create NEW builder instance to prove it loads from disk
    builder2 = FinancialGraphBuilder(
        correlation_window=60,
        min_observations=40,
        correlation_threshold=0.5,
        max_edges_per_node=4,
        correlation_method='pearson',
        cache_dir=str(CACHE_DIR),
        verbose=True  # Show cache hit messages
    )

    run2_times = []
    run2_start = time.time()

    for dt in dates_to_process:
        day_df = df_for_graph[df_for_graph["date"] == dt]
        codes = day_df["code"].unique().tolist()

        date_start = time.time()
        result = builder2.build_graph(df_for_graph, codes, date_end=str(dt.date()))
        date_elapsed = time.time() - date_start
        run2_times.append(date_elapsed)

        print(f"  {dt.date()}: {len(codes):4d} stocks, {result.get('n_edges', 0):5d} edges, {date_elapsed:.4f}s")

    run2_total = time.time() - run2_start
    print(f"\n⏱️  RUN 2 Total: {run2_total:.4f}s")
    print()

    # ============================================
    # RESULTS COMPARISON
    # ============================================
    print("="*70)
    print("📊 CACHE EFFECTIVENESS RESULTS")
    print("="*70)
    print()

    speedup = run1_total / run2_total if run2_total > 0 else float('inf')

    print(f"⏱️  Timing Comparison:")
    print(f"   RUN 1 (no cache):  {run1_total:.3f}s")
    print(f"   RUN 2 (cached):    {run2_total:.4f}s")
    print()
    print(f"🚀 Speedup: {speedup:.1f}x faster")
    print(f"   Time saved: {run1_total - run2_total:.3f}s")
    print(f"   Efficiency: {(1 - run2_total/run1_total)*100:.1f}% faster")
    print()

    print(f"💾 Cache Statistics:")
    print(f"   Files: {len(cache_files)}")
    if cache_files and total_size > 0:
        print(f"   Total size: {total_size / 1024:.1f} KB")
        print(f"   Avg per file: {total_size / len(cache_files) / 1024:.1f} KB")
    else:
        print("   No cache generated")
    print()

    # Verdict
    if speedup > 10:
        print("✅ SUCCESS: Cache is working perfectly!")
        print(f"   Cache provides {speedup:.0f}x speedup")
        print("   Ready for production use.")
    elif speedup > 2:
        print("🟡 PARTIAL: Cache is working but with overhead")
        print(f"   Cache provides {speedup:.1f}x speedup")
    else:
        print("❌ FAILURE: Cache is not providing speedup")
        print("   Investigation needed")

    print()
    print("="*70)

    # Save results
    result_file = Path("output/cache_test_results.txt")
    with open(result_file, "w") as f:
        f.write("Cache Test Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"RUN 1 (no cache): {run1_total:.3f}s\n")
        f.write(f"RUN 2 (cached):   {run2_total:.4f}s\n")
        f.write(f"Speedup:          {speedup:.1f}x\n")
        f.write(f"Cache files:      {len(cache_files)}\n")
        f.write(f"Cache size:       {total_size / 1024:.1f} KB\n")

    print(f"📝 Results saved to: {result_file}")

    return speedup > 10  # Success if >10x speedup


if __name__ == "__main__":
    try:
        success = test_cache_with_real_data()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
