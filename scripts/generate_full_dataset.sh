#!/bin/bash
#
# Generate a complete dataset with all 395 features enabled
#

set -e  # Exit on error

echo "🚀 Starting full dataset generation with all features enabled"
echo "=================================================="

# Set environment variables for GPU and features
export REQUIRE_GPU=1
export USE_GPU_ETL=1
export RMM_POOL_SIZE=70GB
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=src:scripts

# Default date range (can be overridden)
START_DATE="${START:-2020-01-01}"
END_DATE="${END:-2024-12-31}"

echo "📅 Date range: $START_DATE to $END_DATE"
echo ""

# Find the most recent daily margin data
DAILY_MARGIN_PATH=$(find output -name "daily_margin_interest_*.parquet" -type f 2>/dev/null | sort | tail -1)

if [ -n "$DAILY_MARGIN_PATH" ]; then
    echo "✅ Found daily margin data: $DAILY_MARGIN_PATH"
else
    echo "⚠️  No daily margin data found - will fetch from API if credentials available"
fi

# Find other data sources
STATEMENTS_PATH=$(find output -name "statements_*.parquet" -o -name "event_raw_statements_*.parquet" 2>/dev/null | sort | tail -1)
TOPIX_PATH=$(find output -name "topix_history_*.parquet" 2>/dev/null | sort | tail -1)
INDICES_PATH=$(find output -name "indices_ohlc_*.parquet" 2>/dev/null | sort | tail -1)

echo "📦 Data sources found:"
[ -n "$STATEMENTS_PATH" ] && echo "  - Statements: $STATEMENTS_PATH"
[ -n "$TOPIX_PATH" ] && echo "  - TOPIX: $TOPIX_PATH"
[ -n "$INDICES_PATH" ] && echo "  - Indices: $INDICES_PATH"
echo ""

# Build the command
CMD="python scripts/pipelines/run_full_dataset.py"
CMD="$CMD --jquants"
CMD="$CMD --start-date $START_DATE"
CMD="$CMD --end-date $END_DATE"
CMD="$CMD --gpu-etl"
CMD="$CMD --enable-indices"
CMD="$CMD --enable-advanced-features"
CMD="$CMD --enable-sector-cs"
CMD="$CMD --enable-graph-features"
CMD="$CMD --enable-daily-margin"  # Explicitly enable daily margin
CMD="$CMD --enable-margin-weekly"  # Explicitly enable weekly margin
CMD="$CMD --enable-earnings-events"  # Enable earnings events
CMD="$CMD --enable-sector-short-selling"  # Enable sector short selling
CMD="$CMD --enable-short-selling"  # Enable short selling
CMD="$CMD --enable-advanced-vol"  # Enable advanced volatility
CMD="$CMD --enable-option-market-features"  # Enable option market features

# Add paths if found
[ -n "$DAILY_MARGIN_PATH" ] && CMD="$CMD --daily-margin-parquet '$DAILY_MARGIN_PATH'"
[ -n "$STATEMENTS_PATH" ] && CMD="$CMD --statements-parquet '$STATEMENTS_PATH'"
[ -n "$TOPIX_PATH" ] && CMD="$CMD --topix-parquet '$TOPIX_PATH'"
[ -n "$INDICES_PATH" ] && CMD="$CMD --indices-parquet '$INDICES_PATH'"

echo "🔧 Executing command:"
echo "$CMD"
echo ""
echo "⏳ This may take 30-60 minutes depending on data size..."
echo "=================================================="
echo ""

# Execute the command
eval $CMD

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Dataset generation completed successfully!"
    echo ""

    # Run verification
    echo "🔍 Running verification..."
    python scripts/verify_dataset_features.py
else
    echo ""
    echo "❌ Dataset generation failed!"
    exit 1
fi