#!/bin/bash
# Monitor 2022-2024 build progress and auto-execute Phase B' steps when ready

set -euo pipefail

LOG_FILE="/tmp/build_2022_2024_full.log"
CHUNKS_DIR="/workspace/gogooku3/output_g5/chunks"
OUTPUT_DIR="/workspace/gogooku3/output_g5/datasets"
GOGOOKU5_ROOT="/workspace/gogooku3/gogooku5"

echo "========================================================================"
echo "📊 Phase B' Build Monitor & Auto-Executor"
echo "========================================================================"
echo ""

# Function to check if chunk is completed
is_chunk_complete() {
    local chunk_id=$1
    local status_file="${CHUNKS_DIR}/${chunk_id}/status.json"

    if [[ ! -f "$status_file" ]]; then
        return 1
    fi

    local state=$(jq -r '.state' "$status_file" 2>/dev/null || echo "unknown")
    [[ "$state" == "completed" ]]
}

# Function to check if all required chunks are complete
check_all_chunks_complete() {
    local required_chunks=("2022Q1" "2022Q2" "2022Q3" "2022Q4" "2023Q1" "2023Q2" "2023Q3" "2023Q4" "2024Q1" "2024Q2" "2024Q3" "2024Q4")

    for chunk in "${required_chunks[@]}"; do
        if ! is_chunk_complete "$chunk"; then
            return 1
        fi
    done
    return 0
}

# Function to display progress
show_progress() {
    echo "🔄 Current Status:"
    echo ""

    for year in 2022 2023 2024; do
        echo "  ${year}:"
        for q in Q1 Q2 Q3 Q4; do
            local chunk_id="${year}${q}"
            local status_file="${CHUNKS_DIR}/${chunk_id}/status.json"

            if [[ -f "$status_file" ]]; then
                local state=$(jq -r '.state' "$status_file" 2>/dev/null || echo "unknown")
                local rows=$(jq -r '.rows // 0' "$status_file" 2>/dev/null || echo "0")

                if [[ "$state" == "completed" ]]; then
                    echo "    ${chunk_id}: ✅ Completed (${rows} rows)"
                elif [[ "$state" == "running" ]]; then
                    echo "    ${chunk_id}: ⏳ Running..."
                else
                    echo "    ${chunk_id}: ⏸️  Pending"
                fi
            else
                echo "    ${chunk_id}: ⏸️  Not started"
            fi
        done
    done
    echo ""
}

# Function to get total rows from completed chunks
get_total_rows() {
    local total=0
    for year in 2022 2023 2024; do
        for q in Q1 Q2 Q3 Q4; do
            local chunk_id="${year}${q}"
            local status_file="${CHUNKS_DIR}/${chunk_id}/status.json"

            if [[ -f "$status_file" ]]; then
                local rows=$(jq -r '.rows // 0' "$status_file" 2>/dev/null || echo "0")
                total=$((total + rows))
            fi
        done
    done
    echo $total
}

# Main monitoring loop
echo "🕐 Waiting for all chunks to complete..."
echo ""

while true; do
    clear
    echo "========================================================================"
    echo "📊 Phase B' Build Monitor (Auto-refresh every 30s)"
    echo "========================================================================"
    echo ""
    echo "⏰ $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    show_progress

    if check_all_chunks_complete; then
        echo "========================================================================"
        echo "✅ ALL CHUNKS COMPLETED! Starting Phase B' Step 2-6..."
        echo "========================================================================"
        echo ""

        # Get final stats
        total_rows=$(get_total_rows)
        echo "📊 Final Statistics:"
        echo "   Total rows: ${total_rows}"
        echo ""

        # Execute Phase B' Step 2-6
        echo "🚀 Executing Phase B' Steps..."
        echo ""

        # Step 2: Merge chunks
        echo "───────────────────────────────────────────────────────────────────────"
        echo "Step 2/5: Merging 2022-2024 chunks..."
        echo "───────────────────────────────────────────────────────────────────────"
        cd "$GOGOOKU5_ROOT"
        PYTHONPATH=data/src python data/tools/merge_chunks.py \
            --chunks-dir output_g5/chunks \
            --pattern "202[234]Q[1234]" \
            --output output_g5/datasets/ml_dataset_2022_2024_merged.parquet

        # Step 2b: Extract 2023-2024
        echo ""
        echo "───────────────────────────────────────────────────────────────────────"
        echo "Step 2b/5: Extracting 2023-2024 date range..."
        echo "───────────────────────────────────────────────────────────────────────"
        python data/tools/extract_date_range.py \
            --input output_g5/datasets/ml_dataset_2022_2024_merged.parquet \
            --output output_g5/datasets/ml_dataset_2023_2024_extracted.parquet \
            --start-date 2023-01-01 \
            --end-date 2024-12-31

        # Step 3: Post-processing
        echo ""
        echo "───────────────────────────────────────────────────────────────────────"
        echo "Step 3/5: Applying post-processing (Beta/Alpha, Basis Gate, Graph)..."
        echo "───────────────────────────────────────────────────────────────────────"
        # (This will be implemented once post-processing scripts are ready)
        echo "⚠️  Post-processing scripts not yet ready - skipping for now"
        echo "   Using extracted dataset as-is"
        cp output_g5/datasets/ml_dataset_2023_2024_extracted.parquet \
           output_g5/datasets/ml_dataset_2023_2024_final.parquet

        # Step 4: Drop high NULL columns
        echo ""
        echo "───────────────────────────────────────────────────────────────────────"
        echo "Step 4/5: Dropping columns with NULL rate ≥90%..."
        echo "───────────────────────────────────────────────────────────────────────"
        PYTHONPATH=data/src python data/tools/drop_high_null_columns.py \
            --input output_g5/datasets/ml_dataset_2023_2024_final.parquet \
            --output output_g5/datasets/ml_dataset_2023_2024_clean.parquet \
            --threshold 90.0 \
            --keep-col date --keep-col code

        # Step 5: NULL rate comparison
        echo ""
        echo "───────────────────────────────────────────────────────────────────────"
        echo "Step 5/5: Generating NULL rate improvement report..."
        echo "───────────────────────────────────────────────────────────────────────"
        PYTHONPATH=data/src python data/tools/compare_null_rates.py \
            --before output_g5/datasets/ml_dataset_2023_2025_final_pruned.parquet \
            --after output_g5/datasets/ml_dataset_2023_2024_clean.parquet \
            --output docs/NULL_RATE_IMPROVEMENT_REPORT_20251119.md

        echo ""
        echo "========================================================================"
        echo "✅ Phase B' Steps 2-5 COMPLETED!"
        echo "========================================================================"
        echo ""
        echo "📊 Final Output:"
        echo "   Clean dataset: output_g5/datasets/ml_dataset_2023_2024_clean.parquet"
        echo "   NULL report: docs/NULL_RATE_IMPROVEMENT_REPORT_20251119.md"
        echo ""
        echo "📋 Next Steps:"
        echo "   1. Review NULL rate improvement report"
        echo "   2. Update APEX-Ranker config to use clean dataset"
        echo "   3. Start training with improved dataset"
        echo ""

        exit 0
    fi

    # Check build log for errors
    if tail -20 "$LOG_FILE" | grep -qi "error\|exception\|failed"; then
        echo "⚠️  WARNING: Potential errors detected in build log"
        echo "   Check: tail -100 $LOG_FILE"
        echo ""
    fi

    # Estimate remaining time
    local completed_count=0
    for year in 2022 2023 2024; do
        for q in Q1 Q2 Q3 Q4; do
            if is_chunk_complete "${year}${q}"; then
                completed_count=$((completed_count + 1))
            fi
        done
    done

    local pending_count=$((12 - completed_count))
    local est_minutes=$((pending_count * 15))  # ~15 min per chunk average

    echo "⏱️  Estimated remaining time: ~${est_minutes} minutes (${pending_count} chunks remaining)"
    echo ""
    echo "💡 Tip: This script will auto-execute Phase B' Step 2-6 when ready"
    echo "   Press Ctrl+C to stop monitoring (build will continue in background)"
    echo ""

    sleep 30
done
