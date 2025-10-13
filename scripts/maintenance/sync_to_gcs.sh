#!/usr/bin/env bash
#
# GCS Sync Script
# Synchronize local output/ directory to Google Cloud Storage
#
# Usage:
#   bash scripts/maintenance/sync_to_gcs.sh [--dry-run]
#
# Options:
#   --dry-run    Show what would be synced without making changes
#
# Features:
#   - Excludes output/raw/ (local cache only)
#   - Excludes symlinks (prevents duplication)
#   - Deletes remote files not present locally (-d flag)
#   - Uses parallel transfers (-m flag)
#

set -euo pipefail

# Configuration
GCS_BUCKET="${GCS_BUCKET:-gogooku-ml-data}"
LOCAL_DIR="output"
DRY_RUN=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run]"
            exit 1
            ;;
    esac
done

# Check if gsutil is available
if ! command -v gsutil &> /dev/null; then
    # Try to use the one we installed
    if [[ -x "google-cloud-sdk/bin/gsutil" ]]; then
        export PATH="$PATH:$(pwd)/google-cloud-sdk/bin"
    else
        echo "❌ Error: gsutil not found"
        echo "Please run 'make setup' to install Google Cloud SDK"
        exit 1
    fi
fi

# Check if GCS is enabled
if [[ "${GCS_ENABLED:-0}" != "1" ]]; then
    echo "⚠️  GCS_ENABLED is not set to 1 in .env"
    echo "Skipping GCS sync"
    exit 0
fi

# Verify credentials
if [[ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]]; then
    echo "⚠️  GOOGLE_APPLICATION_CREDENTIALS not set"
    echo "GCS authentication may fail"
fi

echo "🔄 GCS Sync Configuration:"
echo "  Source: ${LOCAL_DIR}/"
echo "  Destination: gs://${GCS_BUCKET}/output/"
echo "  Dry-run: $([ $DRY_RUN -eq 1 ] && echo 'Yes' || echo 'No')"
echo

# Build rsync command
RSYNC_CMD=(
    gsutil
    -m  # Parallel transfers
    rsync
    -r  # Recursive
    -d  # Delete remote files not in source
    -x "^raw/|.*ml_dataset_latest_full\.parquet$"  # Exclude raw/ (local cache) and symlinks
)

# Add dry-run flag if requested
if [[ $DRY_RUN -eq 1 ]]; then
    RSYNC_CMD+=(-n)
    echo "🔍 Dry-run mode: showing changes without executing"
    echo
fi

# Add source and destination
RSYNC_CMD+=("${LOCAL_DIR}/" "gs://${GCS_BUCKET}/output/")

# Execute rsync
echo "📤 Executing: ${RSYNC_CMD[*]}"
echo

"${RSYNC_CMD[@]}"

EXIT_CODE=$?

if [[ $EXIT_CODE -eq 0 ]]; then
    if [[ $DRY_RUN -eq 1 ]]; then
        echo
        echo "✅ Dry-run completed successfully"
        echo "Run without --dry-run to apply changes"
    else
        echo
        echo "✅ GCS sync completed successfully"

        # Show final storage usage
        echo
        echo "📊 GCS Storage Summary:"
        gsutil du -sh "gs://${GCS_BUCKET}/output/" 2>/dev/null || echo "  (unable to get size)"
    fi
else
    echo
    echo "❌ GCS sync failed with exit code: $EXIT_CODE"
    exit $EXIT_CODE
fi
