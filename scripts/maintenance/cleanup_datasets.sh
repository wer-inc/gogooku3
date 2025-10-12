#!/usr/bin/env bash
#
# Cleanup old ML dataset files to prevent storage bloat
#
# Usage:
#   bash scripts/maintenance/cleanup_datasets.sh [--keep N] [--dry-run] [--force]
#
# Options:
#   --keep N       Keep N latest generations of datasets (default: 1)
#   --dry-run      Show what would be deleted without actually deleting
#   --force        Skip confirmation prompt
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default parameters
KEEP_GENERATIONS=${DATASET_KEEP_GENERATIONS:-1}
DRY_RUN=0
FORCE=0
DATASETS_DIR="output/datasets"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --keep)
            KEEP_GENERATIONS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --force)
            FORCE=1
            shift
            ;;
        --help)
            echo "Usage: $0 [--keep N] [--dry-run] [--force]"
            echo ""
            echo "Options:"
            echo "  --keep N       Keep N latest generations of datasets (default: 1)"
            echo "  --dry-run      Show what would be deleted without actually deleting"
            echo "  --force        Skip confirmation prompt"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if datasets directory exists
if [[ ! -d "$DATASETS_DIR" ]]; then
    echo -e "${YELLOW}Datasets directory not found: $DATASETS_DIR${NC}"
    exit 0
fi

echo -e "${BLUE}=== ML Dataset Cleanup ===${NC}"
echo "Datasets directory: $DATASETS_DIR"
echo "Keep latest: $KEEP_GENERATIONS generation(s)"
[[ $DRY_RUN -eq 1 ]] && echo -e "${YELLOW}DRY RUN MODE - No files will be deleted${NC}"
echo ""

# Get current datasets size
CURRENT_SIZE=$(du -sh "$DATASETS_DIR" 2>/dev/null | cut -f1 || echo "0")
echo -e "Current datasets size: ${BLUE}$CURRENT_SIZE${NC}"
echo ""

# Find all timestamped datasets (exclude latest symlink)
# Pattern: ml_dataset_YYYYMMDD_HHMMSS_full.parquet
TIMESTAMPED_DATASETS=$(find "$DATASETS_DIR" -maxdepth 1 -type f -name 'ml_dataset_[0-9]*_full.parquet' 2>/dev/null | sort -r || true)

if [[ -z "$TIMESTAMPED_DATASETS" ]]; then
    echo -e "${GREEN}No timestamped datasets found${NC}"
    exit 0
fi

TOTAL_COUNT=$(echo "$TIMESTAMPED_DATASETS" | wc -l)
echo -e "Found ${BLUE}$TOTAL_COUNT${NC} timestamped dataset(s)"
echo ""

# Calculate how many to delete
DELETE_COUNT=$((TOTAL_COUNT - KEEP_GENERATIONS))

if [[ $DELETE_COUNT -le 0 ]]; then
    echo -e "${GREEN}Already keeping only $KEEP_GENERATIONS generation(s), no cleanup needed${NC}"

    # Check if latest symlink exists and is correct
    LATEST_LINK="$DATASETS_DIR/ml_dataset_latest_full.parquet"
    LATEST_DATASET=$(echo "$TIMESTAMPED_DATASETS" | head -1)

    if [[ -L "$LATEST_LINK" ]]; then
        CURRENT_TARGET=$(readlink "$LATEST_LINK")
        EXPECTED_TARGET=$(basename "$LATEST_DATASET")
        if [[ "$CURRENT_TARGET" != "$EXPECTED_TARGET" ]]; then
            echo -e "${YELLOW}Note: Latest symlink points to $CURRENT_TARGET, but newest is $EXPECTED_TARGET${NC}"
        fi
    elif [[ -f "$LATEST_LINK" ]]; then
        echo -e "${YELLOW}Note: ml_dataset_latest_full.parquet is a regular file, not a symlink${NC}"
        echo -e "      Consider converting it to a symlink to save space"
    fi

    exit 0
fi

# Get files to delete (keep the newest N, delete the rest)
FILES_TO_DELETE=$(echo "$TIMESTAMPED_DATASETS" | tail -n +$((KEEP_GENERATIONS + 1)))

echo -e "${YELLOW}Will delete $DELETE_COUNT old dataset(s):${NC}"
echo ""

# Calculate space to be freed and show files
TOTAL_FREED=0
echo "$FILES_TO_DELETE" | while read -r file; do
    if [[ -f "$file" ]]; then
        SIZE=$(du -h "$file" 2>/dev/null | cut -f1 || echo "?")
        SIZE_BYTES=$(stat -c%s "$file" 2>/dev/null || echo "0")
        TOTAL_FREED=$((TOTAL_FREED + SIZE_BYTES))
        BASENAME=$(basename "$file")
        MTIME=$(stat -c %y "$file" 2>/dev/null | cut -d' ' -f1 || echo "?")
        echo "  - $BASENAME ($SIZE, modified: $MTIME)"

        # Also check for associated metadata files
        METADATA_FILE="${file%.parquet}_metadata.json"
        if [[ -f "$METADATA_FILE" ]]; then
            META_SIZE=$(du -h "$METADATA_FILE" 2>/dev/null | cut -f1 || echo "?")
            echo "    + $(basename "$METADATA_FILE") ($META_SIZE)"
        fi
    fi
done

# Calculate total space to be freed
TOTAL_FREED_MB=0
if [[ -n "$FILES_TO_DELETE" ]]; then
    TOTAL_FREED_BYTES=$(echo "$FILES_TO_DELETE" | xargs du -cb 2>/dev/null | tail -1 | cut -f1 || echo "0")
    TOTAL_FREED_MB=$((TOTAL_FREED_BYTES / 1024 / 1024))
fi

echo ""
echo -e "Space to be freed: ${YELLOW}${TOTAL_FREED_MB}MB${NC}"
echo ""

# Files to keep
echo -e "${GREEN}Will keep (newest $KEEP_GENERATIONS):${NC}"
echo "$TIMESTAMPED_DATASETS" | head -n $KEEP_GENERATIONS | while read -r file; do
    BASENAME=$(basename "$file")
    SIZE=$(du -h "$file" 2>/dev/null | cut -f1 || echo "?")
    echo "  ✓ $BASENAME ($SIZE)"
done
echo ""

# Confirm deletion (unless --force or --dry-run)
if [[ $DRY_RUN -eq 0 && $FORCE -eq 0 ]]; then
    echo -e "${YELLOW}This will delete $DELETE_COUNT dataset(s) and free ~${TOTAL_FREED_MB}MB${NC}"
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted"
        exit 0
    fi
fi

# Delete old datasets
DELETED_COUNT=0
DELETED_METADATA_COUNT=0

echo -e "${BLUE}Deleting old datasets...${NC}"
echo "$FILES_TO_DELETE" | while read -r file; do
    if [[ -f "$file" ]]; then
        if [[ $DRY_RUN -eq 1 ]]; then
            echo "  [DRY RUN] Would delete: $(basename $file)"
        else
            if rm -f "$file" 2>/dev/null; then
                echo "  ✓ Deleted: $(basename $file)"
                ((DELETED_COUNT++)) || true
            fi
        fi

        # Delete associated metadata
        METADATA_FILE="${file%.parquet}_metadata.json"
        if [[ -f "$METADATA_FILE" ]]; then
            if [[ $DRY_RUN -eq 1 ]]; then
                echo "    [DRY RUN] Would delete: $(basename $METADATA_FILE)"
            else
                if rm -f "$METADATA_FILE" 2>/dev/null; then
                    ((DELETED_METADATA_COUNT++)) || true
                fi
            fi
        fi
    fi
done

echo ""

# Update or create latest symlink
LATEST_DATASET=$(echo "$TIMESTAMPED_DATASETS" | head -1)
LATEST_LINK="$DATASETS_DIR/ml_dataset_latest_full.parquet"
LATEST_LINK_METADATA="$DATASETS_DIR/ml_dataset_latest_full_metadata.json"

if [[ -n "$LATEST_DATASET" && -f "$LATEST_DATASET" ]]; then
    LATEST_BASENAME=$(basename "$LATEST_DATASET")
    LATEST_METADATA="${LATEST_DATASET%.parquet}_metadata.json"

    echo -e "${BLUE}Updating latest symlink...${NC}"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  [DRY RUN] Would link: ml_dataset_latest_full.parquet -> $LATEST_BASENAME"
    else
        # Remove old link/file if exists
        rm -f "$LATEST_LINK" 2>/dev/null || true
        rm -f "$LATEST_LINK_METADATA" 2>/dev/null || true

        # Create symlink
        cd "$DATASETS_DIR"
        ln -s "$LATEST_BASENAME" "ml_dataset_latest_full.parquet"

        if [[ -f "$LATEST_METADATA" ]]; then
            ln -s "$(basename $LATEST_METADATA)" "ml_dataset_latest_full_metadata.json"
        fi

        cd - > /dev/null

        echo "  ✓ Created symlink: ml_dataset_latest_full.parquet -> $LATEST_BASENAME"
    fi
fi

# Show final size
FINAL_SIZE=$(du -sh "$DATASETS_DIR" 2>/dev/null | cut -f1 || echo "0")
echo ""
echo -e "${GREEN}=== Cleanup Complete ===${NC}"
echo -e "Deleted: ${BLUE}$DELETE_COUNT${NC} dataset(s) + ${BLUE}$DELETED_METADATA_COUNT${NC} metadata file(s)"
echo -e "Current size: ${BLUE}$CURRENT_SIZE${NC} → ${BLUE}$FINAL_SIZE${NC}"
[[ $DRY_RUN -eq 1 ]] && echo -e "${YELLOW}(This was a dry run - no files were actually deleted)${NC}"
