#!/usr/bin/env bash
#
# Cleanup old graph cache files to prevent memory bloat
#
# Usage:
#   bash scripts/maintenance/cleanup_cache.sh [--days N] [--max-size SIZE] [--dry-run] [--force]
#
# Options:
#   --days N         Delete cache older than N days (default: 30)
#   --max-size SIZE  Keep total cache under SIZE (e.g., 500M, 1G) (default: 500M)
#   --dry-run        Show what would be deleted without actually deleting
#   --force          Skip confirmation prompt
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default parameters
DAYS=${CACHE_CLEANUP_DAYS:-30}
MAX_SIZE=${CACHE_MAX_SIZE:-500M}
DRY_RUN=0
FORCE=0
CACHE_DIR="output/graph_cache"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --days)
            DAYS="$2"
            shift 2
            ;;
        --max-size)
            MAX_SIZE="$2"
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
            echo "Usage: $0 [--days N] [--max-size SIZE] [--dry-run] [--force]"
            echo ""
            echo "Options:"
            echo "  --days N         Delete cache older than N days (default: 30)"
            echo "  --max-size SIZE  Keep total cache under SIZE (e.g., 500M, 1G) (default: 500M)"
            echo "  --dry-run        Show what would be deleted without actually deleting"
            echo "  --force          Skip confirmation prompt"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if cache directory exists
if [[ ! -d "$CACHE_DIR" ]]; then
    echo -e "${YELLOW}Cache directory not found: $CACHE_DIR${NC}"
    exit 0
fi

echo -e "${BLUE}=== Graph Cache Cleanup ===${NC}"
echo "Cache directory: $CACHE_DIR"
echo "Age threshold: $DAYS days"
echo "Size limit: $MAX_SIZE"
[[ $DRY_RUN -eq 1 ]] && echo -e "${YELLOW}DRY RUN MODE - No files will be deleted${NC}"
echo ""

# Get current cache size
CURRENT_SIZE=$(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1 || echo "0")
echo -e "Current cache size: ${BLUE}$CURRENT_SIZE${NC}"
echo ""

# Find old files
echo -e "${BLUE}Finding files older than $DAYS days...${NC}"
OLD_FILES=$(find "$CACHE_DIR" -type f -mtime +$DAYS 2>/dev/null || true)
if [[ -n "$OLD_FILES" ]]; then
    OLD_COUNT=$(echo "$OLD_FILES" | wc -l)
else
    OLD_COUNT=0
fi

if [[ $OLD_COUNT -eq 0 ]]; then
    echo -e "${GREEN}No files older than $DAYS days found${NC}"
else
    echo -e "Found ${YELLOW}$OLD_COUNT${NC} files older than $DAYS days"

    # Calculate space to be freed
    OLD_SIZE=0
    if [[ -n "$OLD_FILES" ]]; then
        OLD_SIZE=$(echo "$OLD_FILES" | xargs du -ch 2>/dev/null | tail -1 | cut -f1 || echo "0")
    fi
    echo -e "Space to be freed: ${YELLOW}$OLD_SIZE${NC}"
    echo ""

    # Show sample of files to be deleted
    echo "Sample files to be deleted:"
    echo "$OLD_FILES" | head -10 | while read -r file; do
        SIZE=$(du -h "$file" 2>/dev/null | cut -f1 || echo "?")
        MTIME=$(stat -c %y "$file" 2>/dev/null | cut -d' ' -f1 || echo "?")
        echo "  - $file ($SIZE, modified: $MTIME)"
    done
    [[ $OLD_COUNT -gt 10 ]] && echo "  ... and $(($OLD_COUNT - 10)) more files"
    echo ""
fi

# Check if we need to delete by size
NEED_SIZE_CLEANUP=0
MAX_SIZE_BYTES=0

# Convert MAX_SIZE to bytes
case $MAX_SIZE in
    *M)
        MAX_SIZE_BYTES=$((${MAX_SIZE%M} * 1024 * 1024))
        ;;
    *G)
        MAX_SIZE_BYTES=$((${MAX_SIZE%G} * 1024 * 1024 * 1024))
        ;;
    *)
        echo -e "${RED}Invalid size format: $MAX_SIZE${NC}"
        exit 1
        ;;
esac

CURRENT_SIZE_BYTES=$(du -sb "$CACHE_DIR" 2>/dev/null | cut -f1 || echo "0")

if [[ $CURRENT_SIZE_BYTES -gt $MAX_SIZE_BYTES ]]; then
    NEED_SIZE_CLEANUP=1
    EXCESS_BYTES=$((CURRENT_SIZE_BYTES - MAX_SIZE_BYTES))
    EXCESS_MB=$((EXCESS_BYTES / 1024 / 1024))
    echo -e "${YELLOW}Cache size exceeds limit by ${EXCESS_MB}MB${NC}"
    echo "Will delete oldest files to reach size limit..."
    echo ""
fi

# Exit if nothing to do
if [[ $OLD_COUNT -eq 0 && $NEED_SIZE_CLEANUP -eq 0 ]]; then
    echo -e "${GREEN}Cache is clean, no action needed${NC}"
    exit 0
fi

# Confirm deletion (unless --force or --dry-run)
if [[ $DRY_RUN -eq 0 && $FORCE -eq 0 ]]; then
    echo -e "${YELLOW}This will delete $OLD_COUNT files and potentially free $OLD_SIZE${NC}"
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted"
        exit 0
    fi
fi

# Delete old files
DELETED_COUNT=0
if [[ $OLD_COUNT -gt 0 ]]; then
    echo -e "${BLUE}Deleting old files...${NC}"
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "$OLD_FILES" | while read -r file; do
            echo "  [DRY RUN] Would delete: $file"
        done
        DELETED_COUNT=$OLD_COUNT
    else
        echo "$OLD_FILES" | while read -r file; do
            if rm -f "$file" 2>/dev/null; then
                ((DELETED_COUNT++)) || true
            fi
        done
    fi
    echo -e "${GREEN}Deleted $DELETED_COUNT old files${NC}"
fi

# Delete by size if needed
if [[ $NEED_SIZE_CLEANUP -eq 1 ]]; then
    echo -e "${BLUE}Cleaning up by size...${NC}"

    # Find all files sorted by modification time (oldest first)
    ALL_FILES=$(find "$CACHE_DIR" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | cut -d' ' -f2- || true)

    CURRENT_BYTES=$(du -sb "$CACHE_DIR" 2>/dev/null | cut -f1 || echo "0")
    SIZE_DELETED_COUNT=0

    echo "$ALL_FILES" | while read -r file; do
        if [[ $CURRENT_BYTES -le $MAX_SIZE_BYTES ]]; then
            break
        fi

        FILE_SIZE=$(stat -c%s "$file" 2>/dev/null || echo "0")

        if [[ $DRY_RUN -eq 1 ]]; then
            echo "  [DRY RUN] Would delete: $file ($(($FILE_SIZE / 1024))KB)"
        else
            if rm -f "$file" 2>/dev/null; then
                CURRENT_BYTES=$((CURRENT_BYTES - FILE_SIZE))
                ((SIZE_DELETED_COUNT++)) || true
            fi
        fi
    done

    [[ $SIZE_DELETED_COUNT -gt 0 ]] && echo -e "${GREEN}Deleted $SIZE_DELETED_COUNT files to meet size limit${NC}"
fi

# Clean up empty directories
echo -e "${BLUE}Cleaning up empty directories...${NC}"
if [[ $DRY_RUN -eq 1 ]]; then
    find "$CACHE_DIR" -type d -empty 2>/dev/null | while read -r dir; do
        echo "  [DRY RUN] Would delete: $dir"
    done
else
    find "$CACHE_DIR" -type d -empty -delete 2>/dev/null || true
fi

# Show final size
FINAL_SIZE=$(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1 || echo "0")
echo ""
echo -e "${GREEN}=== Cleanup Complete ===${NC}"
echo -e "Final cache size: ${BLUE}$FINAL_SIZE${NC}"
[[ $DRY_RUN -eq 1 ]] && echo -e "${YELLOW}(This was a dry run - no files were actually deleted)${NC}"
