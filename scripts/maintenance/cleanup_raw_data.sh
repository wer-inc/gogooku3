#!/usr/bin/env bash
#
# Cleanup duplicate raw data files by keeping only the widest date range
#
# Usage:
#   bash scripts/maintenance/cleanup_raw_data.sh [--dry-run] [--force]
#
# Options:
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
DRY_RUN=0
FORCE=0
RAW_DIR="output/raw"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --force)
            FORCE=1
            shift
            ;;
        --help)
            echo "Usage: $0 [--dry-run] [--force]"
            echo ""
            echo "Options:"
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

# Check if raw directory exists
if [[ ! -d "$RAW_DIR" ]]; then
    echo -e "${YELLOW}Raw data directory not found: $RAW_DIR${NC}"
    exit 0
fi

echo -e "${BLUE}=== Raw Data Cleanup ===${NC}"
echo "Raw data directory: $RAW_DIR"
[[ $DRY_RUN -eq 1 ]] && echo -e "${YELLOW}DRY RUN MODE - No files will be deleted${NC}"
echo ""

# Get current raw data size
CURRENT_SIZE=$(du -sh "$RAW_DIR" 2>/dev/null | cut -f1 || echo "0")
echo -e "Current raw data size: ${BLUE}$CURRENT_SIZE${NC}"
echo ""

# Function to convert date string (YYYYMMDD) to Unix timestamp
date_to_timestamp() {
    local date_str=$1
    # Insert hyphens: YYYYMMDD -> YYYY-MM-DD
    local formatted_date="${date_str:0:4}-${date_str:4:2}-${date_str:6:2}"
    date -d "$formatted_date" +%s 2>/dev/null || echo "0"
}

# Function to calculate date range span in days
calculate_span() {
    local start_date=$1
    local end_date=$2
    local start_ts=$(date_to_timestamp "$start_date")
    local end_ts=$(date_to_timestamp "$end_date")
    echo $(( (end_ts - start_ts) / 86400 ))
}

# Track files to delete
declare -A files_to_delete
declare -A files_to_keep
total_files_to_delete=0
total_space_to_free=0

# Process each subdirectory
for subdir in "$RAW_DIR"/*; do
    if [[ ! -d "$subdir" ]]; then
        continue
    fi

    subdir_name=$(basename "$subdir")
    echo -e "${BLUE}Processing $subdir_name/${NC}"

    # Group files by feature type (everything before the first date)
    declare -A feature_groups

    # Enable nullglob for this iteration
    shopt -s nullglob
    for file in "$subdir"/*.parquet; do
        if [[ ! -f "$file" ]]; then
            continue
        fi

        filename=$(basename "$file" .parquet)

        # Extract feature type and dates using regex
        # Pattern: {feature_type}_{start_date}_{end_date}
        # Dates are 8 digits: YYYYMMDD
        if [[ $filename =~ ^(.+)_([0-9]{8})_([0-9]{8})$ ]]; then
            feature_type="${BASH_REMATCH[1]}"
            start_date="${BASH_REMATCH[2]}"
            end_date="${BASH_REMATCH[3]}"

            # Add to feature group
            if [[ -z "${feature_groups[$feature_type]:-}" ]]; then
                feature_groups[$feature_type]="$file"
            else
                feature_groups[$feature_type]="${feature_groups[$feature_type]}|$file"
            fi
        fi
    done

    # For each feature type, find the widest date range
    for feature_type in "${!feature_groups[@]}"; do
        files="${feature_groups[$feature_type]}"
        IFS='|' read -ra file_array <<< "$files"

        # Skip if only one file for this feature type
        if [[ ${#file_array[@]} -le 1 ]]; then
            continue
        fi

        echo "  Feature: $feature_type (${#file_array[@]} files)"

        # Find file with widest date range
        widest_file=""
        widest_span=0

        for file in "${file_array[@]}"; do
            filename=$(basename "$file" .parquet)
            if [[ $filename =~ ^.+_([0-9]{8})_([0-9]{8})$ ]]; then
                start_date="${BASH_REMATCH[1]}"
                end_date="${BASH_REMATCH[2]}"
                span=$(calculate_span "$start_date" "$end_date")

                if [[ $span -gt $widest_span ]]; then
                    widest_span=$span
                    widest_file="$file"
                fi
            fi
        done

        # Mark files for deletion (all except the widest)
        for file in "${file_array[@]}"; do
            if [[ "$file" != "$widest_file" ]]; then
                filename=$(basename "$file")
                file_size=$(stat -c%s "$file" 2>/dev/null || echo "0")
                file_size_mb=$((file_size / 1024 / 1024))

                files_to_delete["$file"]="$file_size"
                total_space_to_free=$((total_space_to_free + file_size))
                ((total_files_to_delete++)) || true

                # Extract dates for display
                basename_file=$(basename "$file" .parquet)
                if [[ $basename_file =~ _([0-9]{8})_([0-9]{8})$ ]]; then
                    echo -e "    ${RED}✗${NC} $filename (${file_size_mb}MB, ${BASH_REMATCH[1]}-${BASH_REMATCH[2]})"
                fi
            else
                files_to_keep["$file"]="1"
                filename=$(basename "$file")

                # Extract dates for display
                basename_file=$(basename "$file" .parquet)
                if [[ $basename_file =~ _([0-9]{8})_([0-9]{8})$ ]]; then
                    file_size=$(stat -c%s "$file" 2>/dev/null || echo "0")
                    file_size_mb=$((file_size / 1024 / 1024))
                    echo -e "    ${GREEN}✓${NC} $filename (${file_size_mb}MB, ${BASH_REMATCH[1]}-${BASH_REMATCH[2]}) [KEEP]"
                fi
            fi
        done
    done
    shopt -u nullglob

    echo ""
done

# Summary
total_space_to_free_mb=$((total_space_to_free / 1024 / 1024))

if [[ $total_files_to_delete -eq 0 ]]; then
    echo -e "${GREEN}No duplicate files found, raw data is clean${NC}"
    exit 0
fi

echo -e "${YELLOW}=== Summary ===${NC}"
echo -e "Files to delete: ${RED}$total_files_to_delete${NC}"
echo -e "Space to free: ${YELLOW}${total_space_to_free_mb}MB${NC}"
echo ""

# Confirm deletion (unless --force or --dry-run)
if [[ $DRY_RUN -eq 0 && $FORCE -eq 0 ]]; then
    echo -e "${YELLOW}This will delete $total_files_to_delete file(s) and free ~${total_space_to_free_mb}MB${NC}"
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted"
        exit 0
    fi
fi

# Delete files
DELETED_COUNT=0

echo -e "${BLUE}Deleting duplicate files...${NC}"
for file in "${!files_to_delete[@]}"; do
    if [[ -f "$file" ]]; then
        if [[ $DRY_RUN -eq 1 ]]; then
            echo "  [DRY RUN] Would delete: $file"
        else
            if rm -f "$file" 2>/dev/null; then
                echo "  ✓ Deleted: $(basename $file)"
                ((DELETED_COUNT++)) || true
            fi
        fi
    fi
done

# Show final size
FINAL_SIZE=$(du -sh "$RAW_DIR" 2>/dev/null | cut -f1 || echo "0")
echo ""
echo -e "${GREEN}=== Cleanup Complete ===${NC}"
echo -e "Deleted: ${BLUE}$DELETED_COUNT${NC} file(s)"
echo -e "Size: ${BLUE}$CURRENT_SIZE${NC} → ${BLUE}$FINAL_SIZE${NC}"
[[ $DRY_RUN -eq 1 ]] && echo -e "${YELLOW}(This was a dry run - no files were actually deleted)${NC}"
