#!/bin/bash
################################################################################
# ATFT Data Cache Cleanup Script
#
# Purpose: Remove output/atft_data/ to force recomputation of normalization
#          statistics and feature conversion with latest settings.
#
# Usage:
#   ./scripts/clean_atft_cache.sh          # Interactive (asks confirmation)
#   ./scripts/clean_atft_cache.sh --force  # Automatic (no confirmation)
#
# Safety: Shows contents before deletion and requires confirmation
################################################################################

set -e

echo "========================================"
echo "🗑️  ATFT Data Cache Cleanup"
echo "========================================"
echo ""

TARGET_DIR="output/atft_data"

# Check if directory exists
if [ ! -d "$TARGET_DIR" ]; then
  echo "ℹ️  Cache directory not found: $TARGET_DIR"
  echo "   (already clean or never generated)"
  exit 0
fi

# Show current contents
echo "📂 Current contents of $TARGET_DIR:"
ls -lh "$TARGET_DIR/" 2>/dev/null || echo "   (empty or inaccessible)"
echo ""

# Count files
TRAIN_COUNT=$(find "$TARGET_DIR/train" -type f 2>/dev/null | wc -l)
VAL_COUNT=$(find "$TARGET_DIR/val" -type f 2>/dev/null | wc -l)
TEST_COUNT=$(find "$TARGET_DIR/test" -type f 2>/dev/null | wc -l)

echo "📊 File counts:"
echo "   - train: $TRAIN_COUNT files"
echo "   - val:   $VAL_COUNT files"
echo "   - test:  $TEST_COUNT files"
echo ""

# Confirmation prompt (skip with --force)
if [ "$1" != "--force" ]; then
  echo "⚠️  This will delete ALL cached ATFT training data."
  echo "   Next training run will regenerate from ml_dataset parquet."
  echo ""
  read -p "Continue? (y/N): " -n 1 -r
  echo ""

  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Aborted by user"
    exit 1
  fi
fi

# Delete
echo "🗑️  Deleting $TARGET_DIR..."
rm -rf "$TARGET_DIR"

echo "✅ Cache cleared successfully"
echo ""
echo "📝 Next steps:"
echo "   1. Run project health check: ./tools/project-health-check.sh"
echo "   2. Start training to regenerate cache with current settings"
echo ""
