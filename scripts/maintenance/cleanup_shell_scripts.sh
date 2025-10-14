#!/usr/bin/env bash
#
# Shell Scripts Cleanup
# Organizes scattered .sh files into proper directory structure
#
# Usage:
#   bash scripts/maintenance/cleanup_shell_scripts.sh [--dry-run]
#
# Options:
#   --dry-run    Show what would be done without making changes
#

set -euo pipefail

# Configuration
DRY_RUN=0
ARCHIVE_DIR="archive/shell_scripts_2025-10-13"

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

# Helper functions
do_mkdir() {
    local dir=$1
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY-RUN] mkdir -p $dir"
    else
        mkdir -p "$dir"
        echo "✅ Created directory: $dir"
    fi
}

do_mv() {
    local src=$1
    local dst=$2
    if [[ ! -f $src ]]; then
        echo "⚠️  File not found: $src (skipping)"
        return 0
    fi
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY-RUN] mv $src $dst"
    else
        mv "$src" "$dst"
        echo "✅ Moved: $src → $dst"
    fi
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🧹 Shell Scripts Cleanup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [[ $DRY_RUN -eq 1 ]]; then
    echo "🔍 DRY-RUN MODE: No changes will be made"
    echo ""
fi

# Phase 1: Create directory structure
echo "📦 Phase 1: Creating directory structure..."
do_mkdir "scripts/data"
do_mkdir "scripts/training"
do_mkdir "tools"
do_mkdir "$ARCHIVE_DIR"
echo ""

# Phase 2: Move scripts to proper locations
echo "📂 Phase 2: Moving scripts to proper locations..."

# Data generation scripts
do_mv "generate_sector_dataset.sh" "scripts/data/"

# Monitoring scripts
do_mv "monitor_training.sh" "scripts/monitoring/"

# Maintenance scripts
do_mv "organize_outputs.sh" "scripts/maintenance/"

# External tools
do_mv "codex-mcp.sh" "tools/"
do_mv "codex-mcp-max.sh" "tools/"

echo ""

# Phase 3: Archive redundant scripts
echo "🗂️  Phase 3: Archiving redundant scripts..."

# Training wrappers (redundant with Makefile targets)
TRAINING_SCRIPTS=(
    "train_optimized.sh"
    "train_improved.sh"
    "train_optimized_final.sh"
    "train_optimized_rankic.sh"
    "train_with_graph.sh"
    "run_optimized.sh"
    "run_production_training.sh"
    "run_stable_simple.sh"
    "run_stable_training.sh"
    "run_single_process.sh"
)

for script in "${TRAINING_SCRIPTS[@]}"; do
    do_mv "$script" "$ARCHIVE_DIR/"
done

# Testing scripts (redundant with Python scripts)
do_mv "smoke_test.sh" "$ARCHIVE_DIR/"
do_mv "test_student_t.sh" "$ARCHIVE_DIR/"
do_mv "test_student_t_direct.sh" "$ARCHIVE_DIR/"

# One-time fix scripts
do_mv "fix_zero_loss.sh" "$ARCHIVE_DIR/"

echo ""

# Phase 4: Create archive documentation
echo "📝 Phase 4: Creating archive documentation..."

ARCHIVE_README="$ARCHIVE_DIR/README.md"

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] Would create: $ARCHIVE_README"
else
    cat > "$ARCHIVE_README" <<'EOF'
# Archived Shell Scripts

**Date**: 2025-10-13
**Reason**: Project organization cleanup

## Why Archived

These scripts were archived because they have been superseded by:

1. **Makefile targets** (preferred entry point)
2. **Python scripts** in scripts/ directory
3. **Better organized shell scripts** in scripts/ subdirectories

## Migration Guide

### Training Commands

| Old Command | New Command |
|-------------|-------------|
| `./train_optimized.sh` | `make train-optimized` |
| `./train_improved.sh` | `make train-improved` |
| `./train_optimized_rankic.sh` | `make train-rankic-boost` |
| `./run_stable_training.sh` | `make train-stable` |
| `./run_production_training.sh` | `make train-stable` |
| `./train_with_graph.sh` | Use `--adv-graph-train` flag |

### Testing Commands

| Old Command | New Command |
|-------------|-------------|
| `./smoke_test.sh` | `python scripts/smoke_test.py --max-epochs 1` |
| `./test_student_t.sh` | Experimental, not needed |

### Monitoring Commands

| Old Command | New Command |
|-------------|-------------|
| `./monitor_training.sh` | `scripts/monitoring/monitor_training.sh` |

### Data Generation

| Old Command | New Command |
|-------------|-------------|
| `./generate_sector_dataset.sh` | `scripts/data/generate_sector_dataset.sh` |

### Maintenance

| Old Command | New Command |
|-------------|-------------|
| `./organize_outputs.sh` | `scripts/maintenance/organize_outputs.sh` |

## What's Better Now?

### Before Cleanup
- ❌ 19 .sh files in root directory
- ❌ Unclear which scripts are current
- ❌ Redundant with Makefile targets
- ❌ Difficult to find the right script

### After Cleanup
- ✅ Clean root directory
- ✅ Organized scripts/ structure
- ✅ Makefile as single entry point
- ✅ Easy to find scripts by purpose
- ✅ Clear separation: data/training/monitoring/maintenance

## If You Need These Scripts

If you absolutely need any of these scripts back:

1. **Check Makefile first**: `make help`
2. **Check scripts/ directory**: Organized by purpose
3. **Copy from this archive** if no alternative exists

Example:
```bash
# Copy from archive if needed
cp archive/shell_scripts_2025-10-13/train_optimized.sh ./
```

## Files in This Archive

### Training Wrappers (10 files)
- train_optimized.sh
- train_improved.sh
- train_optimized_final.sh
- train_optimized_rankic.sh
- train_with_graph.sh
- run_optimized.sh
- run_production_training.sh
- run_stable_simple.sh
- run_stable_training.sh
- run_single_process.sh

### Testing Scripts (3 files)
- smoke_test.sh
- test_student_t.sh
- test_student_t_direct.sh

### Debugging Scripts (1 file)
- fix_zero_loss.sh

**Total**: 14 archived files

## Related Documentation

- **SHELL_SCRIPTS_CLEANUP_PLAN.md** - Full cleanup plan and rationale
- **Makefile** - Current entry point for all commands
- **CLAUDE.md** - Project documentation
- **make help** - List all available commands
EOF
    echo "✅ Created: $ARCHIVE_README"
fi

echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ Cleanup Complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [[ $DRY_RUN -eq 1 ]]; then
    echo "🔍 This was a dry-run. No changes were made."
    echo "Run without --dry-run to apply changes:"
    echo "  bash scripts/maintenance/cleanup_shell_scripts.sh"
else
    echo "📊 Summary:"
    echo "  • Moved to scripts/data/: 1 file"
    echo "  • Moved to scripts/monitoring/: 1 file"
    echo "  • Moved to scripts/maintenance/: 1 file"
    echo "  • Moved to tools/: 2 files"
    echo "  • Archived: 14 files"
    echo ""
    echo "📂 Archive location: $ARCHIVE_DIR/"
    echo "📖 Archive documentation: $ARCHIVE_README"
    echo ""
    echo "🔍 Verify clean root directory:"
    echo "  ls -la *.sh"
    echo "  (Expected: No .sh files)"
    echo ""
    echo "💡 All commands now available via Makefile:"
    echo "  make help"
fi
echo ""
