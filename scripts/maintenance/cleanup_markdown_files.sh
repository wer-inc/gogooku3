#!/usr/bin/env bash
#
# Markdown Files Cleanup
# Organizes scattered .md files into proper docs/ directory structure
#
# Usage:
#   bash scripts/maintenance/cleanup_markdown_files.sh [--dry-run]
#
# Options:
#   --dry-run    Show what would be done without making changes
#

set -euo pipefail

# Configuration
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

# Helper functions
do_mkdir() {
    local dir=$1
    if [[ -d $dir ]]; then
        return 0  # Already exists, skip
    fi
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
echo "  🧹 Markdown Files Cleanup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [[ $DRY_RUN -eq 1 ]]; then
    echo "🔍 DRY-RUN MODE: No changes will be made"
    echo ""
fi

# Phase 1: Create necessary subdirectories
echo "📂 Phase 1: Creating directory structure..."
do_mkdir "docs/guides"
do_mkdir "docs/reports/completion"
do_mkdir "docs/reports/analysis"
do_mkdir "docs/reports/features"
echo ""

# Phase 2: Move guides to docs/guides/
echo "📚 Phase 2: Moving guides to docs/guides/..."
do_mv "QUICK_START.md" "docs/guides/quick_start.md"
do_mv "DATASET_GENERATION_GUIDE.md" "docs/guides/dataset_generation.md"
do_mv "GPU_TRAINING.md" "docs/guides/gpu_training.md"
do_mv "GPU_ETL_USAGE.md" "docs/guides/gpu_etl_usage.md"
do_mv "GPU_ML_PIPELINE.md" "docs/guides/gpu_ml_pipeline.md"
do_mv "manual.md" "docs/guides/manual.md"
do_mv "CODEX_MCP.md" "docs/guides/codex_mcp.md"
echo ""

# Phase 3: Move completion reports to docs/reports/completion/
echo "📋 Phase 3: Moving completion reports to docs/reports/completion/..."
do_mv "SETUP_IMPROVEMENTS.md" "docs/reports/completion/setup_improvements.md"
do_mv "SHELL_CLEANUP_COMPLETE.md" "docs/reports/completion/shell_cleanup_complete.md"
do_mv "SHELL_SCRIPTS_CLEANUP_PLAN.md" "docs/reports/completion/shell_scripts_cleanup_plan.md"
do_mv "TEST_CLEANUP_COMPLETE.md" "docs/reports/completion/test_cleanup_complete.md"
do_mv "TEST_FILES_CLEANUP_PLAN.md" "docs/reports/completion/test_files_cleanup_plan.md"
do_mv "CHANGES_SUMMARY.md" "docs/reports/completion/changes_summary.md"
do_mv "DOCS_REORGANIZATION_COMPLETE.md" "docs/reports/completion/docs_reorganization_complete.md"
do_mv "DOCS_REORGANIZATION_STATUS.md" "docs/reports/completion/docs_reorganization_status.md"
do_mv "ROOT_CAUSE_FIX_COMPLETE.md" "docs/reports/completion/root_cause_fix_complete.md"
do_mv "CACHE_FIX_DOCUMENTATION.md" "docs/reports/completion/cache_fix_documentation.md"
do_mv "FUTURES_INTEGRATION_COMPLETE.md" "docs/reports/completion/futures_integration_complete.md"
do_mv "SECTOR_SHORT_SELLING_INTEGRATION_COMPLETE.md" "docs/reports/completion/sector_short_selling_integration_complete.md"
do_mv "NK225_OPTION_INTEGRATION_STATUS.md" "docs/reports/completion/nk225_option_integration_status.md"
echo ""

# Phase 4: Move feature reports to docs/reports/features/
echo "✨ Phase 4: Moving feature reports to docs/reports/features/..."
do_mv "FEATURE_DEFAULTS_UPDATE.md" "docs/reports/features/feature_defaults_update.md"
echo ""

# Phase 5: Move architecture docs to docs/architecture/
echo "🏗️  Phase 5: Moving architecture docs to docs/architecture/..."
do_mv "ATFT‑GAT‑FAN_IMPL.md" "docs/architecture/atft_gat_fan_implementation.md"
do_mv "MIGRATION.md" "docs/architecture/migration.md"
do_mv "REFACTORING.md" "docs/architecture/refactoring.md"
echo ""

# Phase 6: Move analysis reports to docs/reports/analysis/
echo "📊 Phase 6: Moving analysis reports to docs/reports/analysis/..."
do_mv "OPTIMIZATION_REPORT_20251001.md" "docs/reports/analysis/optimization_report_20251001.md"
do_mv "EFFICIENCY_REPORT.md" "docs/reports/analysis/efficiency_report.md"
do_mv "TRAINING_IMPROVEMENTS.md" "docs/reports/analysis/training_improvements.md"
echo ""

# Phase 7: Move development docs to docs/development/
echo "🔧 Phase 7: Moving development docs to docs/development/..."
do_mv "AGENTS.md" "docs/development/agents.md"
do_mv "CLEAN_UP.md" "docs/development/clean_up.md"
do_mv "github_issues.md" "docs/development/github_issues.md"
do_mv "memories.md" "docs/development/memories.md"
echo ""

# Phase 8: Create README files for new directories
echo "📝 Phase 8: Creating README files..."

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] Would create docs/guides/README.md"
    echo "[DRY-RUN] Would create docs/reports/completion/README.md"
else
    # Create guides README
    if [[ ! -f "docs/guides/README.md" ]]; then
        cat > docs/guides/README.md <<'EOF'
# Guides

User-facing guides and tutorials for the gogooku3 project.

## Available Guides

- [Quick Start](quick_start.md) - Get started quickly
- [Dataset Generation](dataset_generation.md) - How to generate datasets
- [GPU Training](gpu_training.md) - GPU training guide
- [GPU ETL Usage](gpu_etl_usage.md) - GPU-accelerated ETL
- [GPU ML Pipeline](gpu_ml_pipeline.md) - GPU ML pipeline guide
- [Manual](manual.md) - Complete manual
- [Codex MCP](codex_mcp.md) - Codex MCP integration

## See Also

- [Main Documentation Index](../INDEX.md)
- [Architecture Documentation](../architecture/)
- [Reports](../reports/)
EOF
        echo "✅ Created: docs/guides/README.md"
    fi

    # Create completion reports README
    if [[ ! -f "docs/reports/completion/README.md" ]]; then
        cat > docs/reports/completion/README.md <<'EOF'
# Completion Reports

Historical completion and status reports for the gogooku3 project.

These reports document completed features, improvements, and major changes.

## Recent Reports (2025-10-13)

- [Setup Improvements](setup_improvements.md) - Make setup perfection
- [Shell Scripts Cleanup](shell_cleanup_complete.md) - Shell scripts organization
- [Test Files Cleanup](test_cleanup_complete.md) - Test files organization
- [Changes Summary](changes_summary.md) - Summary of all changes

## Feature Integration Reports

- [Futures Integration](futures_integration_complete.md)
- [Sector Short Selling Integration](sector_short_selling_integration_complete.md)
- [NK225 Option Integration Status](nk225_option_integration_status.md)

## System Improvements

- [Cache Fix Documentation](cache_fix_documentation.md)
- [Root Cause Fix Complete](root_cause_fix_complete.md)
- [Docs Reorganization](docs_reorganization_complete.md)
- [Feature Defaults Update](../features/feature_defaults_update.md)

## See Also

- [Analysis Reports](../analysis/)
- [Main Documentation Index](../../INDEX.md)
EOF
        echo "✅ Created: docs/reports/completion/README.md"
    fi
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
    echo "  bash scripts/maintenance/cleanup_markdown_files.sh"
else
    echo "📊 Summary:"
    echo "  • Guides: 7 files → docs/guides/"
    echo "  • Completion reports: 13 files → docs/reports/completion/"
    echo "  • Feature reports: 1 file → docs/reports/features/"
    echo "  • Architecture: 3 files → docs/architecture/"
    echo "  • Analysis: 3 files → docs/reports/analysis/"
    echo "  • Development: 4 files → docs/development/"
    echo "  • Total moved: 31 files"
    echo "  • Kept in root: 4 files (README.md, CLAUDE.md, CHANGELOG.md, TODO.md)"
    echo ""
    echo "🔍 Verify clean root directory:"
    echo "  ls -1 *.md"
    echo "  (Expected: 4 files only)"
    echo ""
    echo "💡 Documentation now organized in docs/"
    echo "  docs/guides/         - User guides"
    echo "  docs/reports/        - Reports (completion, analysis, features)"
    echo "  docs/architecture/   - Architecture documentation"
    echo "  docs/development/    - Development notes"
fi
echo ""
