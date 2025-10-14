#!/usr/bin/env bash
#
# Test Files Cleanup
# Organizes scattered test files into proper tests/ directory structure
#
# Usage:
#   bash scripts/maintenance/cleanup_test_files.sh [--dry-run]
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
echo "  🧹 Test Files Cleanup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [[ $DRY_RUN -eq 1 ]]; then
    echo "🔍 DRY-RUN MODE: No changes will be made"
    echo ""
fi

# Phase 1: Move root directory tests to tests/exploratory/
echo "📂 Phase 1: Moving root directory tests to tests/exploratory/..."

ROOT_EXPLORATORY_TESTS=(
    "test_data_loading.py"
    "test_date_filtering.py"
    "test_env_settings.py"
    "test_normalization.py"
    "test_phase2_dataloader.py"
    "test_phase2_simple.py"
    "test_phase2_verification.py"
)

for test in "${ROOT_EXPLORATORY_TESTS[@]}"; do
    do_mv "$test" "tests/exploratory/"
done

echo ""

# Phase 2: Move scripts/ integration tests to tests/integration/
echo "📂 Phase 2: Moving scripts/ integration tests to tests/integration/..."

INTEGRATION_TESTS=(
    "test_atft_training.py"
    "test_baseline_rankic.py"
    "test_cache_cpu_fallback.py"
    "test_direct_training.py"
    "test_earnings_events.py"
    "test_full_integration.py"
    "test_futures_integration.py"
    "test_graph_cache_effectiveness.py"
    "test_multi_horizon.py"
    "test_normalized_training.py"
    "test_optimization.py"
    "test_phase1_features.py"
    "test_phase2_features.py"
    "test_regime_moe.py"
    "train_simple_test.py"
)

for test in "${INTEGRATION_TESTS[@]}"; do
    do_mv "scripts/$test" "tests/integration/"
done

echo ""

# Phase 3: Move feature test to tests/unit/
echo "📂 Phase 3: Moving feature test to tests/unit/..."
do_mv "scripts/test_default_features.py" "tests/unit/"

echo ""

# Phase 4: Verify smoke_test.py location
echo "📂 Phase 4: Verifying smoke_test.py location..."
if [[ -f "scripts/smoke_test.py" ]]; then
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY-RUN] ✅ smoke_test.py is in correct location (scripts/)"
    else
        echo "✅ smoke_test.py is in correct location (scripts/)"
    fi
else
    echo "⚠️  smoke_test.py not found in scripts/ (may have been moved)"
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
    echo "  bash scripts/maintenance/cleanup_test_files.sh"
else
    echo "📊 Summary:"
    echo "  • Moved to tests/exploratory/: 7 files (from root)"
    echo "  • Moved to tests/integration/: 15 files (from scripts/)"
    echo "  • Moved to tests/unit/: 1 file (from scripts/)"
    echo "  • Kept in scripts/: smoke_test.py"
    echo ""
    echo "🔍 Verify clean directories:"
    echo "  ls -la test*.py"
    echo "  (Expected: No test*.py files in root)"
    echo ""
    echo "  find scripts/ -maxdepth 1 -name 'test*.py'"
    echo "  (Expected: No test*.py files in scripts/)"
    echo ""
    echo "💡 Run tests with pytest:"
    echo "  pytest tests/unit/              # Unit tests"
    echo "  pytest tests/integration/       # Integration tests"
    echo "  pytest tests/exploratory/       # Exploratory tests"
    echo "  python scripts/smoke_test.py    # Smoke test"
fi
echo ""
