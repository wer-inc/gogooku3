#!/bin/bash
# Dagster health check script
# Verifies that Dagster definitions can be loaded without ANTLR conflicts
#
# Usage:
#   ./tools/dagster-health-check.sh           # Quick check
#   ./tools/dagster-health-check.sh --verbose # Verbose output
#
# Cron setup (optional):
#   */5 * * * * /workspace/gogooku3/tools/dagster-health-check.sh >> /var/log/dagster-health.log 2>&1

set -euo pipefail

REPO_ROOT=/workspace/gogooku3
VERBOSE=false
CHECK_FAILED=false

# Parse arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Setup environment
export DAGSTER_HOME="$REPO_ROOT/gogooku5"
export PYTHONPATH="$REPO_ROOT/gogooku5/data/src"
export PYTHONWARNINGS="ignore::DeprecationWarning"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${GREEN}[INFO]${NC} $1"
    fi
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Check 1: Python availability
log_info "Checking Python availability..."
if ! command -v python &> /dev/null; then
    log_error "Python not found"
    exit 1
fi
log_info "✓ Python $(python --version 2>&1 | cut -d' ' -f2) found"

# Check 2: Dagster installation
log_info "Checking Dagster installation..."
if ! python -c "import dagster" 2>/dev/null; then
    log_error "Dagster not installed"
    exit 1
fi
DAGSTER_VERSION=$(python -c "import dagster; print(dagster.__version__)" 2>/dev/null)
log_info "✓ Dagster $DAGSTER_VERSION installed"

# Check 3: Can import definitions (CRITICAL - tests ANTLR conflict)
log_info "Checking Dagster definitions import..."
ASSET_COUNT=$(python - <<'EOF'
import sys
try:
    from dagster_gogooku5.defs import defs
    asset_count = len(defs.assets)
    print(asset_count)
    sys.exit(0)
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
EOF
)

if [ $? -ne 0 ]; then
    log_error "Failed to import Dagster definitions"
    log_error "This usually indicates an ANTLR version conflict"
    exit 1
fi

log_info "✓ Dagster definitions loaded ($ASSET_COUNT assets)"

# Check 4: Hydra compatibility
log_info "Checking Hydra compatibility..."
if ! python -c "import hydra; from omegaconf import OmegaConf" 2>/dev/null; then
    log_warn "Hydra/OmegaConf import failed (non-critical)"
else
    log_info "✓ Hydra/OmegaConf compatible"
fi

# Check 5: Storage directory
log_info "Checking Dagster storage..."
if [ ! -d "$DAGSTER_HOME/storage" ]; then
    log_warn "Storage directory not found: $DAGSTER_HOME/storage"
else
    RUN_COUNT=$(ls -1 "$DAGSTER_HOME/storage"/*.db 2>/dev/null | wc -l)
    log_info "✓ Storage directory exists ($RUN_COUNT run databases)"
fi

# Check 6: Recent activity
log_info "Checking recent activity..."
RECENT_RUNS=$(find "$DAGSTER_HOME/storage" -name "*.db" -mtime -1 2>/dev/null | wc -l)
if [ "$RECENT_RUNS" -gt 0 ]; then
    log_info "✓ Recent activity detected ($RECENT_RUNS runs in last 24h)"
else
    log_info "ℹ  No recent runs (last 24h)"
fi

# Check 7: Custom runner availability
log_info "Checking custom runner..."
RUNNER="$REPO_ROOT/gogooku5/data/tools/materialize_asset.py"
if [ ! -f "$RUNNER" ]; then
    log_error "Custom runner not found: $RUNNER"
    exit 1
fi
log_info "✓ Custom runner available"

# Check 8: Can execute runner help
log_info "Checking runner execution..."
if ! python "$RUNNER" --help &>/dev/null; then
    log_error "Failed to execute runner"
    exit 1
fi
log_info "✓ Runner executable"

# Check 9: Schema manifest availability
log_info "Checking schema manifest..."
SCHEMA_MANIFEST="$REPO_ROOT/gogooku5/data/schema/feature_schema_manifest.json"
if [ ! -f "$SCHEMA_MANIFEST" ]; then
    log_warn "Schema manifest not found: $SCHEMA_MANIFEST"
    SCHEMA_CHECK_AVAILABLE=false
else
    SCHEMA_VERSION=$(python -c "import json; print(json.load(open('$SCHEMA_MANIFEST'))['version'])" 2>/dev/null)
    SCHEMA_HASH=$(python -c "import json; print(json.load(open('$SCHEMA_MANIFEST'))['schema_hash'])" 2>/dev/null)
    log_info "✓ Schema manifest v$SCHEMA_VERSION (hash: $SCHEMA_HASH)"
    SCHEMA_CHECK_AVAILABLE=true
fi

# Check 10: Chunk schema validation
SCHEMA_MISMATCHES=0
if [ "$SCHEMA_CHECK_AVAILABLE" = true ]; then
    log_info "Validating chunk schemas..."
    CHUNKS_DIR="$REPO_ROOT/gogooku5/data/output/chunks"

    if [ ! -d "$CHUNKS_DIR" ]; then
        log_info "ℹ  No chunks directory found (no chunks to validate)"
    else
        CHUNK_COUNT=$(ls -1d "$CHUNKS_DIR"/*/ 2>/dev/null | wc -l)

        if [ "$CHUNK_COUNT" -eq 0 ]; then
            log_info "ℹ  No chunks found to validate"
        else
            log_info "Found $CHUNK_COUNT chunks to validate"

            # Run schema validation
            cd "$REPO_ROOT/gogooku5/data"
            VALIDATION_OUTPUT=$(python tools/check_chunks.py --validate-schema --no-fail-on-schema 2>&1)
            VALIDATION_EXIT=$?

            # Count schema mismatches
            SCHEMA_MISMATCHES=$(echo "$VALIDATION_OUTPUT" | grep -c "✗" || true)

            if [ "$SCHEMA_MISMATCHES" -gt 0 ]; then
                log_warn "Schema validation: $SCHEMA_MISMATCHES/$CHUNK_COUNT chunks have mismatches"
                CHECK_FAILED=true
                if [ "$VERBOSE" = true ]; then
                    echo "$VALIDATION_OUTPUT" | grep "✗" || true
                fi
            else
                log_info "✓ All $CHUNK_COUNT chunks have valid schemas"
            fi
        fi
    fi
fi

# Final status
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
if [ "$SCHEMA_CHECK_AVAILABLE" = true ] && [ "${SCHEMA_MISMATCHES:-0}" -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Dagster health check PASSED with warnings${NC}"
else
    echo -e "${GREEN}✅ Dagster health check PASSED${NC}"
fi
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Dagster:      $DAGSTER_VERSION"
echo "  Assets:       $ASSET_COUNT"
echo "  Recent runs:  $RECENT_RUNS (24h)"
echo "  Storage:      $DAGSTER_HOME/storage"

if [ "$SCHEMA_CHECK_AVAILABLE" = true ]; then
    echo "  Schema:       v$SCHEMA_VERSION ($SCHEMA_HASH)"
    if [ "${CHUNK_COUNT:-0}" -gt 0 ]; then
        if [ "${SCHEMA_MISMATCHES:-0}" -gt 0 ]; then
            echo "  Chunks:       $CHUNK_COUNT total, ${SCHEMA_MISMATCHES} with schema mismatches ⚠️"
        else
            echo "  Chunks:       $CHUNK_COUNT total, all schemas valid ✓"
        fi
    fi
fi

echo ""
echo "Ready to execute:"
echo "  ./scripts/dagster_run.sh production"
echo "  ./scripts/dagster_run.sh incremental"
echo "  ./scripts/dagster_run.sh merge"

if [ "${SCHEMA_MISMATCHES:-0}" -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}⚠️  Schema mismatches detected. Run for details:${NC}"
    echo "  cd $REPO_ROOT/gogooku5/data"
    echo "  python tools/check_chunks.py --validate-schema"
fi

echo ""

if [ "$CHECK_FAILED" = true ]; then
    exit 1
fi

exit 0
