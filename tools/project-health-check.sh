#!/bin/bash
# Project health check and issue detection for gogooku3
# Returns: JSON report of detected issues and recommendations

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Prefer project virtualenv Python if available so editable installs are detected.
if [[ -x "$PROJECT_ROOT/venv/bin/python" ]]; then
    export PATH="$PROJECT_ROOT/venv/bin:$PATH"
    PYTHON_BIN="$PROJECT_ROOT/venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON_BIN="$(command -v python3)"
else
    PYTHON_BIN=""
fi

# Colors for terminal output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

humanize_seconds() {
    local total=${1:-0}
    local hours=$((total / 3600))
    local minutes=$(((total % 3600) / 60))
    local seconds=$((total % 60))
    printf "%02d:%02d:%02d" "$hours" "$minutes" "$seconds"
}

# Thresholds (in seconds) used to classify long-running background jobs
TRAINING_LOG_STALE_THRESHOLD=600

# Initialize report arrays
CRITICAL_ISSUES=()
WARNINGS=()
RECOMMENDATIONS=()
SUCCESSES=()

echo "🔍 Running project health check..."
echo ""

# ============================================================================
# 1. ENVIRONMENT CHECKS
# ============================================================================
echo -e "${BLUE}[1/8] Environment validation...${NC}"

if [ ! -f ".env" ]; then
    CRITICAL_ISSUES+=("Missing .env file - copy from .env.example")
else
    SUCCESSES+=("✓ .env file exists")

    # Check critical environment variables
    if ! grep -q "JQUANTS_AUTH_EMAIL=" .env || ! grep -q "JQUANTS_AUTH_PASSWORD=" .env; then
        CRITICAL_ISSUES+=("JQuants API credentials not configured in .env")
    else
        SUCCESSES+=("✓ JQuants credentials configured")
    fi

    # Check cache settings
    if ! grep -q "USE_CACHE=1" .env; then
        WARNINGS+=("USE_CACHE not set to 1 - pipeline will be slow (45-60s waste per run)")
        RECOMMENDATIONS+=("Add 'USE_CACHE=1' to .env for 95% faster data fetching")
    else
        SUCCESSES+=("✓ Cache enabled (USE_CACHE=1)")
    fi

    # Check Phase 2 cache optimizations
    if ! grep -q "ENABLE_MULTI_CACHE=1" .env; then
        RECOMMENDATIONS+=("Enable Phase 2 multi-cache: ENABLE_MULTI_CACHE=1 in .env")
    fi
fi

# ============================================================================
# 2. DEPENDENCY CHECKS
# ============================================================================
echo -e "${BLUE}[2/8] Dependencies validation...${NC}"

if [ -z "$PYTHON_BIN" ]; then
    CRITICAL_ISSUES+=("Python 3 not found in PATH")
else
    PYTHON_VERSION=$("$PYTHON_BIN" --version | awk '{print $2}')
    PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
    PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
        WARNINGS+=("Python version $PYTHON_VERSION < 3.9 (may cause issues)")
    else
        SUCCESSES+=("✓ Python $PYTHON_VERSION")
    fi
fi

# Check if package is installed
if [ -n "$PYTHON_BIN" ] && ! "$PYTHON_BIN" -c "import gogooku3" 2>/dev/null; then
    CRITICAL_ISSUES+=("gogooku3 package not installed - run: pip install -e .")
elif [ -n "$PYTHON_BIN" ]; then
    VERSION=$("$PYTHON_BIN" -c "import gogooku3; print(gogooku3.__version__)" 2>/dev/null || echo "unknown")
    SUCCESSES+=("✓ gogooku3 v$VERSION installed")
fi

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "Error")
    if [[ "$GPU_INFO" != "Error" ]]; then
        SUCCESSES+=("✓ GPU detected: $GPU_INFO")
    else
        WARNINGS+=("nvidia-smi command failed - GPU may not be accessible")
    fi
else
    WARNINGS+=("No GPU detected (nvidia-smi not found) - training will be slow")
fi

# ============================================================================
# 3. DATA PIPELINE STATUS
# ============================================================================
echo -e "${BLUE}[3/8] Data pipeline status...${NC}"

#############################################
# Check for latest dataset (parquet or arrow)
#############################################

# Helper: compute age/status for a given file path
_report_dataset_file() {
    local path="$1"; local label="$2"
    local size ts now age_s age="recent"
    size=$(du -h "$path" 2>/dev/null | awk '{print $1}' || echo "unknown")
    ts=$(stat -c '%Y' "$path" 2>/dev/null || echo "0")
    now=$(date +%s)
    age_s=$((now - ts))
    if [ "$age_s" -gt 604800 ]; then # 7 days
        age="old"
    fi
    SUCCESSES+=("✓ ${label}: $size")
    if [ "$age" = "old" ]; then
        RECOMMENDATIONS+=("Dataset is >7 days old - consider rebuilding: make dataset-bg")
    fi
}

# Preferred: top-level symlink maintained by pipeline
if [ -L "output/ml_dataset_latest_full.parquet" ]; then
    DATASET_PATH=$(readlink -f "output/ml_dataset_latest_full.parquet")
    _report_dataset_file "$DATASET_PATH" "Dataset exists"
else
    # Accept any produced parquet in output/ (timestamped) as valid
    if ls output/ml_dataset_*.parquet 1> /dev/null 2>&1; then
        WARNINGS+=("Dataset exists but symlink broken - manual intervention needed")
    fi

    # Also accept alternative well-known outputs
    ALT_DATASET=""
    for cand in \
        "output/ml_dataset_phase2_enriched.parquet" \
        "output/datasets/ml_dataset_latest_full.parquet" \
        "output/datasets"/*ml_dataset*_full*.parquet; do
        if [ -f "$cand" ]; then ALT_DATASET="$cand"; break; fi
    done

    # Arrow cache (fast path) is a valid dataset input for training/eval
    ARROW_DATASET=""
    for a in \
        "output/ml_dataset_cached.arrow" \
        "output/ml_dataset_latest_full.arrow" \
        "output/datasets"/*ml_dataset*full*.arrow; do
        if [ -f "$a" ]; then ARROW_DATASET="$a"; break; fi
    done

    if [ -n "$ALT_DATASET" ]; then
        _report_dataset_file "$ALT_DATASET" "Dataset exists (alt parquet)"
    elif [ -n "$ARROW_DATASET" ]; then
        _report_dataset_file "$ARROW_DATASET" "Dataset exists (Arrow cache)"
    else
        CRITICAL_ISSUES+=("No ML dataset found - run: make dataset-bg")
    fi
fi

# Check cache status
if [ -d "output/raw/prices" ] && [ "$(ls -A output/raw/prices 2>/dev/null)" ]; then
    CACHE_SIZE=$(du -sh output/raw/prices 2>/dev/null | awk '{print $1}')
    CACHE_FILES=$(ls output/raw/prices/*.parquet 2>/dev/null | wc -l)
    SUCCESSES+=("✓ Price cache: $CACHE_FILES files ($CACHE_SIZE)")
else
    WARNINGS+=("No price cache found - first dataset build will be slower")
    RECOMMENDATIONS+=("After first dataset build, cache will save ~45-60s per run")
fi

# ============================================================================
# 4. TRAINING STATUS
# ============================================================================
echo -e "${BLUE}[4/8] Training status...${NC}"

# Check for running training processes
if TRAIN_PID=$(pgrep -f "train_atft.py" | head -1 2>/dev/null); then
    TRAIN_PID=${TRAIN_PID:-}
    if [ -n "$TRAIN_PID" ]; then
        read -r TRAIN_STATE TRAIN_ELAPSED TRAIN_CPU <<< "$(ps -p "$TRAIN_PID" -o state=,etimes=,pcpu= --no-headers 2>/dev/null || echo "unknown 0 0")"
        TRAIN_LOG=""
        if ls _logs/training/*.pid 1> /dev/null 2>&1; then
            for PID_FILE in _logs/training/*.pid; do
                if [ "$TRAIN_PID" = "$(cat "$PID_FILE" 2>/dev/null)" ]; then
                    CANDIDATE_LOG="${PID_FILE%.pid}.log"
                    if [ -f "$CANDIDATE_LOG" ]; then
                        TRAIN_LOG="$CANDIDATE_LOG"
                        break
                    fi
                fi
            done
        fi

        if [ -z "$TRAIN_LOG" ]; then
            if [ -f "logs/ml_training.log" ]; then
                TRAIN_LOG="logs/ml_training.log"
            elif [ -f "_logs/ml_training.log" ]; then
                TRAIN_LOG="_logs/ml_training.log"
            fi
        fi

        TRAIN_LOG_AGE=""
        if [ -n "$TRAIN_LOG" ]; then
            NOW_TS=$(date +%s)
            LOG_TS=$(stat -c %Y "$TRAIN_LOG" 2>/dev/null || echo "")
            if [ -n "$LOG_TS" ]; then
                TRAIN_LOG_AGE=$((NOW_TS - LOG_TS))
            fi
        fi

        if [ -n "$TRAIN_LOG_AGE" ] && [ "$TRAIN_LOG_AGE" -le "$TRAINING_LOG_STALE_THRESHOLD" ]; then
            SUCCESSES+=("✓ Training in progress (PID: $TRAIN_PID, state $TRAIN_STATE, elapsed $(humanize_seconds "$TRAIN_ELAPSED"), CPU ${TRAIN_CPU}%, log updated ${TRAIN_LOG_AGE}s ago: $(basename "$TRAIN_LOG"))")
        else
            if [ -n "$TRAIN_LOG_AGE" ]; then
                WARNINGS+=("Training process running (PID: $TRAIN_PID, state $TRAIN_STATE, elapsed $(humanize_seconds "$TRAIN_ELAPSED"), CPU ${TRAIN_CPU}%) - last log update ${TRAIN_LOG_AGE}s ago")
            else
                WARNINGS+=("Training process running (PID: $TRAIN_PID, state $TRAIN_STATE, elapsed $(humanize_seconds "$TRAIN_ELAPSED"), CPU ${TRAIN_CPU}%) - no associated log file found")
            fi
            RECOMMENDATIONS+=("Check training status: make train-status")
        fi
    fi
else
    SUCCESSES+=("✓ No training processes running")
fi

# Check for recent training logs
if ls _logs/training/*.log 1> /dev/null 2>&1; then
    LATEST_LOG=$(ls -t _logs/training/*.log 2>/dev/null | head -1)
    LOG_AGE=$(find "$LATEST_LOG" -mtime +1 2>/dev/null && echo "old" || echo "recent")

    if [ "$LOG_AGE" = "recent" ]; then
        SUCCESSES+=("✓ Recent training log found: $(basename $LATEST_LOG)")
    fi
fi

# Check for trained models (check multiple locations)
MODEL_COUNT=0
if ls models/*.pt 1> /dev/null 2>&1; then
    MODEL_COUNT=$((MODEL_COUNT + $(ls models/*.pt 2>/dev/null | wc -l)))
fi
if ls models/checkpoints/*.pt 1> /dev/null 2>&1; then
    MODEL_COUNT=$((MODEL_COUNT + $(ls models/checkpoints/*.pt 2>/dev/null | wc -l)))
fi
if ls output/models/*.pth 1> /dev/null 2>&1; then
    MODEL_COUNT=$((MODEL_COUNT + $(ls output/models/*.pth 2>/dev/null | wc -l)))
fi

if [ "$MODEL_COUNT" -gt 0 ]; then
    SUCCESSES+=("✓ $MODEL_COUNT trained models found")
elif [ -z "$TRAIN_PID" ]; then
    # Only recommend training if no training is currently running
    RECOMMENDATIONS+=("No trained models found - start training: make train-quick")
fi

# ============================================================================
# 5. CODE QUALITY CHECKS
# ============================================================================
echo -e "${BLUE}[5/8] Code quality checks...${NC}"

# Check for TODO/FIXME comments in critical files
TODO_COUNT=$(grep -r "TODO\|FIXME" src/ scripts/ --include="*.py" 2>/dev/null | wc -l || echo "0")
if [ "$TODO_COUNT" -gt 0 ]; then
    RECOMMENDATIONS+=("Found $TODO_COUNT TODO/FIXME comments - review and address")
fi

# Check if pre-commit is installed
if [ -d ".git/hooks" ] && [ -f ".git/hooks/pre-commit" ]; then
    SUCCESSES+=("✓ pre-commit hooks installed")
else
    WARNINGS+=("pre-commit hooks not installed - run: pre-commit install")
fi

# Check for uncommitted changes (allow documented WIP groups)
if [ -d ".git" ]; then
    if [ -n "${PYTHON_BIN:-}" ] && [ -f "configs/quality/worktree_allowlist.json" ] && [ -f "tools/quality/worktree_audit.py" ]; then
        if WORKTREE_REPORT=$("$PYTHON_BIN" tools/quality/worktree_audit.py --config configs/quality/worktree_allowlist.json 2>/dev/null); then
            AUDIT_EXIT=0
        else
            AUDIT_EXIT=$?
        fi

        if [ -n "$WORKTREE_REPORT" ] && command -v jq >/dev/null 2>&1; then
            TOTAL_CHANGES=$(echo "$WORKTREE_REPORT" | jq '.total // 0')
        else
            TOTAL_CHANGES=0
        fi

        if [ "${AUDIT_EXIT:-2}" -eq 0 ]; then
            if [ "${TOTAL_CHANGES:-0}" -eq 0 ]; then
                SUCCESSES+=("✓ Working directory clean")
            else
                if command -v jq >/dev/null 2>&1; then
                    SUMMARY=$(echo "$WORKTREE_REPORT" | jq -r '
                        (.matched // []) | map("\(.count)x \(.description)") | join("; ")
                    ' 2>/dev/null)
                else
                    SUMMARY=""
                fi
                if [ -z "$SUMMARY" ]; then
                    SUMMARY="Documented WIP change sets"
                fi
                RECOMMENDATIONS+=("Documented WIP change sets detected (${TOTAL_CHANGES} files): $SUMMARY")
            fi
        else
            if command -v jq >/dev/null 2>&1; then
                UNEXPECTED_LIST=$(echo "$WORKTREE_REPORT" | jq -r '
                    (.unexpected // []) | map("\(.status) \(.path)") | join("; ")
                ' 2>/dev/null)
            else
                UNEXPECTED_LIST=""
            fi
            if [ -z "$UNEXPECTED_LIST" ]; then
                UNEXPECTED_LIST="See git status for details"
            fi
            WARNINGS+=("Uncommitted changes outside allowlist: $UNEXPECTED_LIST")
        fi
    else
        UNCOMMITTED=$(git status --porcelain 2>/dev/null | wc -l || echo "0")
        if [ "$UNCOMMITTED" -gt 0 ]; then
            GIT_SUMMARY=$(git status --short 2>/dev/null | head -5 | tr '\n' ';' | sed 's/;$//' || echo "")
            if [ -n "$GIT_SUMMARY" ]; then
                WARNINGS+=("$UNCOMMITTED uncommitted changes in working directory (e.g. $GIT_SUMMARY)")
            else
                WARNINGS+=("$UNCOMMITTED uncommitted changes in working directory")
            fi
        else
            SUCCESSES+=("✓ Working directory clean")
        fi
    fi
fi

# ============================================================================
# 6. PERFORMANCE OPTIMIZATION STATUS
# ============================================================================
echo -e "${BLUE}[6/8] Performance optimization status...${NC}"

if [ -f ".env" ]; then
    # Check for multi-worker DataLoader
    if grep -q "ALLOW_UNSAFE_DATALOADER=1" .env; then
        SUCCESSES+=("✓ Multi-worker DataLoader enabled")
    else
        RECOMMENDATIONS+=("Enable multi-worker DataLoader: ALLOW_UNSAFE_DATALOADER=1 (2-3x GPU utilization)")
    fi

    # Check for torch.compile
    if grep -q "TORCH_COMPILE_MODE=" .env; then
        SUCCESSES+=("✓ torch.compile configured")
    else
        RECOMMENDATIONS+=("Enable torch.compile: TORCH_COMPILE_MODE=max-autotune (10-30% speedup)")
    fi

    # Check for RankIC loss
    if grep -q "USE_RANKIC=1" .env; then
        SUCCESSES+=("✓ RankIC loss enabled")
    else
        RECOMMENDATIONS+=("Enable RankIC loss: USE_RANKIC=1, RANKIC_WEIGHT=0.2 (better financial metrics)")
    fi
fi

# ============================================================================
# 7. DISK SPACE CHECK
# ============================================================================
echo -e "${BLUE}[7/8] Disk space check...${NC}"

AVAILABLE_GB=$(df -BG "$PROJECT_ROOT" | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_GB" -lt 50 ]; then
    WARNINGS+=("Low disk space: ${AVAILABLE_GB}GB available (recommend >50GB)")
    RECOMMENDATIONS+=("Clean old datasets: make dataset-clean")
else
    SUCCESSES+=("✓ Disk space: ${AVAILABLE_GB}GB available")
fi

# ============================================================================
# 8. CONFIGURATION VALIDATION
# ============================================================================
echo -e "${BLUE}[8/8] Configuration validation...${NC}"

# Check for required config files
REQUIRED_CONFIGS=(
    "configs/atft/config_production_optimized.yaml"
    "configs/atft/feature_categories.yaml"
    "Makefile.dataset"
    "Makefile.train"
)

for config in "${REQUIRED_CONFIGS[@]}"; do
    if [ -f "$config" ]; then
        SUCCESSES+=("✓ Config exists: $config")
    else
        WARNINGS+=("Missing config: $config")
    fi
done

# ============================================================================
# GENERATE REPORT
# ============================================================================
echo ""
echo "========================================================================"
echo "                      PROJECT HEALTH REPORT"
echo "========================================================================"
echo ""

# Critical Issues
if [ ${#CRITICAL_ISSUES[@]} -gt 0 ]; then
    echo -e "${RED}❌ CRITICAL ISSUES (${#CRITICAL_ISSUES[@]}):${NC}"
    for issue in "${CRITICAL_ISSUES[@]}"; do
        echo -e "  ${RED}✗${NC} $issue"
    done
    echo ""
fi

# Warnings
if [ ${#WARNINGS[@]} -gt 0 ]; then
    echo -e "${YELLOW}⚠️  WARNINGS (${#WARNINGS[@]}):${NC}"
    for warning in "${WARNINGS[@]}"; do
        echo -e "  ${YELLOW}⚠${NC} $warning"
    done
    echo ""
fi

# Recommendations
if [ ${#RECOMMENDATIONS[@]} -gt 0 ]; then
    echo -e "${BLUE}💡 RECOMMENDATIONS (${#RECOMMENDATIONS[@]}):${NC}"
    for rec in "${RECOMMENDATIONS[@]}"; do
        echo -e "  ${BLUE}→${NC} $rec"
    done
    echo ""
fi

# Successes (summary only)
if [ ${#SUCCESSES[@]} -gt 0 ]; then
    echo -e "${GREEN}✅ HEALTHY CHECKS: ${#SUCCESSES[@]}${NC}"
fi

echo ""
echo "========================================================================"

# Generate JSON report for Claude
JSON_REPORT=$(cat <<EOF
{
  "timestamp": "$(date -Iseconds)",
  "project_root": "$PROJECT_ROOT",
  "summary": {
    "critical_issues": ${#CRITICAL_ISSUES[@]},
    "warnings": ${#WARNINGS[@]},
    "recommendations": ${#RECOMMENDATIONS[@]},
    "healthy_checks": ${#SUCCESSES[@]}
  },
  "critical_issues": $(printf '%s\n' "${CRITICAL_ISSUES[@]}" | jq -R . | jq -s .),
  "warnings": $(printf '%s\n' "${WARNINGS[@]}" | jq -R . | jq -s .),
  "recommendations": $(printf '%s\n' "${RECOMMENDATIONS[@]}" | jq -R . | jq -s .),
  "successes": $(printf '%s\n' "${SUCCESSES[@]}" | jq -R . | jq -s .)
}
EOF
)

# Save report
mkdir -p _logs/health-checks
REPORT_FILE="_logs/health-checks/health-check-$(date +%Y%m%d-%H%M%S).json"
echo "$JSON_REPORT" > "$REPORT_FILE"

echo "📊 Full report saved to: $REPORT_FILE"
echo ""

# Exit code based on severity
if [ ${#CRITICAL_ISSUES[@]} -gt 0 ]; then
    exit 2  # Critical issues found
elif [ ${#WARNINGS[@]} -gt 0 ]; then
    exit 1  # Warnings found
else
    exit 0  # All healthy
fi
