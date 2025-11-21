#!/bin/bash
# Ultimate Codex CLI Launcher v2.0 (gogooku5)
# Based on OpenAI Codex CLI Official Documentation (2025)
# Repository: https://github.com/openai/codex
#
# Usage:
#   ./codex.sh                          # Interactive mode (clean terminal, no restrictions)
#   ./codex.sh --no-check               # Skip health check
#   ./codex.sh --quick                  # Quick mode (GPT-5, medium reasoning)
#   ./codex.sh --max                    # Maximum power (GPT-5-Codex, high reasoning)
#   ./codex.sh --exec <prompt>          # Non-interactive exec mode only
#   ./codex.sh --sandbox                # Enable sandbox (override default)
#   ./codex.sh --enable-sandbox         # Alias for --sandbox
#   ./codex.sh --term-workaround        # Enable terminal workaround (Issue #4960)
#
# Notes:
#   - Sandbox is DISABLED by default (full system access).
#   - Terminal workaround (Issue #4960) is DISABLED by default.
#   - Use --term-workaround if you see OSC sequences in output.
#
# FEATURES:
#   ✅ Dynamic environment detection (GPU, CPU, Memory, CUDA)
#   ✅ Automatic project health diagnosis
#   ✅ Smart model selection (GPT-5, GPT-5-Codex, O3)
#   ✅ Configurable reasoning levels (low/medium/high)
#   ✅ Multiple approval modes (suggest/auto/full-auto)
#   ✅ MCP servers integration
#   ✅ Session logging and debugging
#   ✅ AGENTS.md instructions support
#   ✅ Git integration awareness
#   ✅ Interactive mode by default

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="${PROJECT_ROOT}/logs/codex-sessions"
mkdir -p "$LOG_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ============================================================================
# PREREQUISITE CHECKS
# ============================================================================

echo -e "${CYAN}🔍 OpenAI Codex CLI Ultimate Launcher v2.0 (gogooku5)${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check codex installation
if ! command -v codex &> /dev/null; then
    echo -e "${RED}ERROR: codex is not installed.${NC}" >&2
    echo ""
    echo "Install with: npm install -g @openai/codex"
    echo "Or upgrade:   codex --upgrade"
    exit 127
fi

CODEX_VERSION=$(codex --version 2>/dev/null || echo "unknown")
echo -e "${GREEN}✓${NC} Codex CLI installed: ${CODEX_VERSION}"

# Check Git (recommended but not required)
if command -v git &> /dev/null; then
    GIT_VERSION=$(git --version | cut -d' ' -f3)
    echo -e "${GREEN}✓${NC} Git installed: ${GIT_VERSION}"
else
    echo -e "${YELLOW}⚠${NC} Git not found (recommended for full features)"
fi

echo ""

# ============================================================================
# SETUP MCP CONFIGURATION
# ============================================================================

if [ ! -f ".mcp.json" ]; then
    echo -e "${BLUE}📦 Creating MCP configuration...${NC}"
    cat > .mcp.json <<JSON
{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": ["-y", "@playwright/mcp@latest"]
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem@latest", "${PROJECT_ROOT}"]
    },
    "git": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-git@latest", "--repository", "."]
    },
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search@latest"]
    }
  }
}
JSON
    echo -e "${GREEN}✓${NC} Created .mcp.json"
    echo ""
fi

# ============================================================================
# SETUP AGENTS.MD INSTRUCTIONS
# ============================================================================

if [ ! -f "AGENTS.md" ]; then
    echo -e "${BLUE}📝 Creating AGENTS.md instructions...${NC}"
    cat > AGENTS.md <<'MARKDOWN'
# Codex Agent Instructions - gogooku5 Project

You are an autonomous AI developer working on gogooku5, a modular refactoring of the gogooku3 Japanese stock market prediction system.

## Project Context

**Main Goal**: Modular ML pipeline with separation of dataset generation and model training
**Tech Stack**: Python, PyTorch, Polars, RAPIDS/cuDF, Dagster, MLflow
**Hardware**: NVIDIA A100 80GB GPU, 255-core CPU, 1.8TB RAM

## Architecture Overview

gogooku5 separates concerns into independent packages:
- **data/**: Dataset generation (30+ feature modules, GPU-accelerated ETL)
- **models/**: Model-specific packages (APEX-Ranker, ATFT-GAT-FAN)
- **common/**: Shared utilities (minimal, only when >1 consumer)

## Key Project Files

- `data/src/builder/pipelines/dataset_builder.py` - Dataset orchestration
- `data/src/cli/main.py` - CLI interface for dataset generation
- `models/apex_ranker/scripts/train_v0.py` - APEX-Ranker training
- `CLAUDE.md` - Comprehensive project documentation
- `MIGRATION_PLAN.md` - Migration roadmap
- `tools/health-check.sh` - Health diagnostics

## Development Guidelines

1. **Always Read Before Editing**: Use `codex read <file>` to understand context
2. **Test After Changes**: Run tests and validation scripts
3. **Document Changes**: Update relevant documentation
4. **Optimize for GPU**: Leverage A100's 80GB memory for GPU-ETL (RAPIDS/cuDF)
5. **Financial Data Sensitivity**: Handle market data with proper as-of joins (T+1 logic)

## Autonomous Workflow

When working autonomously:
1. Analyze health check reports thoroughly
2. Create detailed todo lists for complex tasks
3. Fix critical issues first (P0 → P1 → P2)
4. Run validation after each major change
5. Document reasoning for non-obvious decisions

## Code Quality Standards

- Type hints for all functions
- Docstrings for public APIs
- Unit tests for core logic
- Memory-efficient data processing (Polars, GPU-ETL)
- Schema validation for datasets

## Useful Commands

```bash
# Health check
tools/health-check.sh

# Dataset generation (from project root)
make -C data build START=2024-01-01 END=2024-12-31

# APEX-Ranker training
make -C models/apex_ranker train

# Dagster UI
export DAGSTER_HOME=/workspace/gogooku3/gogooku5
PYTHONPATH=data/src dagster dev -m dagster_gogooku5.defs

# GPU monitoring
nvidia-smi

# Git status
git status
```

## Notes

- Be proactive about finding optimization opportunities
- Research latest ML/financial modeling techniques when relevant
- Explain complex decisions clearly
- Ask for clarification when requirements are ambiguous
- Respect separation of concerns (dataset ≠ training)
MARKDOWN
    echo -e "${GREEN}✓${NC} Created AGENTS.md"
    echo ""
fi

# ============================================================================
# DYNAMIC ENVIRONMENT DETECTION
# ============================================================================

echo -e "${CYAN}🔍 Detecting environment...${NC}"

# Python environment
PYTHON_VERSION="unknown"
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
fi

# CUDA and PyTorch
CUDA_AVAILABLE="unknown"
TORCH_VERSION="unknown"
if command -v python &> /dev/null; then
    CUDA_AVAILABLE=$(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'unknown')
    if [ "$CUDA_AVAILABLE" = "True" ]; then
        TORCH_VERSION=$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'unknown')
    fi
fi

# GPU information
GPU_NAME="unknown"
GPU_MEMORY="unknown"
GPU_UTILIZATION="unknown"
GPU_MEMORY_USED="unknown"
CUDA_VERSION="unknown"

if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'unknown')
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 'unknown')
    GPU_UTILIZATION=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 'unknown')
    GPU_MEMORY_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 'unknown')
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' || echo 'unknown')
fi

# CPU information
CPU_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 'unknown')
CPU_MODEL=$(lscpu 2>/dev/null | grep "Model name" | cut -d':' -f2 | xargs || sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'unknown')

# Memory information
TOTAL_RAM="unknown"
AVAILABLE_RAM="unknown"
if command -v free &> /dev/null; then
    TOTAL_RAM=$(free -h | awk '/^Mem:/ {print $2}')
    AVAILABLE_RAM=$(free -h | awk '/^Mem:/ {print $7}')
elif command -v vm_stat &> /dev/null; then
    TOTAL_RAM=$(sysctl -n hw.memsize | awk '{printf "%.0fG", $1/1024/1024/1024}')
fi

# Disk space
DISK_USAGE=$(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $5}')
DISK_AVAILABLE=$(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $4}')

# Git information
GIT_BRANCH="unknown"
GIT_COMMIT="unknown"
GIT_DIRTY=""
if command -v git &> /dev/null && [ -d .git ]; then
    GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')
    GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')
    if ! git diff-index --quiet HEAD -- 2>/dev/null; then
        GIT_DIRTY=" (uncommitted changes)"
    fi
fi

# Display environment summary
echo -e "${GREEN}📊 Environment Summary:${NC}"
echo "  Python: $PYTHON_VERSION"
echo "  PyTorch: $TORCH_VERSION"
echo "  CUDA: $CUDA_VERSION (Available: $CUDA_AVAILABLE)"
echo "  GPU: $GPU_NAME"
if [ "$GPU_MEMORY" != "unknown" ]; then
    echo "  GPU Memory: ${GPU_MEMORY_USED}MB / ${GPU_MEMORY}MB (Utilization: ${GPU_UTILIZATION}%)"
fi
echo "  CPU: $CPU_CORES cores"
echo "  RAM: $AVAILABLE_RAM available / $TOTAL_RAM total"
echo "  Disk: $DISK_AVAILABLE available (${DISK_USAGE} used)"
if [ "$GIT_BRANCH" != "unknown" ]; then
    echo "  Git: $GIT_BRANCH @ $GIT_COMMIT${GIT_DIRTY}"
fi
echo ""

# ============================================================================
# PARSE ARGUMENTS
# ============================================================================

SKIP_CHECK=false
QUICK_MODE=false
MAX_MODE=false
EXEC_MODE=false
DISABLE_SANDBOX=true  # デフォルトでサンドボックスを無効化
DISABLE_TERM_WORKAROUND=true  # Issue #4960のワークアラウンドをデフォルトで無効化
INITIAL_PROMPT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-check)
            SKIP_CHECK=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --max)
            MAX_MODE=true
            shift
            ;;
        --exec)
            EXEC_MODE=true
            shift
            if [[ $# -gt 0 ]]; then
                INITIAL_PROMPT="$*"
                break
            fi
            ;;
        --no-sandbox|--dangerously-bypass-sandbox)
            DISABLE_SANDBOX=true
            shift
            ;;
        --sandbox|--enable-sandbox)
            DISABLE_SANDBOX=false
            shift
            ;;
        --no-term-workaround|--disable-term-workaround)
            DISABLE_TERM_WORKAROUND=true
            shift
            ;;
        --term-workaround|--enable-term-workaround)
            DISABLE_TERM_WORKAROUND=false
            shift
            ;;
        *)
            # Any other argument becomes initial prompt for interactive mode
            INITIAL_PROMPT="$*"
            break
            ;;
    esac
done

# ============================================================================
# PROJECT HEALTH CHECK
# ============================================================================

HEALTH_CONTEXT=""
if [ "$SKIP_CHECK" = false ]; then
    echo -e "${YELLOW}🔍 Running project health check...${NC}"

    if [ -x "${PROJECT_ROOT}/tools/health-check.sh" ]; then
        set +e
        "${PROJECT_ROOT}/tools/health-check.sh" > /dev/null 2>&1
        HEALTH_EXIT=$?
        set -e

        if [ $HEALTH_EXIT -eq 0 ]; then
            HEALTH_CONTEXT="
✅ Project health check passed
System is healthy. Consider proactive optimization opportunities."
            echo -e "${GREEN}✅ Project healthy${NC}"
        elif [ $HEALTH_EXIT -eq 1 ]; then
            HEALTH_CONTEXT="
⚠️ WARNINGS DETECTED in health check
Consider reviewing and addressing warnings when convenient."
            echo -e "${YELLOW}⚠️  Warnings detected${NC}"
        else
            HEALTH_CONTEXT="
🚨 CRITICAL ISSUES DETECTED in health check
Please review and address critical issues when appropriate."
            echo -e "${RED}⚠️  Critical issues detected${NC}"
        fi
        echo ""
    fi
fi

# ============================================================================
# MODEL SELECTION
# ============================================================================

if [ "$QUICK_MODE" = true ]; then
    SELECTED_MODEL="gpt-5"
    REASONING_LEVEL="medium"
    echo -e "${CYAN}⚡ Quick Mode:${NC} GPT-5 (medium reasoning)"
elif [ "$MAX_MODE" = true ]; then
    SELECTED_MODEL="gpt-5-codex"
    REASONING_LEVEL="high"
    echo -e "${MAGENTA}🚀 Maximum Power:${NC} GPT-5-Codex (high reasoning)"
else
    # Interactive model selection
    echo -e "${CYAN}🤖 Select AI Model:${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Models (Official Codex CLI Support):"
    echo "  1) GPT-5-Codex (推奨) - Optimized for coding, best for autonomous work"
    echo "  2) GPT-5 - Latest general model, fast reasoning"
    echo "  3) O3 - Advanced reasoning for complex problems"
    echo ""
    echo "Reasoning Levels:"
    echo "  L) Low - Fast, simple tasks"
    echo "  M) Medium (default) - Balanced speed/quality"
    echo "  H) High - Deep reasoning, complex tasks"
    echo ""

    read -p "Choose model (1-3) [1]: " MODEL_CHOICE
    MODEL_CHOICE=${MODEL_CHOICE:-1}

    case "$MODEL_CHOICE" in
        1) SELECTED_MODEL="gpt-5-codex" ;;
        2) SELECTED_MODEL="gpt-5" ;;
        3) SELECTED_MODEL="o3" ;;
        *) SELECTED_MODEL="gpt-5-codex" ;;
    esac

    read -p "Choose reasoning level (L/M/H) [M]: " REASONING_CHOICE
    REASONING_CHOICE=${REASONING_CHOICE:-M}

    case "${REASONING_CHOICE^^}" in
        L) REASONING_LEVEL="low" ;;
        M) REASONING_LEVEL="medium" ;;
        H) REASONING_LEVEL="high" ;;
        *) REASONING_LEVEL="medium" ;;
    esac

    echo ""
    echo -e "${GREEN}✅ Selected:${NC} $SELECTED_MODEL (reasoning: $REASONING_LEVEL)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
fi

# ============================================================================
# PREPARE ENVIRONMENT CONTEXT FOR AGENTS.MD
# ============================================================================

# Create a temporary environment info file that AGENTS.md can reference
ENV_INFO_FILE="${PROJECT_ROOT}/.codex-env-info.md"
cat > "$ENV_INFO_FILE" <<EOF
# Current Environment Information (gogooku5)
# Auto-generated by Codex Launcher - $(date)

## Hardware & Software
- **Python**: $PYTHON_VERSION
- **PyTorch**: $TORCH_VERSION (CUDA Available: $CUDA_AVAILABLE)
- **GPU**: $GPU_NAME
- **CUDA Version**: $CUDA_VERSION
EOF

if [ "$GPU_MEMORY" != "unknown" ]; then
    FREE_GPU_MEM=$((GPU_MEMORY - GPU_MEMORY_USED))
    cat >> "$ENV_INFO_FILE" <<EOF
- **GPU Memory**: ${GPU_MEMORY_USED}MB used / ${GPU_MEMORY}MB total (${GPU_UTILIZATION}% utilization)
- **Available GPU Memory**: ${FREE_GPU_MEM}MB
EOF
fi

cat >> "$ENV_INFO_FILE" <<EOF
- **CPU**: $CPU_CORES cores ($CPU_MODEL)
- **RAM**: $AVAILABLE_RAM available / $TOTAL_RAM total
- **Disk**: $DISK_AVAILABLE available (${DISK_USAGE} used)
EOF

if [ "$GIT_BRANCH" != "unknown" ]; then
    cat >> "$ENV_INFO_FILE" <<EOF
- **Git Branch**: $GIT_BRANCH @ $GIT_COMMIT${GIT_DIRTY}
EOF
fi

cat >> "$ENV_INFO_FILE" <<EOF

## Optimization Opportunities
EOF

# Add optimization suggestions
if [ "$GPU_UTILIZATION" != "unknown" ] && [ "$GPU_UTILIZATION" -lt 50 ]; then
    echo "- GPU utilization is ${GPU_UTILIZATION}% - consider increasing batch size or enabling GPU-ETL" >> "$ENV_INFO_FILE"
fi

if [ "$GPU_MEMORY" != "unknown" ] && [ "$GPU_MEMORY_USED" != "unknown" ]; then
    FREE_MEM=$((GPU_MEMORY - GPU_MEMORY_USED))
    if [ $FREE_MEM -gt 20000 ]; then
        echo "- ${FREE_MEM}MB GPU memory available - can support larger models or batch sizes" >> "$ENV_INFO_FILE"
    fi
fi

if [ "$CUDA_VERSION" != "unknown" ]; then
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d'.' -f1)
    if [ "$CUDA_MAJOR" -ge 12 ]; then
        echo "- CUDA $CUDA_VERSION detected - RAPIDS/cuDF GPU-ETL available" >> "$ENV_INFO_FILE"
    fi
fi

cat >> "$ENV_INFO_FILE" <<EOF
$HEALTH_CONTEXT

## Project Structure (gogooku5)
- **Root**: $PROJECT_ROOT
- **Dataset Builder**: data/src/builder/pipelines/dataset_builder.py
- **CLI Interface**: data/src/cli/main.py
- **APEX-Ranker**: models/apex_ranker/
- **Documentation**: CLAUDE.md, AGENTS.md, MIGRATION_PLAN.md
- **Health Check**: tools/health-check.sh

## Key Workflows

### Dataset Generation
\`\`\`bash
# From project root
make -C data build START=2024-01-01 END=2024-12-31

# Or with CLI
PYTHONPATH=data/src python -m cli.main build --start 2024-01-01 --end 2024-12-31
\`\`\`

### Model Training (APEX-Ranker)
\`\`\`bash
make -C models/apex_ranker train
\`\`\`

### Dagster Orchestration
\`\`\`bash
export DAGSTER_HOME=/workspace/gogooku3/gogooku5
PYTHONPATH=data/src dagster dev -m dagster_gogooku5.defs
\`\`\`

---
*Note: Read .codex-env-info.md for current environment details*
EOF

# ============================================================================
# APPROVAL MODE SELECTION
# ============================================================================

# Note: Codex CLI 0.47.0 doesn't support approval mode flags on command line
# Use /approvals command during session to change modes
# Default is "auto" mode (balanced)
APPROVAL_MODE=""
if [ "$EXEC_MODE" = true ]; then
    APPROVAL_DESC="Auto (default, will prompt for risky operations)"
else
    APPROVAL_DESC="Auto (default) - Use /approvals in session to change"
fi

# ============================================================================
# BUILD CODEX COMMAND
# ============================================================================

# Always disable OSC color queries (Issue #4960) unless explicitly overridden.
: "${CODEX_DISABLE_OSC_COLOR_QUERY:=1}"
export CODEX_DISABLE_OSC_COLOR_QUERY

# Session logging
SESSION_LOG="${LOG_DIR}/session-$(date +%Y%m%d-%H%M%S).log"

# Sandbox configuration
SANDBOX_ARGS=""
SANDBOX_STATUS_MSG=""
if [ "$DISABLE_SANDBOX" = true ]; then
    SANDBOX_ARGS="--dangerously-bypass-approvals-and-sandbox"
    SANDBOX_STATUS_MSG="${RED}DISABLED (default)${NC}"
else
    SANDBOX_STATUS_MSG="${GREEN}ENABLED${NC}"
    echo -e "${GREEN}✅ Sandbox enabled (all commands will require approval)${NC}"
    echo ""
fi

# ============================================================================
# WORKAROUND FOR ISSUE #4960: Terminal Output Corruption
# ============================================================================
#
# Issue: Codex CLI causes terminal to display repeated OSC (Operating System Command)
#        color query responses like "11;rgb:0404/0404/0404" as visible garbage text.
#
# Root Cause: Codex issues OSC 10/11 terminal color queries to detect theme colors.
#             Some terminals (macOS Terminal, SSH sessions, tmux) echo the responses
#             as plain text instead of suppressing them.
#
# Solution: Set terminal type to 'dumb' or disable color detection
#
# References:
# - https://github.com/openai/codex/issues/4960
# - https://github.com/openai/codex/issues/4945

# Apply workaround only if explicitly enabled
if [ "$DISABLE_TERM_WORKAROUND" = false ]; then
    # Save original TERM value
    ORIGINAL_TERM="${TERM:-xterm-256color}"

    # Apply workaround for known problematic terminals
    case "$ORIGINAL_TERM" in
        xterm*|screen*|tmux*|vt100*|rxvt*)
            # These terminals may exhibit OSC response corruption
            # Temporarily set to 'xterm' without color query support indicators
            export TERM="xterm"
            export COLORTERM=""  # Disable true color detection
            echo -e "${GREEN}✅ Terminal workaround enabled (Issue #4960):${NC}"
            echo -e "${GREEN}   TERM changed from '$ORIGINAL_TERM' to '$TERM' to prevent output corruption${NC}"
            echo ""
            ;;
        dumb)
            # Already safe
            echo -e "${GREEN}✅ Terminal is already safe (TERM=dumb)${NC}"
            echo ""
            ;;
        *)
            # Unknown terminal, keep original
            ;;
    esac

    # Additional environment variables to suppress color queries
    export TERM_PROGRAM_BACKGROUND=""   # Disable macOS Terminal background detection
    export COLORFGBG=""                 # Disable color detection via COLORFGBG
fi

# ============================================================================
# LAUNCH CODEX
# ============================================================================

echo ""
echo -e "${CYAN}🚀 Launching OpenAI Codex CLI${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "  Model: ${MAGENTA}$SELECTED_MODEL${NC} (reasoning: $REASONING_LEVEL)"
echo -e "  Mode: ${GREEN}Interactive${NC}"
echo -e "  Approval Mode: ${YELLOW}$APPROVAL_DESC${NC}"
echo -e "  Sandbox: $SANDBOX_STATUS_MSG"
echo -e "  Session Log: ${BLUE}$SESSION_LOG${NC}"
echo -e "  MCP Servers: ${GREEN}playwright, filesystem, git, brave-search${NC}"
if [ -f "AGENTS.md" ]; then
    echo -e "  Instructions: ${GREEN}AGENTS.md + .codex-env-info.md loaded${NC}"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ "$EXEC_MODE" = true ]; then
    echo -e "${MAGENTA}⚠️  Exec Mode (Non-Interactive)${NC}"
    if [ -n "$INITIAL_PROMPT" ]; then
        echo -e "${BLUE}Prompt: ${INITIAL_PROMPT}${NC}"
    fi
    echo ""
    # Use codex exec for non-interactive mode (with logging)
    if [ -n "$INITIAL_PROMPT" ]; then
        codex exec --model "$SELECTED_MODEL" $SANDBOX_ARGS "$INITIAL_PROMPT" 2>&1 | tee "$SESSION_LOG"
    else
        echo -e "${RED}ERROR: --exec requires a prompt${NC}"
        exit 1
    fi
else
    echo "💡 Tips:"
    echo "  - Type your requests naturally in Japanese or English"
    echo "  - Use /model to change model during session"
    echo "  - Use /approvals to change approval mode (suggest/auto/full)"
    echo "  - Type 'read .codex-env-info.md' to see current environment details"
    if [ -n "$HEALTH_CONTEXT" ]; then
        echo "  - Health check results are available - ask about them if needed"
    fi
    echo "  - Press Ctrl+C to cancel current operation"
    echo "  - Type 'exit' to quit"
    echo ""
    echo -e "${CYAN}Note: Session will be logged to $SESSION_LOG${NC}"
    echo ""

    # Interactive mode - direct execution without tee (tee breaks interactive mode)
    if [ -n "$INITIAL_PROMPT" ]; then
        echo -e "${BLUE}Starting with initial prompt: ${INITIAL_PROMPT}${NC}"
        echo ""
        exec codex --model "$SELECTED_MODEL" $SANDBOX_ARGS "$INITIAL_PROMPT"
    else
        exec codex --model "$SELECTED_MODEL" $SANDBOX_ARGS
    fi
fi
