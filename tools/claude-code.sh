#!/bin/bash
# Enhanced Claude Code launcher with dynamic environment detection
# Usage:
#   ./claude-code-enhanced.sh                    # Auto-diagnose and fix issues
#   ./claude-code-enhanced.sh --no-check         # Skip health check
#   ./claude-code-enhanced.sh --interactive      # Standard interactive mode
#   ./claude-code-enhanced.sh <prompt>           # Direct prompt

set -euo pipefail

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Check claude is installed
if ! command -v claude &> /dev/null; then
    echo "ERROR: claude is not installed. Please install it first." >&2
    exit 127
fi

# Setup MCP configuration if not exists
if [ ! -f ".mcp.json" ]; then
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
    echo "✓ Created .mcp.json"
fi

# ============================================================================
# 🚀 ENHANCED: Dynamic Environment Detection
# ============================================================================

echo "🔍 Detecting environment..."

# Python environment
PYTHON_VERSION="unknown"
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
fi

# CUDA availability
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
CPU_MODEL=$(lscpu 2>/dev/null | grep "Model name" | cut -d':' -f2 | xargs || echo 'unknown')

# Memory information
TOTAL_RAM="unknown"
AVAILABLE_RAM="unknown"
if command -v free &> /dev/null; then
    TOTAL_RAM=$(free -h | awk '/^Mem:/ {print $2}')
    AVAILABLE_RAM=$(free -h | awk '/^Mem:/ {print $7}')
elif command -v vm_stat &> /dev/null; then
    # macOS
    TOTAL_RAM=$(sysctl -n hw.memsize | awk '{print $1/1024/1024/1024 "G"}')
fi

# Disk space
DISK_USAGE=$(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $5}')
DISK_AVAILABLE=$(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $4}')

# Git information
GIT_BRANCH="unknown"
GIT_COMMIT="unknown"
if command -v git &> /dev/null && [ -d .git ]; then
    GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')
    GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')
fi

# Display environment summary
echo "📊 Environment Summary:"
echo "  Python: $PYTHON_VERSION"
echo "  PyTorch: $TORCH_VERSION"
echo "  CUDA: $CUDA_VERSION (Available: $CUDA_AVAILABLE)"
echo "  GPU: $GPU_NAME"
echo "  GPU Memory: ${GPU_MEMORY_USED}MB / ${GPU_MEMORY}MB (Utilization: ${GPU_UTILIZATION}%)"
echo "  CPU: $CPU_CORES cores ($CPU_MODEL)"
echo "  RAM: $AVAILABLE_RAM available / $TOTAL_RAM total"
echo "  Disk: $DISK_AVAILABLE available (${DISK_USAGE} used)"
echo "  Git: $GIT_BRANCH @ $GIT_COMMIT"
echo ""

# ============================================================================
# Parse arguments
# ============================================================================

SKIP_CHECK=false
INTERACTIVE_MODE=false
USER_PROMPT=""

for arg in "$@"; do
    case "$arg" in
        --no-check)
            SKIP_CHECK=true
            shift
            ;;
        --interactive)
            INTERACTIVE_MODE=true
            shift
            ;;
        *)
            USER_PROMPT="$*"
            break
            ;;
    esac
done

# Run health check and generate autonomous prompt
AUTO_PROMPT=""
if [ "$SKIP_CHECK" = false ] && [ "$INTERACTIVE_MODE" = false ] && [ -z "$USER_PROMPT" ]; then
    echo "🔍 Running autonomous project health check..."

    if [ -x "${PROJECT_ROOT}/tools/project-health-check.sh" ]; then
        set +e
        "${PROJECT_ROOT}/tools/project-health-check.sh" > /dev/null 2>&1
        HEALTH_EXIT=$?
        set -e

        # Find latest health report
        LATEST_REPORT=$(ls -t _logs/health-checks/health-check-*.json 2>/dev/null | head -1)

        if [ -n "$LATEST_REPORT" ] && [ -f "$LATEST_REPORT" ]; then
            echo "📊 Health report: $LATEST_REPORT"

            # Generate autonomous prompt based on health status
            if [ $HEALTH_EXIT -eq 2 ]; then
                AUTO_PROMPT="🚨 AUTONOMOUS MODE: Critical issues detected.

Health Report: $LATEST_REPORT

Your mission:
1. Read and analyze the health check report using the Read tool
2. Fix ALL critical issues immediately (priority: P0)
3. Address warnings and recommendations (priority: P1-P2)
4. Run verification: tools/project-health-check.sh
5. Report completion status

Start by reading the health report with the Read tool, create a todo list, and systematically fix all issues."
            elif [ $HEALTH_EXIT -eq 1 ]; then
                AUTO_PROMPT="⚠️ AUTONOMOUS MODE: Warnings detected.

Health Report: $LATEST_REPORT

Your mission:
1. Read and analyze the health check report
2. Investigate warnings and determine root causes
3. Implement fixes and optimizations
4. Verify improvements: tools/project-health-check.sh
5. Report what was improved

Start by reading the health report and prioritizing improvements."
            else
                AUTO_PROMPT="✅ AUTONOMOUS MODE: Project healthy. Proactive optimization.

Health Report: $LATEST_REPORT

Your mission:
1. Review recent git commits and training logs
2. Identify performance bottlenecks or technical debt
3. Research and implement improvements (code quality, performance, architecture)
4. Run tests to ensure no regressions
5. Document improvements made

Start by exploring the codebase for optimization opportunities."
            fi

            echo ""
            echo "🤖 Autonomous improvement mode activated"
            echo ""
        fi
    fi
fi

# Determine permission mode based on user
if [ "$(id -u)" -eq 0 ]; then
    PERMISSION_MODE="acceptEdits"
    PERMISSION_DESC="Auto-accept edits (root user)"
else
    PERMISSION_MODE="bypassPermissions"
    PERMISSION_DESC="Full bypass permissions"
fi

# ============================================================================
# 🚀 ENHANCED: Prepare enriched system prompt
# ============================================================================

SYSTEM_PROMPT="You are an autonomous AI developer working on the ATFT-GAT-FAN project (Japanese stock market prediction using Graph Attention Networks).

🖥️ CURRENT ENVIRONMENT:
- Python: $PYTHON_VERSION
- PyTorch: $TORCH_VERSION (CUDA: $CUDA_AVAILABLE)
- GPU: $GPU_NAME
  * CUDA Version: $CUDA_VERSION
  * Memory: ${GPU_MEMORY_USED}MB used / ${GPU_MEMORY}MB total (${GPU_UTILIZATION}% utilization)
  * Available Memory: $((GPU_MEMORY - GPU_MEMORY_USED))MB
- CPU: $CPU_CORES cores ($CPU_MODEL)
- RAM: $AVAILABLE_RAM available / $TOTAL_RAM total
- Disk: $DISK_AVAILABLE available (${DISK_USAGE} used)
- Git: $GIT_BRANCH @ $GIT_COMMIT

📁 PROJECT STRUCTURE:
- Main codebase: $PROJECT_ROOT
- Training pipeline: scripts/integrated_ml_training_pipeline.py
- Dataset builder: scripts/pipelines/run_full_dataset.py
- See CLAUDE.md for full project documentation

💡 OPTIMIZATION OPPORTUNITIES:
$(if [ "$GPU_UTILIZATION" != "unknown" ] && [ "$GPU_UTILIZATION" -lt 50 ]; then
    echo "- GPU utilization is ${GPU_UTILIZATION}% - consider increasing batch size or model complexity"
fi)
$(if [ "$GPU_MEMORY" != "unknown" ] && [ "$GPU_MEMORY_USED" != "unknown" ]; then
    FREE_MEM=$((GPU_MEMORY - GPU_MEMORY_USED))
    if [ $FREE_MEM -gt 20000 ]; then
        echo "- ${FREE_MEM}MB GPU memory available - can support larger models or batch sizes"
    fi
fi)
$(if [ "$CUDA_VERSION" != "unknown" ]; then
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d'.' -f1)
    if [ "$CUDA_MAJOR" -ge 12 ]; then
        echo "- CUDA $CUDA_VERSION detected - FlashAttention 2 and other optimizations available"
    fi
fi)

🔧 YOUR AUTONOMOUS WORKFLOW:
1. Use TodoWrite to track all tasks (REQUIRED - always create todos)
2. Read relevant files before making changes
3. Fix issues systematically (critical → warnings → optimizations)
4. Verify fixes by running health checks or tests
5. Report completion status clearly

🛠️ TOOLS AT YOUR DISPOSAL:
- Read/Write/Edit for file operations
- Bash for running commands (training, tests, health checks)
- Grep/Glob for code exploration
- Task tool for complex multi-step operations
- MCP servers: playwright (web), filesystem, git, brave-search

Be proactive, thorough, and autonomous. Use the environment information above to make data-driven optimization decisions."

# ============================================================================
# Launch claude with enhanced configuration
# ============================================================================

if [ "$INTERACTIVE_MODE" = true ]; then
    echo "🚀 Launching Claude Code (interactive mode)"
    echo ""
    exec claude "$@"
elif [ -n "$USER_PROMPT" ]; then
    echo "🚀 Launching Claude Code with user prompt"
    echo "  - $PERMISSION_DESC"
    echo ""
    exec claude --permission-mode "$PERMISSION_MODE" --append-system-prompt "$SYSTEM_PROMPT" "$USER_PROMPT"
elif [ -n "$AUTO_PROMPT" ]; then
    echo "🚀 Launching Claude Code (autonomous mode)"
    echo "  - $PERMISSION_DESC"
    echo "  - Environment-aware system prompt"
    echo "  - MCP servers: playwright, filesystem, git, brave-search"
    echo ""
    exec claude --permission-mode "$PERMISSION_MODE" --append-system-prompt "$SYSTEM_PROMPT" "$AUTO_PROMPT"
else
    echo "🚀 Launching Claude Code (default mode)"
    echo "  - $PERMISSION_DESC"
    echo ""
    exec claude --permission-mode "$PERMISSION_MODE" --append-system-prompt "$SYSTEM_PROMPT" "$@"
fi
