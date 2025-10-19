#!/bin/bash
# Claude Code launcher with full autonomous mode
# Usage:
#   ./claude-code.sh                    # Auto-diagnose and fix issues
#   ./claude-code.sh --no-check         # Skip health check
#   ./claude-code.sh --interactive      # Standard interactive mode
#   ./claude-code.sh <prompt>           # Direct prompt
#
# AUTONOMOUS FEATURES:
#   - Automatic project health diagnosis on startup
#   - Full bypass permissions (all operations auto-approved)
#   - Project-aware system prompt
#   - MCP servers: playwright, filesystem, git, brave-search

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

# Parse arguments
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
    # Running as root - use acceptEdits mode (bypassPermissions not allowed with root)
    PERMISSION_MODE="acceptEdits"
    PERMISSION_DESC="Auto-accept edits (root user)"
else
    # Non-root user - use full bypass
    PERMISSION_MODE="bypassPermissions"
    PERMISSION_DESC="Full bypass permissions"
fi

# Prepare system prompt
SYSTEM_PROMPT="You are an autonomous AI developer working on the ATFT-GAT-FAN project (Japanese stock market prediction using Graph Attention Networks).

Key project details:
- Hardware: NVIDIA A100 80GB, 24-core CPU, 216GB RAM
- Main codebase: /workspace/gogooku3
- Training pipeline: scripts/integrated_ml_training_pipeline.py
- Dataset builder: scripts/pipelines/run_full_dataset.py
- See CLAUDE.md for full project documentation

Your autonomous workflow:
1. Use TodoWrite to track all tasks (REQUIRED - always create todos)
2. Read relevant files before making changes
3. Fix issues systematically (critical → warnings → optimizations)
4. Verify fixes by running health checks or tests
5. Report completion status clearly

Tools at your disposal:
- Read/Write/Edit for file operations
- Bash for running commands (training, tests, health checks)
- Grep/Glob for code exploration
- Task tool for complex multi-step operations
- MCP servers: playwright (web), filesystem, git, brave-search

Be proactive, thorough, and autonomous. Fix all issues you discover."

# Launch claude
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
    echo "  - Project-aware system prompt"
    echo "  - MCP servers: playwright, filesystem, git, brave-search"
    echo ""
    exec claude --permission-mode "$PERMISSION_MODE" --append-system-prompt "$SYSTEM_PROMPT" "$AUTO_PROMPT"
else
    echo "🚀 Launching Claude Code (default mode)"
    echo "  - $PERMISSION_DESC"
    echo ""
    exec claude --permission-mode "$PERMISSION_MODE" --append-system-prompt "$SYSTEM_PROMPT" "$@"
fi
