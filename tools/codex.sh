#!/bin/bash
# Codex launcher with full autonomous mode
# Usage:
#   ./codex.sh                    # Auto-diagnose and fix issues
#   ./codex.sh --no-check         # Skip health check
#   ./codex.sh <prompt>           # Direct prompt
#
# AUTONOMOUS FEATURES:
#   - Automatic project health diagnosis on startup
#   - Web search enabled (--search)
#   - Full auto mode (--full-auto: -a on-failure, workspace sandbox)
#   - Project-aware system prompt
#   - MCP servers: playwright, filesystem, git, brave-search

set -euo pipefail

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Check codex is installed
if ! command -v codex &> /dev/null; then
    echo "ERROR: codex is not installed. Please install it first." >&2
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
USER_PROMPT=""

for arg in "$@"; do
    case "$arg" in
        --no-check)
            SKIP_CHECK=true
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
if [ "$SKIP_CHECK" = false ] && [ -z "$USER_PROMPT" ]; then
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
                AUTO_PROMPT="🚨 AUTONOMOUS MODE (Deep Reasoning): Critical issues detected.

Health Report: $LATEST_REPORT

Your mission:
1. Read and deeply analyze the health check report
2. Think step-by-step about root causes
3. Fix ALL critical issues with careful consideration
4. Address warnings and optimizations
5. Verify fixes: tools/project-health-check.sh
6. Explain your reasoning and decisions

Use your o1 reasoning capabilities to solve complex issues systematically."
            elif [ $HEALTH_EXIT -eq 1 ]; then
                AUTO_PROMPT="⚠️ AUTONOMOUS MODE (Deep Reasoning): Warnings detected.

Health Report: $LATEST_REPORT

Your mission:
1. Analyze warnings and their implications
2. Research best practices and solutions
3. Design and implement optimal fixes
4. Verify improvements: tools/project-health-check.sh
5. Document your reasoning

Apply deep reasoning to find the best solutions."
            else
                AUTO_PROMPT="✅ AUTONOMOUS MODE (Deep Reasoning): Proactive optimization.

Health Report: $LATEST_REPORT

Your mission:
1. Review codebase architecture and recent changes
2. Research latest ML/financial modeling techniques
3. Identify advanced optimization opportunities
4. Design and implement improvements
5. Run comprehensive tests
6. Document technical decisions

Use your reasoning skills to find non-obvious improvements."
            fi

            echo ""
            echo "🤖 Autonomous deep reasoning mode activated"
            echo ""
        fi
    fi
fi

# Prepare system prompt (codex doesn't support --append-system-prompt, so we prepend to user prompt)
SYSTEM_CONTEXT="[SYSTEM CONTEXT - ATFT-GAT-FAN Project]
You are an autonomous AI developer with deep reasoning capabilities working on a Japanese stock market prediction system using Graph Attention Networks.

Project details:
- Hardware: NVIDIA A100 80GB, 24-core CPU, 216GB RAM
- Main codebase: /workspace/gogooku3
- Training: scripts/integrated_ml_training_pipeline.py
- Dataset: scripts/pipelines/run_full_dataset.py
- Documentation: CLAUDE.md

Your autonomous workflow:
1. Think deeply before acting - use your o1 reasoning
2. Create detailed todo lists (TodoWrite tool)
3. Read files before editing
4. Run tests and validations
5. Explain your reasoning clearly

Tools: Read/Write/Edit, Bash, Grep/Glob, Task, Web search, MCP servers

Be thorough, reason deeply, and autonomous.

---

"

# Launch codex
if [ -n "$USER_PROMPT" ]; then
    echo "🚀 Launching Codex with user prompt"
    echo ""
    exec codex --search --full-auto "${SYSTEM_CONTEXT}${USER_PROMPT}"
elif [ -n "$AUTO_PROMPT" ]; then
    echo "🚀 Launching Codex (autonomous deep reasoning mode)"
    echo "  - Web search enabled (--search)"
    echo "  - Full auto mode (--full-auto)"
    echo "  - Project-aware context"
    echo "  - MCP servers: playwright, filesystem, git, brave-search"
    echo ""
    exec codex --search --full-auto "${SYSTEM_CONTEXT}${AUTO_PROMPT}"
else
    echo "🚀 Launching Codex (default mode)"
    echo ""
    exec codex --search --full-auto "$@"
fi
