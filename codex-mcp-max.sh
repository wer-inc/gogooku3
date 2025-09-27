#!/bin/bash
# Codex launcher with maximum capabilities (network enabled, no approvals, full FS access)
# Usage:
#   ./codex-mcp-max.sh            # open Codex with max permissions
#   ./codex-mcp-max.sh <args...>  # pass-through extra args

set -euo pipefail

# Ensure project-local MCP config exists
if [ ! -f ".mcp.json" ]; then
  cat > .mcp.json <<'JSON'
{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": ["-y", "@playwright/mcp@latest"]
    }
  }
}
JSON
fi

# Compose MCP servers config for Codex (-c "key=value")
MCP_CFG=$(jq .mcpServers .mcp.json -cM | sed 's/\":/\"=/g')

# Optional corporate proxy passthrough (export before running this script if needed)
# export HTTPS_PROXY="https://proxy.example.com:8443"
# export HTTP_PROXY="http://proxy.example.com:8080"
# export NO_PROXY="localhost,127.0.0.1,::1"

echo "Launching Codex with:"
echo "  - web search=enabled"
echo "  - sandbox=danger-full-access"
echo "  - approvals=never"
echo "  - model=gpt-5"

codex \
  --sandbox danger-full-access \
  -a never \
  --search \
  -c "mcp_servers=${MCP_CFG}" \
  -m "gpt-5" \
  "$@"
