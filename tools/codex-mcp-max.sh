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
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem@latest", "/"]
    },
    "git": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-git@latest", "--repository", "."]
    },
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search@latest"]
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres@latest"]
    },
    "sqlite": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sqlite@latest"]
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

echo "Launching Codex with MAXIMUM PERMISSIONS:"
echo "  - web search=enabled"
echo "  - sandbox=danger-full-access"
echo "  - approvals=never"
echo "  - model=gpt-5"
echo "  - MCP servers=playwright, filesystem, git, brave-search, postgres, sqlite"
echo "  - full filesystem access"
echo "  - network access enabled"
echo "  - all tool permissions granted"

codex \
  --sandbox danger-full-access \
  -a never \
  --search \
  --allow-all-tools \
  --allow-network \
  --allow-filesystem \
  -c "mcp_servers=${MCP_CFG}" \
  -m "gpt-5" \
  "$@"
