#!/bin/bash
# Codex MCP launcher for gogooku3-standalone project

# Check if .mcp.json exists
if [ ! -f ".mcp.json" ]; then
    echo "Error: .mcp.json not found in current directory"
    echo "Creating default .mcp.json with MAXIMUM PERMISSIONS..."
    cat > .mcp.json <<EOF
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
EOF
    echo "Created .mcp.json with MAXIMUM PERMISSIONS MCP servers"
fi

# Launch Codex with MCP servers from .mcp.json
echo "Starting Codex with MAXIMUM PERMISSIONS MCP servers from .mcp.json..."
echo "Available MCP servers:"
codex -c "mcp_servers=$(jq .mcpServers .mcp.json -cM | sed 's/\":/\"=/g')" mcp list

echo ""
echo "Launching Codex with MAXIMUM PERMISSIONS:"
echo "  - web search=enabled"
echo "  - sandbox=danger-full-access"
echo "  - approvals=never"
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
  -c "mcp_servers=$(jq .mcpServers .mcp.json -cM | sed 's/\":/\"=/g')" \
  "$@"