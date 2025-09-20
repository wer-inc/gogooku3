#!/bin/bash
# Codex MCP launcher for gogooku3-standalone project

# Check if .mcp.json exists
if [ ! -f ".mcp.json" ]; then
    echo "Error: .mcp.json not found in current directory"
    echo "Creating default .mcp.json..."
    cat > .mcp.json <<EOF
{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": ["-y", "@playwright/mcp@latest"]
    }
  }
}
EOF
    echo "Created .mcp.json with playwright MCP server"
fi

# Launch Codex with MCP servers from .mcp.json
echo "Starting Codex with MCP servers from .mcp.json..."
echo "Available MCP servers:"
codex -c "mcp_servers=$(jq .mcpServers .mcp.json -cM | sed 's/\":/\"=/g')" mcp list

echo ""
echo "Launching Codex with MCP support..."
codex -c "mcp_servers=$(jq .mcpServers .mcp.json -cM | sed 's/\":/\"=/g')" "$@"