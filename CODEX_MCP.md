# Codex MCP Setup for gogooku3-standalone

## Overview
This project is configured to use Codex with project-specific MCP (Model Context Protocol) servers.

## Setup Complete ✅

### Files Created:
- `.mcp.json` - Project-specific MCP server configuration
- `codex-mcp.sh` - Launcher script for Codex with MCP support
- `.bashrc` updated with convenient aliases

## Usage

### Method 1: Using the launcher script
```bash
./codex-mcp.sh
```

### Method 2: Using the alias (after sourcing .bashrc)
```bash
source ~/.bashrc
codex-mcp
```

### Method 3: Direct command
```bash
codex -c "mcp_servers=$(jq .mcpServers .mcp.json -cM | sed 's/\":/\"=/g')"
```

## Available MCP Servers

### Configured in `.mcp.json`:
- **playwright** - Browser automation MCP server

### Additional servers via alias:
- **serena** - Code analysis tool (use `codex-serena` alias)

## Adding More MCP Servers

Edit `.mcp.json` to add more MCP servers:

```json
{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": ["-y", "@playwright/mcp@latest"]
    },
    "your-server": {
      "command": "your-command",
      "args": ["arg1", "arg2"]
    }
  }
}
```

## Notes
- The configuration is project-specific and won't affect global Codex settings
- `.mcp.json` should be added to `.gitignore` if it contains sensitive information
- The launcher script will create a default `.mcp.json` if it doesn't exist