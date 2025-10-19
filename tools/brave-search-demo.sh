#!/bin/bash
# Brave Search MCP Demo and Usage Examples
# This script demonstrates how to use the newly added Brave Search MCP

set -euo pipefail

cat <<'EOF'
╔═══════════════════════════════════════════════════════════════════╗
║           Brave Search MCP - Usage Examples                       ║
╔═══════════════════════════════════════════════════════════════════╝

Brave Search MCP has been added to provide advanced web search capabilities
beyond the built-in search functionality.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 AVAILABLE SEARCH METHODS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Built-in Web Search (--search flag)
   ✓ Always enabled
   ✓ No API key required
   ✓ Good for general research

2. Playwright MCP
   ✓ Browser automation
   ✓ Can scrape and interact with websites
   ✓ Good for dynamic content

3. Brave Search API (NEW!)
   ✓ Programmatic search API
   ✓ High-quality search results
   ✓ Good for structured queries
   ⚠️ Requires BRAVE_API_KEY (optional)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔧 SETUP (OPTIONAL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Brave Search MCP can work without an API key (limited functionality),
but for full features, you can set up an API key:

1. Get a Brave Search API key:
   https://brave.com/search/api/

2. Set environment variable:
   export BRAVE_API_KEY="your_api_key_here"

   Or add to .env:
   echo "BRAVE_API_KEY=your_api_key_here" >> .env

3. Restart your terminal or reload .env

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 USAGE EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Example 1: Research Latest Technologies
────────────────────────────────────────

  ./tools/claude-code.sh

  Prompt:
  "Use Brave Search to find the latest PyTorch 2.x features announced
   in 2024-2025. Compare them with what we're currently using in this
   project and suggest which new features we should adopt."


Example 2: API Documentation Research
──────────────────────────────────────

  ./tools/claude-code.sh

  Prompt:
  "Search for JQuants API updates and new endpoints released in the
   last 6 months. Check if there are any new data sources we're not
   using yet."


Example 3: Competitive Analysis
────────────────────────────────

  ./tools/codex.sh

  Prompt:
  "Research and compare machine learning architectures used by top
   financial prediction systems. Use Brave Search to find recent papers
   and implementations. Create a comparison report."


Example 4: Code Pattern Research
─────────────────────────────────

  ./tools/claude-code.sh

  Prompt:
  "Search for best practices in implementing Graph Attention Networks
   for time-series financial data. Find recent GitHub repositories and
   research papers. Suggest improvements to our current implementation."


Example 5: Bug Investigation
─────────────────────────────

  ./tools/claude-code.sh

  Prompt:
  "I'm getting a 'CUDA out of memory' error with PyTorch. Search for
   solutions and optimization techniques specific to Graph Attention
   Networks on A100 GPUs. Suggest fixes for our codebase."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 WHEN TO USE EACH METHOD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Built-in Search:
  ✓ Quick fact-checking
  ✓ General questions
  ✓ News and current events

Playwright:
  ✓ Scraping specific websites
  ✓ Interacting with web forms
  ✓ Testing web applications
  ✓ Dynamic content that requires JavaScript

Brave Search API:
  ✓ Structured search queries
  ✓ Academic/research papers
  ✓ Technical documentation
  ✓ Code repository searches
  ✓ When you need precise, high-quality results

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ VERIFICATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Check that Brave Search MCP is properly configured:

EOF

echo "Checking MCP configuration..."
if [ -f ".mcp.json" ]; then
    if jq -e '.mcpServers."brave-search"' .mcp.json >/dev/null 2>&1; then
        echo "✅ Brave Search MCP is configured in .mcp.json"
        jq -C '.mcpServers."brave-search"' .mcp.json
    else
        echo "❌ Brave Search MCP is NOT configured in .mcp.json"
        exit 1
    fi
else
    echo "⚠️  .mcp.json not found (will be created on first launcher run)"
fi

echo ""
echo "Environment variables:"
if [ -n "${BRAVE_API_KEY:-}" ]; then
    echo "✅ BRAVE_API_KEY is set (${#BRAVE_API_KEY} characters)"
else
    echo "⚠️  BRAVE_API_KEY is not set (API will work with limited functionality)"
fi

cat <<'EOF'

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 ADDITIONAL RESOURCES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Brave Search API Docs: https://brave.com/search/api/
- MCP Server Docs: https://github.com/modelcontextprotocol
- Project README: tools/README.md

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ready to start? Run:
  ./tools/claude-code.sh  (for general tasks)
  ./tools/codex.sh        (for complex reasoning)

╚═══════════════════════════════════════════════════════════════════╝
EOF
